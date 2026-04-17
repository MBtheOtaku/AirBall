import json
import os
import sys
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import ollama
import mediapipe as mp
from fastapi import APIRouter, HTTPException, UploadFile, File, status

# Allow importing shot_detector from the Server root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shot_detector import ShotDetector
from ball_detector import detect_ball_state

router = APIRouter(prefix="/analyze", tags=["analysis"])

LLM_MODEL = "gemma3:1b"
POSE_MIN_DETECTION_CONFIDENCE = float(os.getenv("POSE_MIN_DETECTION_CONFIDENCE", "0.3"))
POSE_NUM_POSES = int(os.getenv("POSE_NUM_POSES", "1"))

_LLM_MODEL_OVERRIDE = os.getenv("AIRBALL_LLM_MODEL")
if _LLM_MODEL_OVERRIDE:
    LLM_MODEL = _LLM_MODEL_OVERRIDE

# Path to the PoseLandmarker model file
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "pose_landmarker_lite.task")


@dataclass
class _FakeLandmark:
    """Mimic the old mp.solutions.pose landmark interface for ShotDetector compatibility."""
    x: float
    y: float
    z: float
    visibility: float


def _update_detector_with_frame(
    detector: ShotDetector,
    frame: np.ndarray,
    landmarks: list[_FakeLandmark],
    frame_w: int,
    frame_h: int,
    ts: float,
):
    """Compute optional ball context and update shot detector for one frame."""
    ball_state = None
    try:
        ball_state = detect_ball_state(frame, landmarks, frame_w, frame_h)
    except Exception:
        # Ball context is optional; continue shot detection when unavailable.
        ball_state = None

    return detector.update(landmarks, frame_w, frame_h, ts, ball_state=ball_state)


def _process_video(file_path: str) -> list[dict]:
    """Run MediaPipe PoseLandmarker + ShotDetector on every frame of a video file."""
    model_path = os.path.abspath(_MODEL_PATH)
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pose model not found at {model_path}. Download pose_landmarker_lite.task.",
        )

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not open video file",
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create PoseLandmarker using Tasks API
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=POSE_NUM_POSES,
        min_pose_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    detector = ShotDetector()
    shots: list[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps
        timestamp_ms = int(frame_idx * 1000 / fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            raw_landmarks = result.pose_landmarks[0]
            # Convert NormalizedLandmark to objects with .x, .y, .z, .visibility
            landmarks = [
                _FakeLandmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility if lm.visibility is not None else 0.0,
                )
                for lm in raw_landmarks
            ]

            shot = _update_detector_with_frame(detector, frame, landmarks, frame_w, frame_h, ts)
            if shot is not None:
                shots.append(shot)

        frame_idx += 1

    cap.release()
    landmarker.close()
    return shots


def _derive_scores(shot: dict) -> dict:
    """Derive the four summary stats from real shot data."""
    data_quality = shot.get("data_quality", {})
    confidence = data_quality.get("confidence", "low")
    occlusion_flags = data_quality.get("occlusion_flags", {})
    guardrail_mode = shot.get("feedback_guardrails", {}).get("mode", "normal")

    base = {"high": 80, "medium": 70, "low": 60}.get(confidence, 65)

    if shot.get("timing", {}).get("leg_drive_before_arm_extension"):
        base += 10
    hold = shot.get("metrics", {}).get("follow_through", {}).get("hold_duration_s") or 0
    if hold > 0.1:
        base += 5

    quality_penalty = 0
    if occlusion_flags.get("lower_body_occluded"):
        quality_penalty += 8
    if occlusion_flags.get("upper_body_occluded"):
        quality_penalty += 5
    if (data_quality.get("wrist_visibility_ratio") or 1.0) < 0.6:
        quality_penalty += 4
    if (data_quality.get("knee_visibility_ratio") or 1.0) < 0.6:
        quality_penalty += 4
    if guardrail_mode == "conservative":
        quality_penalty += 8

    shot_score = min(100, max(40, base - quality_penalty))
    if guardrail_mode == "conservative":
        shot_score = min(shot_score, 75)

    elbow = shot.get("metrics", {}).get("angles", {}).get("elbow", {})
    arc_angle = round(elbow.get("at_release_deg") or 0) or None

    peak_vel = shot.get("metrics", {}).get("velocities", {}).get("peak_wrist_vertical_px_s") or 0
    release_speed = round(abs(peak_vel) / 100, 1) if peak_vel else None

    ft_score = 70
    if hold > 0.05:
        ft_score += min(30, int(hold * 100))
    follow_through_score = min(100, ft_score)
    if guardrail_mode == "conservative":
        follow_through_score = min(follow_through_score, 80)

    return {
        "shot_score": shot_score,
        "arc_angle": arc_angle,
        "release_speed": release_speed,
        "follow_through_score": follow_through_score,
    }


def _build_shot_summary(shot: dict) -> str:
    """Convert raw shot data into a human-readable summary the LLM can reason about."""
    lines = []

    # Data quality context
    quality = shot.get("data_quality", {})
    confidence = quality.get("confidence", "unknown")
    occlusion = quality.get("occlusion_flags", {})
    lines.append(f"Tracking confidence: {confidence}")
    if occlusion.get("lower_body_occluded"):
        lines.append("Lower body was not fully visible — no leg drive or knee data available.")
    if occlusion.get("upper_body_occluded"):
        lines.append("Upper body was partially occluded.")

    # Elbow mechanics
    elbow = shot.get("metrics", {}).get("angles", {}).get("elbow", {})
    set_deg = elbow.get("at_set_deg")
    release_deg = elbow.get("at_release_deg")
    if set_deg is not None:
        if set_deg < 70:
            lines.append("Elbow is very bent at the set point — arm is compact")
        elif set_deg < 100:
            lines.append("Elbow is moderately bent at the set point")
        else:
            lines.append("Elbow is fairly open at the set point")
    if release_deg is not None:
        if release_deg < 140:
            lines.append("Elbow did not fully extend at release — arm is still too bent when letting go of the ball")
        elif release_deg < 160:
            lines.append("Elbow extension at release is decent but could be straighter")
        else:
            lines.append("Good full elbow extension at release")

    # Knee mechanics (if available)
    knee = shot.get("metrics", {}).get("angles", {}).get("knee", {})
    knee_load = knee.get("min_during_load_deg")
    if knee_load is not None:
        if knee_load < 120:
            lines.append("Good deep knee bend during the load phase — getting power from the legs")
        elif knee_load < 150:
            lines.append("Moderate knee bend during load — could bend a bit more for extra power")
        else:
            lines.append("Barely any knee bend during load — not using legs enough")

    # Release point
    release = shot.get("metrics", {}).get("release", {})
    above_head = release.get("wrist_above_head_norm")
    if above_head is not None:
        if above_head > 0:
            lines.append("Release point is BELOW the head — should aim to release above the head")
        else:
            lines.append("Release point is above the head — good release height")

    # Follow-through
    ft = shot.get("metrics", {}).get("follow_through", {})
    hold = ft.get("hold_duration_s")
    if hold is not None:
        if hold < 0.05:
            lines.append("No follow-through hold detected — hand dropped immediately after release")
        elif hold < 0.15:
            lines.append("Very short follow-through — hand drops too quickly after release")
        else:
            lines.append("Good follow-through hold — hand stays up after release")

    # Timing / leg drive
    timing = shot.get("timing", {})
    leg_drive = timing.get("leg_drive_before_arm_extension")
    if leg_drive is True:
        lines.append("Good sequencing: legs drive before arm extension")
    elif leg_drive is False:
        lines.append("Arm extended before legs — should initiate shot from the legs up")

    # Wrist snap
    wrist_snap = shot.get("phases", {}).get("wrist_snap", {})
    snap_vel = wrist_snap.get("angular_velocity_rad_s")
    if snap_vel is not None:
        if snap_vel > 100:
            lines.append("Strong wrist snap at release")
        elif snap_vel > 30:
            lines.append("Moderate wrist snap at release")
        else:
            lines.append("Weak wrist snap — needs more flick of the wrist at release")

    # Stability
    stability = shot.get("metrics", {}).get("stability", {})
    head_var = stability.get("head_vertical_variance_norm")
    if head_var is not None:
        if head_var < 0.002:
            lines.append("Head stays steady during the shot — good balance")
        else:
            lines.append("Noticeable head movement during the shot — work on balance")

    # Guardrails
    guardrails = shot.get("feedback_guardrails", {})
    mode = guardrails.get("mode", "normal")
    if mode == "conservative":
        lines.append("NOTE: Limited visibility means some metrics are missing. Feedback focuses on what was observable.")

    return "\n".join(lines)


def _generate_feedback(shot_data: dict) -> str:
    """Call local Ollama LLM to generate coaching feedback from shot data."""
    shot_summary = _build_shot_summary(shot_data)

    prompt = f"""You are a basketball shooting coach. A player filmed their shot and motion tracking analyzed it.

Here is what the analysis found:
{shot_summary}

Write short coaching feedback for this player. Use the second person ("you" / "your"). Follow this exact structure:

OVERALL: Write one sentence about the shot overall.

STRENGTHS:
- Write one thing the player did well.
- Write another thing the player did well.

WORK ON:
- Write one thing to improve and include a specific drill to fix it.
- Write another thing to improve and include a specific drill to fix it.

Important: Start directly with "OVERALL:" — no introduction or preamble. Do not include numbers or data. Do not ask questions. Keep each bullet to one or two sentences. Sound like a real basketball coach giving encouragement after practice."""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.get("message", {}).get("content", "").strip()
    except Exception as exc:
        return f"LLM feedback unavailable: {exc}"


@router.post("/video")
async def analyze_video(file: UploadFile = File(...)):
    """Accept a video upload, run pose analysis + LLM feedback, return results."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video",
        )

    # Save uploaded file to a temp path so OpenCV can read it
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        shots = _process_video(tmp.name)

        if not shots:
            return {
                "status": "no_shots_detected",
                "shots": [],
                "message": "No basketball shots were detected in the video. Try a clearer angle showing your full body.",
            }

        results = []
        for shot in shots:
            scores = _derive_scores(shot)
            feedback = _generate_feedback(shot)
            results.append({
                "shot_id": shot.get("id"),
                "scores": scores,
                "feedback": feedback,
                "shot_data": shot,
            })

        # Use the first (or best) shot as the primary result
        primary = results[0]

        return {
            "status": "analyzed",
            "shot_score": primary["scores"]["shot_score"],
            "arc_angle": primary["scores"]["arc_angle"],
            "release_speed": primary["scores"]["release_speed"],
            "follow_through_score": primary["scores"]["follow_through_score"],
            "llm_feedback": primary["feedback"],
            "shot_data": primary["shot_data"],
            "total_shots_detected": len(results),
            "all_shots": results,
        }
    finally:
        os.unlink(tmp.name)
