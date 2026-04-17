import io
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure Server root is on the path for shot_detector import
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app

client = TestClient(app)


def _make_test_video(frames=90, fps=30, width=640, height=480) -> bytes:
    """Create a minimal mp4 video in memory and return the bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp.name, fourcc, float(fps), (width, height))
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 320 + int(100 * np.sin(i / 10.0))
            y = 240 + int(50 * np.cos(i / 10.0))
            cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
            writer.write(frame)
        writer.release()
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp.name)


# -------------------------------------------------------------------
# Health endpoint
# -------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# -------------------------------------------------------------------
# Rejection tests
# -------------------------------------------------------------------

def test_reject_non_video_file():
    """Non-video content type should be rejected with 400."""
    resp = client.post(
        "/analyze/video",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400
    assert "video" in resp.json()["detail"].lower()


def test_reject_missing_file():
    """Request without a file should fail with 422."""
    resp = client.post("/analyze/video")
    assert resp.status_code == 422


# -------------------------------------------------------------------
# No-shot detection (dummy video with no person)
# -------------------------------------------------------------------

def test_no_shots_detected():
    """A video with no person should return no_shots_detected."""
    video_bytes = _make_test_video(frames=30)
    resp = client.post(
        "/analyze/video",
        files={"file": ("test.mp4", video_bytes, "video/mp4")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "no_shots_detected"
    assert data["shots"] == []


# -------------------------------------------------------------------
# Mocked shot detection to test the full pipeline
# -------------------------------------------------------------------

def _make_fake_shot():
    """Return a minimal shot dict matching what ShotDetector._finalize_shot returns."""
    return {
        "id": "test-shot-001",
        "detection_window": {"start": 0.0, "end": 1.0, "duration": 1.0},
        "fps": 30.0,
        "phases": {
            "set": {"ts": 0.0},
            "load": {"ts": 0.2, "knee_angle_deg": 140.0},
            "hip_extension_start": {"ts": 0.3},
            "elbow_extension_start": {"ts": 0.4},
            "wrist_snap": {"ts": 0.6, "angular_velocity_rad_s": 5.0},
            "release": {"ts": 0.7},
            "follow_through": {"start_ts": 0.8, "end_ts": 1.0, "hold_duration": 0.2},
        },
        "timing": {
            "leg_drive_before_arm_extension": True,
            "leg_to_elbow_delay_s": 0.1,
        },
        "metrics": {
            "angles": {
                "elbow": {"at_set_deg": 90.0, "at_load_deg": 85.0, "at_release_deg": 160.0},
                "knee": {"min_during_load_deg": 140.0, "at_release_deg": 170.0},
                "hip": {"at_load_deg": 150.0, "peak_extension_deg": 175.0},
            },
            "velocities": {
                "peak_wrist_vertical_px_s": 800.0,
                "peak_forearm_angular_velocity_rad_s": 5.0,
            },
            "release": {
                "ts": 0.7,
                "wrist_y_px": 100.0,
                "wrist_above_head_norm": -0.1,
                "wrist_above_shoulder_norm": -0.3,
            },
            "follow_through": {"hold_duration_s": 0.2},
            "stability": {"head_vertical_variance_norm": 0.001},
        },
        "data_quality": {
            "confidence": "high",
            "upper_body_visibility_ratio": 0.95,
            "lower_body_visibility_ratio": 0.9,
            "wrist_visibility_ratio": 0.92,
            "knee_visibility_ratio": 0.88,
            "occlusion_flags": {
                "upper_body_occluded": False,
                "lower_body_occluded": False,
                "wrist_often_missing": False,
                "knee_often_missing": False,
            },
        },
        "ball_context": {
            "supported": False,
            "ball_presence_ratio": 0.0,
            "ball_in_hand_score": None,
            "ball_in_hand_confirmed": False,
            "palm_gap_px_mean": None,
            "palm_gap_px_std": None,
            "grip_feedback_eligible": False,
        },
        "feedback_guardrails": {
            "mode": "normal",
            "message": "Tracking confidence is sufficient for detailed shot-form feedback.",
            "allow_grip_feedback": False,
        },
        "frame_count": 30,
    }


def test_analyzed_with_mocked_shot():
    """When a shot is detected, the response should include scores and LLM feedback."""
    video_bytes = _make_test_video(frames=30)
    fake_shot = _make_fake_shot()

    with patch("app.routes.analysis._process_video", return_value=[fake_shot]), \
         patch("app.routes.analysis._generate_feedback", return_value="Great form! Keep it up."):
        resp = client.post(
            "/analyze/video",
            files={"file": ("shot.mp4", video_bytes, "video/mp4")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "analyzed"
    assert data["shot_score"] is not None
    assert data["follow_through_score"] is not None
    assert data["llm_feedback"] == "Great form! Keep it up."
    assert data["total_shots_detected"] == 1


# -------------------------------------------------------------------
# Score derivation unit tests
# -------------------------------------------------------------------

def test_derive_scores_high_confidence():
    from app.routes.analysis import _derive_scores
    shot = _make_fake_shot()
    scores = _derive_scores(shot)

    assert 80 <= scores["shot_score"] <= 100
    assert scores["arc_angle"] == 160  # elbow at release
    assert scores["release_speed"] is not None
    assert scores["follow_through_score"] >= 70


def test_derive_scores_low_confidence():
    from app.routes.analysis import _derive_scores
    shot = _make_fake_shot()
    shot["data_quality"]["confidence"] = "low"
    shot["timing"]["leg_drive_before_arm_extension"] = False
    shot["metrics"]["follow_through"]["hold_duration_s"] = 0.0
    scores = _derive_scores(shot)

    assert scores["shot_score"] == 60
    assert scores["follow_through_score"] == 70


def test_derive_scores_conservative_mode_caps_scores():
    from app.routes.analysis import _derive_scores
    shot = _make_fake_shot()
    shot["feedback_guardrails"]["mode"] = "conservative"
    shot["metrics"]["follow_through"]["hold_duration_s"] = 0.5
    scores = _derive_scores(shot)

    assert scores["shot_score"] <= 75
    assert scores["follow_through_score"] <= 80


def test_derive_scores_penalize_occlusion_and_low_visibility():
    from app.routes.analysis import _derive_scores
    shot = _make_fake_shot()
    shot["data_quality"]["occlusion_flags"]["lower_body_occluded"] = True
    shot["data_quality"]["occlusion_flags"]["upper_body_occluded"] = True
    shot["data_quality"]["wrist_visibility_ratio"] = 0.4
    shot["data_quality"]["knee_visibility_ratio"] = 0.5

    scores = _derive_scores(shot)
    assert scores["shot_score"] < 95


def test_update_detector_with_frame_passes_ball_state():
    from app.routes.analysis import _update_detector_with_frame

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    landmarks = [MagicMock(x=0.5, y=0.5, z=0.0, visibility=0.9)]
    fake_detector = MagicMock()
    fake_detector.update.return_value = {"id": "shot-1"}
    fake_ball_state = {"detected": True, "in_hand_score": 0.8, "palm_gap_px": 4.2}

    with patch("app.routes.analysis.detect_ball_state", return_value=fake_ball_state):
        shot = _update_detector_with_frame(
            detector=fake_detector,
            frame=frame,
            landmarks=landmarks,
            frame_w=160,
            frame_h=120,
            ts=0.1,
        )

    assert shot == {"id": "shot-1"}
    fake_detector.update.assert_called_once_with(
        landmarks,
        160,
        120,
        0.1,
        ball_state=fake_ball_state,
    )


def test_update_detector_with_frame_ball_detector_failure():
    from app.routes.analysis import _update_detector_with_frame

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    landmarks = [MagicMock(x=0.5, y=0.5, z=0.0, visibility=0.9)]
    fake_detector = MagicMock()
    fake_detector.update.return_value = None

    with patch("app.routes.analysis.detect_ball_state", side_effect=RuntimeError("boom")):
        shot = _update_detector_with_frame(
            detector=fake_detector,
            frame=frame,
            landmarks=landmarks,
            frame_w=160,
            frame_h=120,
            ts=0.2,
        )

    assert shot is None
    fake_detector.update.assert_called_once_with(
        landmarks,
        160,
        120,
        0.2,
        ball_state=None,
    )
