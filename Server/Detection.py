import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import queue
import threading
from collections import deque
from datetime import datetime
from flask import Flask, Response
from ultralytics import YOLO
import ollama

from shot_detector import ShotDetector
from camera import picam2
from ball_detector import detect_ball_state

app = Flask(__name__)

# Initialize person detection with YOLO
person_detector = YOLO('yolov8n.pt')  

# Buffers to store recent ball and wrist positions for verification
ball_buffer = deque(maxlen=180)
wrist_buffer = deque(maxlen=180)

# Initialize body detection
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=0)

# Shot detection and camera setup moved to modules
detector = ShotDetector()

# Shot clip + cooldown settings
SHOT_COOLDOWN_SECONDS = 5.0
CLIP_PRE_SECONDS = 1.5
CLIP_POST_SECONDS = 1.5
FRAME_BUFFER_SECONDS = 8.0
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Shots')
LLM_MODEL = 'gemma3:1b'

# Run LLM feedback in the background so streaming inference does not block.
feedback_queue = queue.Queue(maxsize=32)


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_clip(frames, shot_id):
    if not frames:
        return None

    _ensure_output_dir()

    first_ts = frames[0]['ts']
    last_ts = frames[-1]['ts']
    if len(frames) > 1 and last_ts > first_ts:
        fps = (len(frames) - 1) / (last_ts - first_ts)
        fps = max(10.0, min(60.0, fps))
    else:
        fps = 20.0

    first_frame = frames[0]['frame']
    height, width = first_frame.shape[:2]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    safe_id = (shot_id or 'unknown')[:8]
    out_path = os.path.join(OUTPUT_DIR, f"shot_{safe_id}_{stamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        return None

    for entry in frames:
        writer.write(entry['frame'])
    writer.release()
    return out_path


def _find_shot_json_path(shot_id):
    candidates = [
        os.path.join(OUTPUT_DIR, f'shot_{shot_id}.json'),
        os.path.join('Shots', f'shot_{shot_id}.json'),
        os.path.join(os.path.dirname(__file__), 'Shots', f'shot_{shot_id}.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _print_llm_feedback_for_shot(shot_id):
    shot_path = _find_shot_json_path(shot_id)
    if not shot_path:
        print(f"LLM skipped: shot JSON not found for shot {shot_id[:8]}")
        return

    try:
        with open(shot_path, 'r', encoding='utf-8') as f:
            shot_data = json.load(f)

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful sports performance coach assistant. Provide clear, concise, and actionable shooting feedback.'
                },
                {
                    'role': 'user',
                    'content': (
                        'I collected basketball shot form data. Provide feedback in four sections: '
                        'overall assessment, strengths, areas for improvement, and top 3 prioritized recommendations. '
                        'Avoid raw timestamps and precise floating-point values. Keep the tone encouraging and practical. '
                        f'Data: {json.dumps(shot_data, indent=2)}'
                    )
                }
            ]
        )
        feedback = response.get('message', {}).get('content', '').strip()
        if feedback:
            _ensure_output_dir()
            feedback_path = os.path.join(OUTPUT_DIR, f'shot_feedback_{shot_id[:8]}.txt')
            with open(feedback_path, 'w', encoding='utf-8') as f:
                f.write(feedback)
            print(f"\n=== LLM Feedback for shot {shot_id[:8]} ===\n{feedback}\n")
            print(f"Feedback saved: {feedback_path}")
        else:
            print(f"LLM returned empty feedback for shot {shot_id[:8]}")
    except Exception as exc:
        print(f"LLM feedback failed for shot {shot_id[:8]}: {exc}")


def _feedback_worker_loop():
    while True:
        shot_id = feedback_queue.get()
        try:
            _print_llm_feedback_for_shot(shot_id)
        finally:
            feedback_queue.task_done()


feedback_worker = threading.Thread(target=_feedback_worker_loop, daemon=True)
feedback_worker.start()


def generate_frames():
    frame_buffer = deque()
    pending_clip = None
    last_accepted_shot_ts = -1e9

    while True:
        raw_frame = picam2.capture_array()
        ts = time.time()
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2BGR)
        # Resize for speed
        scale = 0.25
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        small_h, small_w = small_frame.shape[:2]
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Person detection with YOLO
        results = person_detector(rgb_small_frame, conf=0.3, classes=[0])  
        # Ball detection 
        ball_results = person_detector(rgb_small_frame, conf=0.3, classes=[32])

        # extract first detected ball
        ball_center_full = None
        for bres in ball_results:
            for box in bres.boxes:
                x1b, y1b, x2b, y2b = box.xyxy[0].cpu().numpy()
                x1b, y1b, x2b, y2b = int(x1b), int(y1b), int(x2b), int(y2b)
                # center in small frame
                cx = int((x1b + x2b) / 2)
                cy = int((y1b + y2b) / 2)
                # convert to full frame coords
                full_cx = int(cx / scale)
                full_cy = int(cy / scale)
                ball_center_full = (full_cx, full_cy)
                ball_buffer.append({'ts': ts, 'pos': ball_center_full})
                # draw ball detection
                cv2.rectangle(frame, (int(x1b/scale), int(y1b/scale)), (int(x2b/scale), int(y2b/scale)), (0, 0, 255), 2)
                cv2.putText(frame, "Ball", (int(x1b/scale), int(y1b/scale) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                break
            if ball_center_full is not None:
                break

        people_landmarks = []
        processed = False
        for result in results:
            if processed:
                break
            for box in result.boxes:
                if processed:
                    break
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Crop the person
                person_crop = rgb_small_frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # Pose detection on crop
                pose_results = pose_detector.process(person_crop)
                if pose_results.pose_landmarks:
                    crop_h, crop_w = person_crop.shape[:2]
                    landmarks = pose_results.pose_landmarks.landmark

                    # Adjust landmarks to be relative to small_frame
                    for lm in landmarks:
                        lm.x = (x1 + lm.x * crop_w) / small_w
                        lm.y = (y1 + lm.y * crop_h) / small_h

                    # Compute coords for drawing (full frame)
                    coords = []
                    for lm in landmarks:
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        coords.append((x, y))

                    people_landmarks.append(landmarks)  

                    # store wrist position in full-frame coords for verification
                    # choose wrist by visibility if available
                    try:
                        rw_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value
                        lw_idx = mp_pose.PoseLandmark.LEFT_WRIST.value
                        # pick the wrist with higher visibility if available
                        vis_r = landmarks[rw_idx].visibility if hasattr(landmarks[rw_idx], 'visibility') else 0.0
                        vis_l = landmarks[lw_idx].visibility if hasattr(landmarks[lw_idx], 'visibility') else 0.0
                        wrist_idx = rw_idx if vis_r >= vis_l else lw_idx
                    except Exception:
                        wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value
                    # compute wrist full-frame coord
                    wlm = landmarks[wrist_idx]
                    wrist_full = (int(wlm.x * frame.shape[1]), int(wlm.y * frame.shape[0]))
                    wrist_buffer.append({'ts': ts, 'pos': wrist_full})

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1/scale), int(y1/scale)), (int(x2/scale), int(y2/scale)), (0, 255, 0), 2)
                    cv2.putText(frame, "Person Detected", (int(x1/scale), int(y1/scale) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Draw skeleton
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(coords) and end_idx < len(coords):
                            cv2.line(frame, coords[start_idx], coords[end_idx], (0, 200, 255), 2)

                    # Draw keypoints
                    for (x, y) in coords:
                        cv2.circle(frame, (x, y), 3, (0, 165, 255), -1)

                    processed = True
                    break

        # For shot detection, use the first person's landmarks if any
        if people_landmarks:
            # use current ts (captured near frame read)
            shot = detector.update(people_landmarks[0], frame.shape[1], frame.shape[0], ts)
            if shot is not None:
                # verify shot by checking that the ball leaves the hand after the release timestamp
                try:
                    release_ts = float(shot['phases']['release']['ts'])
                except Exception:
                    release_ts = shot.get('detection_window', {}).get('end', ts)

                # find wrist position nearest to release
                wrist_at_release = None
                if wrist_buffer:
                    wrist_at_release = min(wrist_buffer, key=lambda e: abs(e['ts'] - release_ts))

                # find ball positions after release (+ small epsilon)
                ball_after = [e for e in list(ball_buffer) if e['ts'] > release_ts + 0.01]

                accepted = False
                if wrist_at_release and ball_after:
                    threshold_px = max(40, int(frame.shape[1] * 0.03))
                    wrist_pos = wrist_at_release['pos']
                    max_sep = 0.0
                    for be in ball_after:
                        sep = ((be['pos'][0] - wrist_pos[0])**2 + (be['pos'][1] - wrist_pos[1])**2)**0.5
                        if sep > max_sep:
                            max_sep = sep
                        if be['ts'] - release_ts > 0.5:
                            break
                    if max_sep > threshold_px:
                        accepted = True

                if accepted:
                    txt = f"Shot detected: {shot['id'][:8]} dur={shot['detection_window']['duration']:.2f}s"
                    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if ts - last_accepted_shot_ts >= SHOT_COOLDOWN_SECONDS:
                        last_accepted_shot_ts = ts
                        pending_clip = {
                            'shot_id': shot.get('id', 'unknown'),
                            'start_ts': ts - CLIP_PRE_SECONDS,
                            'end_ts': ts + CLIP_POST_SECONDS,
                            'saved': False,
                        }
                    else:
                        cooldown_left = SHOT_COOLDOWN_SECONDS - (ts - last_accepted_shot_ts)
                        cooldown_txt = f"Cooldown active: {cooldown_left:.1f}s"
                        cv2.putText(frame, cooldown_txt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
                else:
                    txt = f"Shot ignored (no ball separation)"
                    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

        frame_buffer.append({'ts': ts, 'frame': frame.copy()})
        while frame_buffer and frame_buffer[0]['ts'] < ts - FRAME_BUFFER_SECONDS:
            frame_buffer.popleft()

        if pending_clip and not pending_clip['saved'] and ts >= pending_clip['end_ts']:
            clip_frames = [
                f for f in frame_buffer
                if pending_clip['start_ts'] <= f['ts'] <= pending_clip['end_ts']
            ]
            saved_path = _save_clip(clip_frames, pending_clip['shot_id'])
            if saved_path:
                print(f"Saved shot clip: {saved_path}")
                try:
                    feedback_queue.put_nowait(pending_clip['shot_id'])
                except queue.Full:
                    print("Feedback queue is full, skipping LLM request for this shot")
            else:
                print("Failed to save shot clip")
            pending_clip['saved'] = True
            pending_clip = None

        # Web Stream
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>AirBall Live Feed</h1><img src='/video_feed' width='640'>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
