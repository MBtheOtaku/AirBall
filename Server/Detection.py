import cv2
import mediapipe as mp
import numpy as np
import os
import time
from flask import Flask, Response

from shot_detector import ShotDetector
from camera import picam2

app = Flask(__name__)

# Initialize body detection
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Shot detection and camera setup moved to modules
detector = ShotDetector()


def _dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _compute_scale_px(landmarks, frame_w, frame_h):
    try:
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_w
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_w
        ls_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_h
        rs_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_h
        shoulder_width = _dist((ls, ls_y), (rs, rs_y))
        return max(shoulder_width, 1.0)
    except Exception:
        return max(frame_h * 0.25, 1.0)


def _choose_wrist(landmarks, frame_w, frame_h):
    try:
        rv = float(getattr(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], 'visibility', 0.0))
        lv = float(getattr(landmarks[mp_pose.PoseLandmark.LEFT_WRIST], 'visibility', 0.0))
        wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST if rv >= lv else mp_pose.PoseLandmark.LEFT_WRIST
        wrist = landmarks[wrist_idx]
        return (float(wrist.x) * frame_w, float(wrist.y) * frame_h)
    except Exception:
        return None


def detect_ball_state(frame_bgr, landmarks, frame_w, frame_h):
    wrist = _choose_wrist(landmarks, frame_w, frame_h)
    if wrist is None:
        return {'detected': False, 'in_hand_score': None, 'palm_gap_px': None}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=frame_h * 0.12,
        param1=120,
        param2=25,
        minRadius=8,
        maxRadius=int(frame_h * 0.15)
    )

    if circles is None or len(circles) == 0:
        return {'detected': False, 'in_hand_score': None, 'palm_gap_px': None}

    circles = np.round(circles[0, :]).astype(int)
    best = None
    best_dist = None
    for (x, y, r) in circles:
        d = _dist((x, y), wrist)
        if best is None or d < best_dist:
            best = (x, y, r)
            best_dist = d

    if best is None:
        return {'detected': False, 'in_hand_score': None, 'palm_gap_px': None}

    x, y, r = best
    scale_px = _compute_scale_px(landmarks, frame_w, frame_h)
    palm_gap_px = max(0.0, best_dist - float(r)) if best_dist is not None else None
    in_hand_score = None
    if best_dist is not None:
        denom = max(0.45 * scale_px, 1.0)
        in_hand_score = max(0.0, min(1.0, 1.0 - (best_dist / denom)))

    return {
        'detected': True,
        'in_hand_score': float(in_hand_score) if in_hand_score is not None else None,
        'palm_gap_px': float(palm_gap_px) if palm_gap_px is not None else None,
        'ball_center_px': {'x': int(x), 'y': int(y)},
        'ball_radius_px': int(r)
    }


def generate_frames():
    while True:
        raw_frame = picam2.capture_array()
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGBA2BGR)
        # Resize for speed
        scale = 0.25
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Pose detection
        results = pose_detector.process(rgb_small_frame)
        if results.pose_landmarks:
            small_h, small_w = rgb_small_frame.shape[:2]
            landmarks = results.pose_landmarks.landmark
            coords = []
            for lm in landmarks:
                x = int(lm.x * small_w * (1/scale))
                y = int(lm.y * small_h * (1/scale))
                coords.append((x, y))

            # Update shot detector with full-resolution frame size and timestamp
            ts = time.time()
            ball_state = detect_ball_state(frame, landmarks, frame.shape[1], frame.shape[0])
            shot = detector.update(landmarks, frame.shape[1], frame.shape[0], ts, ball_state=ball_state)
            if shot is not None:
                # overlay a small notification
                txt = f"Shot detected: {shot['id'][:8]} dur={shot['duration']:.2f}s"
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            min_x, max_x = max(min(xs) - 20, 0), min(max(xs) + 20, frame.shape[1])
            min_y, max_y = max(min(ys) - 20, 0), min(max(ys) + 20, frame.shape[0])

            # Bounding box + label
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            cv2.putText(frame, "Person Detected", (min_x, min_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw skeleton using connections
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(coords) and end_idx < len(coords):
                    cv2.line(frame, coords[start_idx], coords[end_idx], (0, 200, 255), 2)

            # Draw keypoints
            for (x, y) in coords:
                cv2.circle(frame, (x, y), 3, (0, 165, 255), -1)

        # Web Stream
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Security Feed (Body Tracking Only)</h1><img src='/video_feed' width='640'>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
