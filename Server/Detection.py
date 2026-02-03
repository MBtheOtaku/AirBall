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
            shot = detector.update(landmarks, frame.shape[1], frame.shape[0], ts)
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
