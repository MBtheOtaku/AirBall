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

# Initialize body detection
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=0)

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
