import cv2
import mediapipe as mp
import numpy as np
import os


mp_pose = mp.solutions.pose


BALL_HOUGH_DP = float(os.getenv("AIRBALL_BALL_HOUGH_DP", "1.2"))
BALL_HOUGH_MIN_DIST_RATIO = float(os.getenv("AIRBALL_BALL_HOUGH_MIN_DIST_RATIO", "0.12"))
BALL_HOUGH_PARAM1 = int(os.getenv("AIRBALL_BALL_HOUGH_PARAM1", "120"))
BALL_HOUGH_PARAM2 = int(os.getenv("AIRBALL_BALL_HOUGH_PARAM2", "25"))
BALL_MIN_RADIUS_PX = int(os.getenv("AIRBALL_BALL_MIN_RADIUS_PX", "8"))
BALL_MAX_RADIUS_RATIO = float(os.getenv("AIRBALL_BALL_MAX_RADIUS_RATIO", "0.15"))
WRIST_MIN_VISIBILITY = float(os.getenv("AIRBALL_WRIST_MIN_VISIBILITY", "0.45"))


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
        if max(rv, lv) < WRIST_MIN_VISIBILITY:
            return None
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
        dp=BALL_HOUGH_DP,
        minDist=frame_h * BALL_HOUGH_MIN_DIST_RATIO,
        param1=BALL_HOUGH_PARAM1,
        param2=BALL_HOUGH_PARAM2,
        minRadius=BALL_MIN_RADIUS_PX,
        maxRadius=max(int(frame_h * BALL_MAX_RADIUS_RATIO), BALL_MIN_RADIUS_PX + 1),
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
