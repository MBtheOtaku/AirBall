from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("mediapipe")

from ball_detector import detect_ball_state


class _Landmark:
    def __init__(self, x: float, y: float, visibility: float):
        self.x = x
        self.y = y
        self.visibility = visibility


def _make_landmarks(right_vis: float, left_vis: float):
    points = [_Landmark(0.5, 0.5, 0.0) for _ in range(33)]
    # MediaPipe pose indices: right wrist 16, left wrist 15.
    points[16] = _Landmark(0.6, 0.6, right_vis)
    points[15] = _Landmark(0.4, 0.6, left_vis)
    points[11] = _Landmark(0.45, 0.3, 1.0)  # left shoulder
    points[12] = _Landmark(0.55, 0.3, 1.0)  # right shoulder
    return points


def test_detect_ball_state_returns_not_detected_for_low_wrist_visibility():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    landmarks = _make_landmarks(right_vis=0.1, left_vis=0.2)

    state = detect_ball_state(frame, landmarks, frame_w=160, frame_h=120)

    assert state["detected"] is False
    assert state["in_hand_score"] is None


def test_detect_ball_state_selects_nearest_circle_to_wrist():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    landmarks = _make_landmarks(right_vis=0.9, left_vis=0.4)

    # Two circles: one close to wrist, one far away.
    circles = np.array([[[95, 72, 12], [20, 20, 10]]], dtype=np.float32)

    with patch("ball_detector.cv2.HoughCircles", return_value=circles):
        state = detect_ball_state(frame, landmarks, frame_w=160, frame_h=120)

    assert state["detected"] is True
    assert state["ball_center_px"] == {"x": 95, "y": 72}
    assert 0.0 <= state["in_hand_score"] <= 1.0


def test_detect_ball_state_returns_not_detected_when_no_circle_found():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    landmarks = _make_landmarks(right_vis=0.9, left_vis=0.8)

    with patch("ball_detector.cv2.HoughCircles", return_value=None):
        state = detect_ball_state(frame, landmarks, frame_w=160, frame_h=120)

    assert state["detected"] is False
    assert state["palm_gap_px"] is None
