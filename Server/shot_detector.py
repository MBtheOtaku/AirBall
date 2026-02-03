import os
import time
import json
import uuid
import math
from collections import deque

import numpy as np
import mediapipe as mp


class ShotDetector:
    def __init__(self, buffer_size=90):
        self.buf = deque(maxlen=buffer_size)
        self.in_shot = False
        self.current_shot_frames = []
        self.last_shot_id = 0

        # landmarks indices
        self.RW = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value
        self.LW = mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
        self.RE = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value
        self.LE = mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value
        self.RS = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
        self.LS = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        self.MID_HIP = mp.solutions.pose.PoseLandmark.LEFT_HIP.value

    def _angle(self, a, b, c):
        # angle at b between points a-b-c in degrees
        ax, ay = a
        bx, by = b
        cx, cy = c
        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])
        if mag1 * mag2 == 0:
            return 0.0
        cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cosang))

    def _choose_side(self, lm_list):
        # choose dominant hand by visibility of wrists
        try:
            rv = lm_list[self.RW].visibility
            lv = lm_list[self.LW].visibility
            return 'right' if rv >= lv else 'left'
        except Exception:
            return 'right'

    def update(self, lm_list, frame_w, frame_h, ts):
        # store normalized landmark list + pixel positions + timestamp
        entry = {'lm': lm_list, 'ts': ts}
        # compute pixel positions for convenience
        pix = []
        for lm in lm_list:
            pix.append((float(lm.x) * frame_w, float(lm.y) * frame_h, float(lm.z), float(getattr(lm, 'visibility', 0.0))))
        entry['pix'] = pix
        self.buf.append(entry)

        # compute wrist vertical velocity (pixels/sec) using last frames
        if len(self.buf) < 3:
            return None

        # choose side
        side = self._choose_side(lm_list)
        wrist_idx = self.RW if side == 'right' else self.LW
        elbow_idx = self.RE if side == 'right' else self.LE
        shoulder_idx = self.RS if side == 'right' else self.LS

        # get last two frames
        a = self.buf[-2]
        b = self.buf[-1]
        dt = b['ts'] - a['ts'] if b['ts'] != a['ts'] else 1/30
        ay = a['pix'][wrist_idx][1]
        by = b['pix'][wrist_idx][1]
        # upward velocity positive (since we compute ay - by)
        vy = (ay - by) / dt

        # elbow angle change
        def elbow_angle_at(entry):
            pix = entry['pix']
            shoulder = (pix[shoulder_idx][0], pix[shoulder_idx][1])
            elbow = (pix[elbow_idx][0], pix[elbow_idx][1])
            wrist = (pix[wrist_idx][0], pix[wrist_idx][1])
            return self._angle(shoulder, elbow, wrist)

        elbow_prev = elbow_angle_at(a)
        elbow_now = elbow_angle_at(b)

        # Detection thresholds (tweakable)
        VY_START = 300.0  # pixels/sec upward
        ELBOW_EXTENSION_MIN = 5.0  # degrees increase

        if not self.in_shot:
            # start when wrist moves up quickly and elbow starts extending
            if vy > VY_START and (elbow_now - elbow_prev) > ELBOW_EXTENSION_MIN:
                self.in_shot = True
                self.current_shot_frames = list(self.buf)  # include buffer pre-roll
                return None
            else:
                return None

        # in shot: append current frame
        self.current_shot_frames.append(b)

        # determine apex (min wrist y) and detect end when hand descending and velocity small
        ys = [f['pix'][wrist_idx][1] for f in self.current_shot_frames]
        min_y = min(ys)
        min_idx = ys.index(min_y)

        # compute recent vy
        recent_vy = vy
        # End conditions
        VY_END = 80.0
        MAX_DURATION = 3.0
        start_ts = self.current_shot_frames[0]['ts']
        if (recent_vy < VY_END and b['ts'] - start_ts > 0.12) or (b['ts'] - start_ts) > MAX_DURATION:
            # finalize shot
            shot = self._finalize_shot(self.current_shot_frames, wrist_idx, elbow_idx, shoulder_idx)
            self.in_shot = False
            self.current_shot_frames = []
            return shot

        return None

    def _finalize_shot(self, frames, wrist_idx, elbow_idx, shoulder_idx):
        # frames: list of entries with 'pix' and 'ts'
        start_ts = frames[0]['ts']
        end_ts = frames[-1]['ts']
        duration = end_ts - start_ts

        # compute apex (min wrist y)
        ys = [f['pix'][wrist_idx][1] for f in frames]
        apex_i = int(np.argmin(ys))
        apex_frame = frames[apex_i]

        # metrics: peak elbow extension (max angle), peak wrist velocity
        elbow_angles = []
        wrist_vys = []
        hip_ys = []
        for i in range(1, len(frames)):
            f0 = frames[i-1]
            f1 = frames[i]
            dt = f1['ts'] - f0['ts'] if f1['ts'] - f0['ts'] != 0 else 1/30
            # elbow
            shoulder = (f1['pix'][shoulder_idx][0], f1['pix'][shoulder_idx][1])
            elbow = (f1['pix'][elbow_idx][0], f1['pix'][elbow_idx][1])
            wrist = (f1['pix'][wrist_idx][0], f1['pix'][wrist_idx][1])
            elbow_angles.append(self._angle(shoulder, elbow, wrist))
            # wrist vy
            vy = (f0['pix'][wrist_idx][1] - f1['pix'][wrist_idx][1]) / dt
            wrist_vys.append(vy)
            # hip
            hip_ys.append(f1['pix'][self.MID_HIP][1] if self.MID_HIP < len(f1['pix']) else 0)

        peak_elbow = float(np.max(elbow_angles)) if elbow_angles else 0.0
        peak_wrist_v = float(np.max(wrist_vys)) if wrist_vys else 0.0
        hip_drop = 0.0
        if hip_ys:
            hip_drop = float(hip_ys[0] - min(hip_ys))  # positive if hips moved up (y decreased -> negative), keep magnitude

        release = {'ts': apex_frame['ts'], 'wrist_y': float(apex_frame['pix'][wrist_idx][1])}

        # angles at release
        sh = (apex_frame['pix'][shoulder_idx][0], apex_frame['pix'][shoulder_idx][1])
        el = (apex_frame['pix'][elbow_idx][0], apex_frame['pix'][elbow_idx][1])
        wr = (apex_frame['pix'][wrist_idx][0], apex_frame['pix'][wrist_idx][1])
        release_elbow_angle = float(self._angle(sh, el, wr))
        release_shoulder_angle = float(self._angle((el[0], el[1]), sh, (apex_frame['pix'][self.MID_HIP][0], apex_frame['pix'][self.MID_HIP][1]) if self.MID_HIP < len(apex_frame['pix']) else (sh[0], sh[1]+10)))

        shot_id = str(uuid.uuid4())
        shot = {
            'id': shot_id,
            'start_time': start_ts,
            'end_time': end_ts,
            'duration': duration,
            'release': release,
            'metrics': {
                'peak_elbow_angle_deg': peak_elbow,
                'peak_wrist_velocity_px_s': peak_wrist_v,
                'hip_vertical_displacement_px': hip_drop,
                'release_elbow_angle_deg': release_elbow_angle,
                'release_shoulder_angle_deg': release_shoulder_angle
            },
            'frame_count': len(frames)
        }
        # Save JSON to Shots folder
        try:
            os.makedirs('Shots', exist_ok=True)
            fname = os.path.join('Shots', f'shot_{shot_id}.json')
            with open(fname, 'w') as f:
                json.dump(shot, f, indent=2)
        except Exception:
            pass

        return shot
