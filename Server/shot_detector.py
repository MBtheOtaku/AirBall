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

    def _dist(self, p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def _compute_scale(self, pix):
        # compute normalization scale from shoulders and torso
        # pix is list of (x,y,z,vis)
        try:
            ls = pix[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = pix[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            lh = pix[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            rh = pix[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            shoulder_width = self._dist((ls[0], ls[1]), (rs[0], rs[1]))
            torso_len_l = self._dist((ls[0], ls[1]), (lh[0], lh[1]))
            torso_len_r = self._dist((rs[0], rs[1]), (rh[0], rh[1]))
            torso_len = (torso_len_l + torso_len_r) / 2.0
            scale = max(shoulder_width, torso_len, 1.0)
            return float(scale)
        except Exception:
            return 1.0

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
        # compute normalization scale per-frame for robustness
        entry['scale'] = self._compute_scale(pix)
        self.buf.append(entry)

        # compute wrist vertical velocity (pixels/sec) using last frames
        if len(self.buf) < 3:
            return None

        # choose side
        side = self._choose_side(lm_list)
        wrist_idx = self.RW if side == 'right' else self.LW
        elbow_idx = self.RE if side == 'right' else self.LE
        shoulder_idx = self.RS if side == 'right' else self.LS
        knee_idx = mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value if side == 'right' else mp.solutions.pose.PoseLandmark.LEFT_KNEE.value
        ankle_idx = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value if side == 'right' else mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value
        hip_idx = mp.solutions.pose.PoseLandmark.RIGHT_HIP.value if side == 'right' else mp.solutions.pose.PoseLandmark.LEFT_HIP.value

        # get last two frames
        a = self.buf[-2]
        b = self.buf[-1]
        dt = b['ts'] - a['ts'] if b['ts'] != a['ts'] else 1/30
        ay = a['pix'][wrist_idx][1]
        by = b['pix'][wrist_idx][1]
        # upward velocity positive (since we compute ay - by)
        vy = (ay - by) / dt

        # normalize by body scale (px) to get "body-lengths per second"
        scale_px = float(b.get('scale', 1.0)) if b else 1.0
        scale_px = max(scale_px, 1.0)
        vy_norm = vy / scale_px

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
        VY_START_NORM = 1.2       # body-lengths per second upward
        VY_END_NORM   = 0.25      # body-lengths per second upward
        ELBOW_EXTENSION_MIN = 5.0  # degrees increase
        KNEE_BEND_START_DEG = 4.0  # degrees from baseline

        # compute knee angles to help detect load
        def knee_angle_at(entry):
            pix = entry['pix']
            hip = (pix[hip_idx][0], pix[hip_idx][1])
            knee = (pix[knee_idx][0], pix[knee_idx][1])
            ankle = (pix[ankle_idx][0], pix[ankle_idx][1])
            return self._angle(hip, knee, ankle)

        knee_prev = knee_angle_at(a)
        knee_now = knee_angle_at(b)

        if not self.in_shot:
            # detect movement start if wrist moves up quickly and elbow starts extending
            # or if knee begins bending compared to buffer baseline
            baseline_knee = np.mean([knee_angle_at(e) for e in list(self.buf)[:-2]]) if len(self.buf) > 5 else knee_prev
            knee_drop = baseline_knee - knee_now

            if (vy_norm > VY_START_NORM and (elbow_now - elbow_prev) > ELBOW_EXTENSION_MIN)or (knee_drop > KNEE_BEND_START_DEG):
                self.in_shot = True
                self.current_shot_frames = list(self.buf)  # include buffer pre-roll
                return None
            else:
                return None

        # in shot: append current frame
        self.current_shot_frames.append(b)

        # determine apex (min wrist y)
        ys = [f['pix'][wrist_idx][1] for f in self.current_shot_frames]
        min_y = min(ys)
        min_idx = ys.index(min_y)

        # compute recent vy and end conditions
        recent_vy_norm = vy_norm
        MAX_DURATION = 3.0
        MIN_SHOT_DURATION = 0.08
        start_ts = self.current_shot_frames[0]['ts']
        if ((recent_vy_norm < VY_END_NORM and b['ts'] - start_ts > MIN_SHOT_DURATION) or (b['ts'] - start_ts) > MAX_DURATION):
            shot = self._finalize_shot(self.current_shot_frames, wrist_idx, elbow_idx, shoulder_idx, knee_idx, hip_idx, ankle_idx)
            self.in_shot = False
            self.current_shot_frames = []
            return shot

        return None

    def _finalize_shot(self, frames, wrist_idx, elbow_idx, shoulder_idx, knee_idx, hip_idx, ankle_idx):
        # Build a phase-aware, normalized JSON describing the shot
        start_ts = frames[0]['ts']
        end_ts = frames[-1]['ts']
        detection_window = {'start': start_ts, 'end': end_ts, 'duration': end_ts - start_ts}

        # compute dt array and fps
        dts = [frames[i]['ts'] - frames[i-1]['ts'] for i in range(1, len(frames))]
        median_dt = float(np.median(dts)) if dts else 1.0/30.0
        fps = 1.0 / median_dt if median_dt > 0 else 30.0

        # per-frame angles
        elbow_angles = []
        knee_angles = []
        hip_angles = []
        wrist_positions = []
        shoulder_positions = []
        head_positions = []
        scales = []
        ts_list = [f['ts'] for f in frames]

        nose_idx = mp.solutions.pose.PoseLandmark.NOSE.value

        for f in frames:
            pix = f['pix']
            scales.append(f.get('scale', 1.0))
            sh = (pix[shoulder_idx][0], pix[shoulder_idx][1])
            el = (pix[elbow_idx][0], pix[elbow_idx][1])
            wr = (pix[wrist_idx][0], pix[wrist_idx][1])
            hip = (pix[hip_idx][0], pix[hip_idx][1])
            knee = (pix[knee_idx][0], pix[knee_idx][1])
            ankle = (pix[ankle_idx][0], pix[ankle_idx][1])
            nose = (pix[nose_idx][0], pix[nose_idx][1]) if nose_idx < len(pix) else (sh[0], sh[1]-100)

            elbow_angles.append(self._angle(sh, el, wr))
            knee_angles.append(self._angle(hip, knee, ankle))
            hip_angles.append(self._angle(sh, hip, knee))
            wrist_positions.append({'x': wr[0], 'y': wr[1]})
            shoulder_positions.append({'x': sh[0], 'y': sh[1]})
            head_positions.append({'x': nose[0], 'y': nose[1]})

        # choose a representative scale (median)
        scale = float(np.median(scales)) if scales else 1.0

        # release (apex) based on wrist y (minimum y)
        wrist_ys = [p['y'] for p in wrist_positions]
        release_i = int(np.argmin(wrist_ys))
        release_ts = ts_list[release_i]

        # Load: maximum knee bend (minimum knee angle)
        load_i = int(np.argmin(knee_angles))
        load_ts = ts_list[load_i]

        # Hip extension start: first index after load where hip angle increases by threshold
        HIP_EXT_THRESH = 4.0
        hip_at_load = hip_angles[load_i]
        hip_ext_i = None
        for i in range(load_i, len(hip_angles)):
            if hip_angles[i] - hip_at_load > HIP_EXT_THRESH:
                hip_ext_i = i
                break
        hip_ext_ts = ts_list[hip_ext_i] if hip_ext_i is not None else None

        # Elbow extension start: first index where elbow angle increases notably before release
        ELBOW_EXT_THRESH = 5.0
        elbow_min_before_release = float(np.min(elbow_angles[:release_i+1]))
        elbow_ext_i = None
        for i in range(load_i, release_i+1):
            if elbow_angles[i] - elbow_min_before_release > ELBOW_EXT_THRESH:
                elbow_ext_i = i
                break
        elbow_ext_ts = ts_list[elbow_ext_i] if elbow_ext_i is not None else None

        # Wrist snap: max angular velocity of forearm vector (elbow->wrist)
        forearm_angles = []
        for i in range(len(frames)):
            pix = frames[i]['pix']
            el = (pix[elbow_idx][0], pix[elbow_idx][1])
            wr = (pix[wrist_idx][0], pix[wrist_idx][1])
            ang = math.atan2(wr[1]-el[1], wr[0]-el[0])
            forearm_angles.append(ang)
        forearm_ang_vel = [abs((forearm_angles[i] - forearm_angles[i-1]) / (ts_list[i] - ts_list[i-1] if ts_list[i] - ts_list[i-1] != 0 else median_dt)) for i in range(1, len(forearm_angles))]
        if forearm_ang_vel:
            snap_i_rel = int(np.argmax(forearm_ang_vel)) + 1
            snap_ts = ts_list[snap_i_rel]
            snap_ang_v = float(forearm_ang_vel[snap_i_rel-1])
        else:
            snap_i_rel = None
            snap_ts = None
            snap_ang_v = 0.0

        # Sequencing
        leg_drive_before_arm = None
        leg_to_elbow_delay = None
        if hip_ext_ts and elbow_ext_ts:
            leg_drive_before_arm = hip_ext_ts < elbow_ext_ts
            leg_to_elbow_delay = float((elbow_ext_ts - hip_ext_ts))

        # Release-relative heights (normalized)
        rel_head_y = (head_positions[release_i]['y'] - wrist_positions[release_i]['y']) / scale
        rel_shoulder_y = (shoulder_positions[release_i]['y'] - wrist_positions[release_i]['y']) / scale

        # Knee angles at load and at release
        knee_at_load = float(knee_angles[load_i])
        knee_at_release = float(knee_angles[release_i])

        # Elbow angles at set (start), load, release, follow-through
        elbow_at_set = float(elbow_angles[0])
        elbow_at_load = float(elbow_angles[load_i])
        elbow_at_release = float(elbow_angles[release_i])

        # Head stability: vertical variance around release frame (+/- 5 frames)
        window = 5
        r0 = max(0, release_i - window)
        r1 = min(len(head_positions)-1, release_i + window)
        head_ys = [head_positions[i]['y'] for i in range(r0, r1+1)]
        head_var = float(np.var(head_ys) / (scale*scale)) if head_ys else 0.0

        # Follow-through: detect wrist flexion after release and hold duration
        # approximate wrist flexion by change in forearm angle relative to release
        follow_start_i = None
        follow_end_i = None
        if forearm_angles:
            ref_ang = forearm_angles[release_i]
            FLEX_THRESH = 0.15  # radians
            for i in range(release_i+1, len(forearm_angles)):
                if abs(forearm_angles[i] - ref_ang) > FLEX_THRESH:
                    follow_start_i = i
                    break
            if follow_start_i is not None:
                # hold duration: how long angle stays beyond threshold
                hold_i = follow_start_i
                while hold_i < len(forearm_angles) and abs(forearm_angles[hold_i] - ref_ang) > FLEX_THRESH:
                    hold_i += 1
                follow_end_i = hold_i
        follow_start_ts = ts_list[follow_start_i] if follow_start_i is not None else None
        follow_end_ts = ts_list[follow_end_i] if follow_end_i is not None and follow_end_i < len(ts_list) else None
        follow_hold_duration = float(follow_end_ts - follow_start_ts) if follow_start_ts and follow_end_ts else 0.0

        # Build metrics (both raw and normalized where appropriate)
        shot_id = str(uuid.uuid4())
        shot = {
            'id': shot_id,
            'detection_window': detection_window,
            'fps': float(fps),
            'phases': {
                'set': {'ts': float(start_ts)},
                'load': {'ts': float(load_ts), 'knee_angle_deg': knee_at_load},
                'hip_extension_start': {'ts': float(hip_ext_ts) if hip_ext_ts else None},
                'elbow_extension_start': {'ts': float(elbow_ext_ts) if elbow_ext_ts else None},
                'wrist_snap': {'ts': float(snap_ts) if snap_ts else None, 'angular_velocity_rad_s': snap_ang_v},
                'release': {'ts': float(release_ts)} ,
                'follow_through': {'start_ts': follow_start_ts, 'end_ts': follow_end_ts, 'hold_duration': follow_hold_duration}
            },
            'timing': {
                'leg_drive_before_arm_extension': leg_drive_before_arm,
                'leg_to_elbow_delay_s': leg_to_elbow_delay
            },
            'metrics': {
                'angles': {
                    'elbow': {
                        'at_set_deg': elbow_at_set,
                        'at_load_deg': elbow_at_load,
                        'at_release_deg': elbow_at_release
                    },
                    'knee': {
                        'min_during_load_deg': knee_at_load,
                        'at_release_deg': knee_at_release
                    },
                    'hip': {
                        'at_load_deg': float(hip_angles[load_i]) if hip_angles else 0.0,
                        'peak_extension_deg': float(max(hip_angles)) if hip_angles else 0.0
                    }
                },
                'velocities': {
                    'peak_wrist_vertical_px_s': float(np.max(np.diff(wrist_ys) / np.maximum(np.array(dts), 1e-6))) if len(dts) else 0.0,
                    'peak_forearm_angular_velocity_rad_s': float(max(forearm_ang_vel)) if forearm_ang_vel else 0.0
                },
                'release': {
                    'ts': float(release_ts),
                    'wrist_y_px': float(wrist_positions[release_i]['y']),
                    'wrist_above_head_norm': rel_head_y,
                    'wrist_above_shoulder_norm': rel_shoulder_y
                },
                'follow_through': {
                    'hold_duration_s': follow_hold_duration
                },
                'stability': {
                    'head_vertical_variance_norm': head_var
                }
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
