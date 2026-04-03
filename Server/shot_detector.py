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
        self.min_landmark_visibility = 0.55

        # landmarks indices
        self.RW = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value
        self.LW = mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
        self.RE = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value
        self.LE = mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value
        self.RS = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
        self.LS = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        self.MID_HIP = mp.solutions.pose.PoseLandmark.LEFT_HIP.value

    def _visibility(self, entry, idx):
        try:
            return float(entry['pix'][idx][3])
        except Exception:
            return 0.0

    def _has_visibility(self, entry, indices, min_visibility=None):
        threshold = self.min_landmark_visibility if min_visibility is None else float(min_visibility)
        return all(self._visibility(entry, idx) >= threshold for idx in indices)

    def _safe_float(self, value):
        return None if value is None else float(value)

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

    def update(self, lm_list, frame_w, frame_h, ts, ball_state=None):
        # store normalized landmark list + pixel positions + timestamp
        entry = {'lm': lm_list, 'ts': ts}
        # compute pixel positions for convenience
        pix = []
        for lm in lm_list:
            pix.append((float(lm.x) * frame_w, float(lm.y) * frame_h, float(lm.z), float(getattr(lm, 'visibility', 0.0))))
        entry['pix'] = pix
        entry['ball'] = ball_state if isinstance(ball_state, dict) else None
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

        upper_indices = [wrist_idx, elbow_idx, shoulder_idx]
        lower_indices = [hip_idx, knee_idx, ankle_idx]

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
            if not self._has_visibility(entry, upper_indices):
                return None
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
            if not self._has_visibility(entry, lower_indices):
                return None
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
            arm_start = elbow_prev is not None and elbow_now is not None and (vy_norm > VY_START_NORM) and ((elbow_now - elbow_prev) > ELBOW_EXTENSION_MIN)
            knee_start = False
            if knee_now is not None:
                baseline_candidates = [knee_angle_at(e) for e in list(self.buf)[:-2]] if len(self.buf) > 5 else [knee_prev]
                baseline_vals = [v for v in baseline_candidates if v is not None]
                if baseline_vals:
                    baseline_knee = float(np.mean(baseline_vals))
                    knee_drop = baseline_knee - knee_now
                    knee_start = knee_drop > KNEE_BEND_START_DEG

            if arm_start or knee_start:
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
        upper_indices = [wrist_idx, elbow_idx, shoulder_idx]
        lower_indices = [hip_idx, knee_idx, ankle_idx]

        wrist_vis = [self._visibility(f, wrist_idx) for f in frames]
        knee_vis = [self._visibility(f, knee_idx) for f in frames]
        upper_vis_ratio = float(np.mean([1.0 if self._has_visibility(f, upper_indices) else 0.0 for f in frames])) if frames else 0.0
        lower_vis_ratio = float(np.mean([1.0 if self._has_visibility(f, lower_indices) else 0.0 for f in frames])) if frames else 0.0
        wrist_vis_ratio = float(np.mean([1.0 if v >= self.min_landmark_visibility else 0.0 for v in wrist_vis])) if wrist_vis else 0.0
        knee_vis_ratio = float(np.mean([1.0 if v >= self.min_landmark_visibility else 0.0 for v in knee_vis])) if knee_vis else 0.0

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

            elbow_angles.append(self._angle(sh, el, wr) if self._has_visibility(f, upper_indices) else np.nan)
            knee_angles.append(self._angle(hip, knee, ankle) if self._has_visibility(f, lower_indices) else np.nan)
            hip_angles.append(self._angle(sh, hip, knee) if self._has_visibility(f, lower_indices) else np.nan)
            wrist_positions.append({'x': wr[0], 'y': wr[1]})
            shoulder_positions.append({'x': sh[0], 'y': sh[1]})
            head_positions.append({'x': nose[0], 'y': nose[1]})

        # choose a representative scale (median)
        scale = float(np.median(scales)) if scales else 1.0

        # release (apex) based on wrist y (minimum y)
        wrist_ys = [p['y'] for p in wrist_positions]
        wrist_candidate_indices = [i for i, v in enumerate(wrist_vis) if v >= self.min_landmark_visibility]
        release_i = min(wrist_candidate_indices, key=lambda i: wrist_ys[i]) if wrist_candidate_indices else int(np.argmin(wrist_ys))
        release_ts = ts_list[release_i]

        # Load: maximum knee bend (minimum knee angle)
        valid_knee_indices = [i for i, v in enumerate(knee_angles) if np.isfinite(v)]
        load_i = min(valid_knee_indices, key=lambda i: knee_angles[i]) if valid_knee_indices else None
        load_ts = ts_list[load_i] if load_i is not None else None

        # Hip extension start: first index after load where hip angle increases by threshold
        HIP_EXT_THRESH = 4.0
        hip_at_load = hip_angles[load_i] if load_i is not None and np.isfinite(hip_angles[load_i]) else None
        hip_ext_i = None
        if load_i is not None and hip_at_load is not None:
            for i in range(load_i, len(hip_angles)):
                if np.isfinite(hip_angles[i]) and hip_angles[i] - hip_at_load > HIP_EXT_THRESH:
                    hip_ext_i = i
                    break
        hip_ext_ts = ts_list[hip_ext_i] if hip_ext_i is not None else None

        # Elbow extension start: first index where elbow angle increases notably before release
        ELBOW_EXT_THRESH = 5.0
        elbow_vals_before_release = [v for v in elbow_angles[:release_i+1] if np.isfinite(v)]
        elbow_min_before_release = float(min(elbow_vals_before_release)) if elbow_vals_before_release else None
        elbow_ext_i = None
        elbow_start_i = load_i if load_i is not None else 0
        if elbow_min_before_release is not None:
            for i in range(elbow_start_i, release_i+1):
                if np.isfinite(elbow_angles[i]) and elbow_angles[i] - elbow_min_before_release > ELBOW_EXT_THRESH:
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
        if hip_ext_ts is not None and elbow_ext_ts is not None:
            leg_drive_before_arm = hip_ext_ts < elbow_ext_ts
            leg_to_elbow_delay = float((elbow_ext_ts - hip_ext_ts))

        # Release-relative heights (normalized)
        rel_head_y = (head_positions[release_i]['y'] - wrist_positions[release_i]['y']) / scale
        rel_shoulder_y = (shoulder_positions[release_i]['y'] - wrist_positions[release_i]['y']) / scale

        # Knee angles at load and at release
        knee_at_load = float(knee_angles[load_i]) if load_i is not None and np.isfinite(knee_angles[load_i]) else None
        knee_at_release = float(knee_angles[release_i]) if np.isfinite(knee_angles[release_i]) else None

        # Elbow angles at set (start), load, release, follow-through
        elbow_at_set = float(elbow_angles[0]) if np.isfinite(elbow_angles[0]) else None
        elbow_at_load = float(elbow_angles[load_i]) if load_i is not None and np.isfinite(elbow_angles[load_i]) else None
        elbow_at_release = float(elbow_angles[release_i]) if np.isfinite(elbow_angles[release_i]) else None

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
        follow_hold_duration = float(follow_end_ts - follow_start_ts) if follow_start_ts is not None and follow_end_ts is not None else 0.0

        quality_score = 0.6 * upper_vis_ratio + 0.4 * lower_vis_ratio
        if quality_score >= 0.8:
            confidence = 'high'
        elif quality_score >= 0.55:
            confidence = 'medium'
        else:
            confidence = 'low'

        ball_states = [f.get('ball') for f in frames]
        ball_supported = any(isinstance(state, dict) for state in ball_states)
        supported_ball_states = [state for state in ball_states if isinstance(state, dict)]
        ball_presence_ratio = 0.0
        if supported_ball_states:
            detected_count = sum(1 for state in supported_ball_states if bool(state.get('detected')))
            ball_presence_ratio = float(detected_count / len(supported_ball_states))

        in_hand_scores = [float(state.get('in_hand_score')) for state in supported_ball_states if state.get('in_hand_score') is not None]
        ball_in_hand_score = float(np.mean(in_hand_scores)) if in_hand_scores else None

        palm_gap_vals = [float(state.get('palm_gap_px')) for state in supported_ball_states if state.get('palm_gap_px') is not None]
        palm_gap_px_mean = float(np.mean(palm_gap_vals)) if palm_gap_vals else None
        palm_gap_px_std = float(np.std(palm_gap_vals)) if len(palm_gap_vals) > 1 else (0.0 if len(palm_gap_vals) == 1 else None)

        ball_in_hand_confirmed = ball_in_hand_score is not None and ball_in_hand_score >= 0.6 and ball_presence_ratio >= 0.5
        grip_feedback_eligible = ball_in_hand_confirmed and palm_gap_px_mean is not None
        allow_shot_feedback = bool(ball_in_hand_confirmed)

        conservative_feedback = confidence == 'low' or lower_vis_ratio < 0.45
        if not ball_supported:
            feedback_message = 'Ball context is not available in this run, so shot feedback is disabled until ball-in-hand evidence is available.'
        elif not ball_in_hand_confirmed:
            feedback_message = 'Ball-in-hand evidence is weak for this shot. Shot feedback is disabled to avoid overconfident conclusions.'
        elif conservative_feedback:
            feedback_message = 'Ball-in-hand is confirmed, but tracking confidence is limited due to partial body visibility. Keep all major joints in frame (especially hips, knees, and ankles) for more accurate coaching feedback.'
        else:
            feedback_message = 'Tracking confidence is sufficient for detailed shot-form feedback.'

        # Build metrics (both raw and normalized where appropriate)
        shot_id = str(uuid.uuid4())
        shot = {
            'id': shot_id,
            'detection_window': detection_window,
            'fps': float(fps),
            'phases': {
                'set': {'ts': float(start_ts)},
                'load': {'ts': self._safe_float(load_ts), 'knee_angle_deg': self._safe_float(knee_at_load)},
                'hip_extension_start': {'ts': float(hip_ext_ts) if hip_ext_ts is not None else None},
                'elbow_extension_start': {'ts': float(elbow_ext_ts) if elbow_ext_ts is not None else None},
                'wrist_snap': {'ts': float(snap_ts) if snap_ts is not None else None, 'angular_velocity_rad_s': snap_ang_v},
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
                        'at_set_deg': self._safe_float(elbow_at_set),
                        'at_load_deg': self._safe_float(elbow_at_load),
                        'at_release_deg': self._safe_float(elbow_at_release)
                    },
                    'knee': {
                        'min_during_load_deg': self._safe_float(knee_at_load),
                        'at_release_deg': self._safe_float(knee_at_release)
                    },
                    'hip': {
                        'at_load_deg': self._safe_float(hip_angles[load_i]) if load_i is not None and np.isfinite(hip_angles[load_i]) else None,
                        'peak_extension_deg': self._safe_float(np.nanmax(hip_angles)) if np.any(np.isfinite(hip_angles)) else None
                    }
                },
                'velocities': {
                    'peak_wrist_vertical_px_s': float(np.max(-np.diff(wrist_ys) / np.maximum(np.array(dts), 1e-6))) if len(dts) else 0.0,
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
            'data_quality': {
                'confidence': confidence,
                'upper_body_visibility_ratio': upper_vis_ratio,
                'lower_body_visibility_ratio': lower_vis_ratio,
                'wrist_visibility_ratio': wrist_vis_ratio,
                'knee_visibility_ratio': knee_vis_ratio,
                'occlusion_flags': {
                    'upper_body_occluded': upper_vis_ratio < 0.5,
                    'lower_body_occluded': lower_vis_ratio < 0.5,
                    'wrist_often_missing': wrist_vis_ratio < 0.6,
                    'knee_often_missing': knee_vis_ratio < 0.6
                }
            },
            'ball_context': {
                'supported': ball_supported,
                'ball_presence_ratio': ball_presence_ratio,
                'ball_in_hand_score': self._safe_float(ball_in_hand_score),
                'ball_in_hand_confirmed': ball_in_hand_confirmed,
                'palm_gap_px_mean': self._safe_float(palm_gap_px_mean),
                'palm_gap_px_std': self._safe_float(palm_gap_px_std),
                'grip_feedback_eligible': grip_feedback_eligible
            },
            'feedback_guardrails': {
                'mode': 'conservative' if conservative_feedback else 'normal',
                'message': feedback_message,
                'allow_shot_feedback': allow_shot_feedback,
                'allow_grip_feedback': grip_feedback_eligible
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
