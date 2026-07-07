import joblib
import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import os
import csv
import uuid
from datetime import datetime
from collections import deque
from flask          import Flask, Response, jsonify, send_from_directory, send_file, request
from flask_cors     import CORS
from flask_socketio import SocketIO, emit
import pandas as pd

FEATURE_SCHEMA = [
    "ear_ratio",
    "blink_rate",
    "blink_irregularity",
    "blink_per_10s",
    "gaze_stability",
    "off_screen_pct",
    "head_steadiness",
    "brow_adjusted",
    "fatigue_index",
    "overload_index",
    "focus_stability",
    "fatigue_trend",
    "emo_angry",
    "emo_disgust",
    "emo_fear",
    "emo_happy",
    "emo_sad",
    "emo_surprise",
    "emo_neutral"
]

ml_model = joblib.load("cognitive_model_v7.pkl")
ml_encoder = joblib.load("label_encoder_v7.pkl")
scaler = joblib.load("scaler_v7.pkl")

def build_ml_features(eye, cog, emo):
    f = {
        "ear_ratio": eye.get("ear_avg", 0),
        "blink_rate": eye.get("blink_rate", 0),
        "blink_irregularity": eye.get("blink_rate", 0),
        "blink_per_10s": eye.get("blink_rate", 0) / 6,
        "gaze_stability": eye.get("gaze_stability", 0),
        "off_screen_pct": eye.get("off_screen_pct", 0),
        "head_steadiness": eye.get("head_steadiness", 0),
        "brow_adjusted": eye.get("brow_raise", 0),

        "fatigue_index": int(eye.get("blink_rate", 0) < 6) + int(eye.get("micro_sleep", 0)),
        "overload_index": cog.get("confusion_score", 0) + cog.get("overload_score", 0),
        "focus_stability": (eye.get("gaze_stability", 0) * eye.get("head_steadiness", 0)) / 100,
        "fatigue_trend": eye.get("blink_rate", 0),

        "emo_angry": emo.get("angry", 0),
        "emo_disgust": emo.get("disgust", 0),
        "emo_fear": emo.get("fear", 0),
        "emo_happy": emo.get("happy", 0),
        "emo_sad": emo.get("sad", 0),
        "emo_surprise": emo.get("surprise", 0),
        "emo_neutral": emo.get("neutral", 0),
    }

    return f

def align_features(features: dict):
    return {col: features.get(col, 0) for col in FEATURE_SCHEMA}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 1 â€” DATASET RECORDER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

DATASET_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
MASTER_CSV_PATH = os.path.join(DATASET_DIR, 'combined_master_dataset.csv')
os.makedirs(DATASET_DIR, exist_ok=True)

ALL_STATES = ['focused', 'distracted', 'high_cognitive_load', 'fatigued']

STATE_ALIASES = {
    'light_load': 'focused',
    'overloaded': 'high_cognitive_load',
    'confused': 'high_cognitive_load',
}

STATE_LABELS = {
    'focused': 'Focused',
    'distracted': 'Distracted',
    'high_cognitive_load': 'High Cognitive Load',
    'fatigued': 'Fatigued',
}

def normalize_state(state):
    if state is None:
        return None
    return STATE_ALIASES.get(state, state)

CSV_COLUMNS = [
    'timestamp', 'session_id', 'session_second',
    'ear_left', 'ear_right', 'ear_avg', 'ear_smooth', 'openness_pct',
    'blink_count', 'blink_rate', 'blink_irregularity', 'is_drowsy', 'micro_sleep',
    'gaze_direction', 'gaze_h', 'gaze_v', 'gaze_stability', 'off_screen_pct',
    'brow_raise', 'brow_adjusted',
    'head_pitch', 'head_yaw', 'head_roll', 'head_label', 'head_steadiness',
    'pose_reliable',
    'emotion_dominant',
    'emo_angry', 'emo_disgust', 'emo_fear',
    'emo_happy', 'emo_sad', 'emo_surprise', 'emo_neutral',
    'focus_score', 'avg_score', 'confidence',
    'sig_ear', 'sig_blink', 'sig_gaze', 'sig_head', 'sig_emotion', 'sig_brow',
    'sig_blink_pattern', 'attention_trend',
    'confusion_score', 'overload_score',
    'state_label', 'label_source', 'state_changed',
]

RECORD_INTERVAL_NORMAL = 0.5
RECORD_INTERVAL_RARE   = 0.25


class DatasetRecorder:

    def __init__(self):
        self.data = []   # ðŸ”¥ THIS IS REQUIRED
        
        self._session_id         = None
        self._file               = None
        self._writer             = None
        self._last_save          = 0
        self._row_count          = 0
        self._total_rows         = 0
        self._lock               = threading.Lock()
        self.recording           = False
        self._state_counts       = {s: 0 for s in ALL_STATES}
        self._total_state_counts = {s: 0 for s in ALL_STATES}
        self._last_state         = None
        self._manual_label       = None
        self._manual_until       = 0
        self._count_existing_rows()

    def _count_existing_rows(self):
        if os.path.exists(MASTER_CSV_PATH):
            try:
                with open(MASTER_CSV_PATH, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._total_rows += 1
                        s = normalize_state(row.get('state_label', ''))
                        if s in self._total_state_counts:
                            self._total_state_counts[s] += 1
            except Exception:
                self._total_rows = 0
        print(f'[Dataset] {self._total_rows} existing rows | {self._total_state_counts}')

    def start_session(self):
        self._session_id   = str(uuid.uuid4())[:8]
        self._row_count    = 0
        self._last_save    = 0
        self._last_state   = None
        self._state_counts = {s: 0 for s in ALL_STATES}
        file_exists        = os.path.exists(MASTER_CSV_PATH)
        self._file         = open(MASTER_CSV_PATH, 'a', newline='', encoding='utf-8')
        self._writer       = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        if not file_exists or os.path.getsize(MASTER_CSV_PATH) == 0:
            self._writer.writeheader()
        self._file.flush()
        self.recording = True
        return MASTER_CSV_PATH

    def force_label(self, label: str, duration: float = 5.0):
        if label not in ALL_STATES:
            return False
        self._manual_label = label
        self._manual_until = time.time() + duration
        print(f'[Dataset] Manual label: {label} for {duration}s')
        return True

    def record(self, eye, pose, emo, cog, session_time):
        try:
            row = {
                "ear_avg": eye.get("ear_avg", 0),
                "blink_rate": eye.get("blink_rate", 0),
                "gaze_stability": eye.get("gaze_stability", 0),
                "off_screen_pct": eye.get("off_screen_pct", 0),
                "head_steadiness": eye.get("head_steadiness", 0),
                "is_drowsy": int(eye.get("is_drowsy", False)),
                "micro_sleep": int(eye.get("micro_sleep", False)),
                "brow_raise": eye.get("brow_raise", 0),
                "gaze_h": eye.get("gaze_h", 0),
                "gaze_v": eye.get("gaze_v", 0),

                "attention_trend": cog.get("attention_trend", 0),
                "confusion_score": cog.get("confusion_score", 0),
                "overload_score": cog.get("overload_score", 0),

                "state_label": cog.get("state", "unknown"),
                "session_time": session_time
            }

            self.data.append(row)

            self._row_count += 1
            self._state_counts[cog.get("state", "unknown")] = \
                                                self._state_counts.get(cog.get("state", "unknown"), 0) + 1

        except Exception as e:
            print("[RECORDER ERROR]", e)

    def stop_session(self):
        self.recording = False

        print(f'[Dataset] Session: {self._row_count} rows | Counts: {self._state_counts}')

        self._row_count = 0
        self._state_counts = {s: 0 for s in ALL_STATES}

        return self._row_count, len(self.data)

    def get_stats(self):
        try:
            return {
                "total_rows": len(self.data),
                "session_rows": self._row_count,
                "counts": self._state_counts,
                "manual_label_active": self._manual_label is not None
            }
        except Exception as e:
            print("[STATS ERROR]", e)
            return {
                "total_rows": 0,
                "session_rows": 0,
                "counts": {}
            }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 2 â€” VOICE ALERT ENGINE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

VOICE_MESSAGES = {
    'fatigued':    ["You look tired. Please take a 5 minute break.",
                    "Fatigue detected. Rest your eyes for a few minutes.",
                    "Your eyes are closing. Take a short break now."],
    'high_cognitive_load':  ["High cognitive load detected. Slow down and breathe.",
                             "You seem mentally overloaded. Focus on one concept at a time.",
                             "Take a deep breath. Break the topic into smaller parts."],
    'distracted':  ["You seem distracted. Bring your focus back to the screen.",
                    "Attention drift detected. Try to refocus on your study material.",
                    "Stay focused. You are almost there."],
    'micro_sleep': ["Wake up! You are falling asleep.",
                    "Micro sleep detected. Please take a break immediately."],
}

VOICE_COOLDOWN = {
    'fatigued':   90,
    'high_cognitive_load': 60,
    'distracted': 60,
    'micro_sleep':30,
}


class VoiceAlertEngine:
    def __init__(self):
        self._engine      = None
        self._ok          = None
        self._queue       = queue.Queue()
        self._last_spoken = {}
        self._msg_index   = {}
        self._thread      = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self.enabled      = True

    def _init_engine(self):
        if self._ok is not None:
            return self._ok
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate',   165)
            self._engine.setProperty('volume', 0.92)
            self._ok = True
        except Exception as e:
            print(f'[Voice] pyttsx3 not available: {e}')
            self._ok = False
        return self._ok

    def _worker(self):
        while True:
            msg = self._queue.get()
            if self._init_engine() and self._engine:
                try:
                    self._engine.say(msg)
                    self._engine.runAndWait()
                except Exception as e:
                    print(f'[Voice] Error: {e}')

    def speak(self, alert_type: str):
        if not self.enabled:
            return False
        now  = time.time()
        last = self._last_spoken.get(alert_type, 0)
        if now - last < VOICE_COOLDOWN.get(alert_type, 60):
            return False
        msgs = VOICE_MESSAGES.get(alert_type, [])
        if not msgs:
            return False
        idx                           = self._msg_index.get(alert_type, 0) % len(msgs)
        self._msg_index[alert_type]   = idx + 1
        self._last_spoken[alert_type] = now
        self._queue.put(msgs[idx])
        return True

    def speak_custom(self, text: str):
        if self.enabled:
            self._queue.put(text)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 3 â€” EYE TRACKER
#  v5.1 FIX D: EAR smoothing changed from mean â†’ 30th-percentile of last 15
#              frames. Suppresses single-frame right-EAR landmark spikes
#              (0.28 â†’ 1.52 seen in CSV) without dulling genuine blink drops.
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

LEFT_EYE_INDICES   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES  = [33,  160, 158, 133, 153, 144]
LEFT_IRIS_INDICES  = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]
LEFT_BROW_INDICES  = [336, 296, 334, 293, 300]
RIGHT_BROW_INDICES = [107,  66, 105,  63,  70]


class EyeTracker:

    def __init__(self):
        self.blink_count           = 0
        self.frame_counter         = 0
        self.drowsy_counter        = 0
        self.is_drowsy             = False
        self.blink_times           = deque(maxlen=120)
        self.ear_history           = deque(maxlen=120)
        self.gaze_history          = deque(maxlen=60)
        self.closed_since          = None
        self._calibration_ears     = deque(maxlen=60)
        self._calibrated           = False
        self.ear_open_baseline     = 0.30
        self.ear_blink_thresh      = 0.22
        self.ear_drowsy_thresh     = 0.20
        self.recent_blink_gaps     = deque(maxlen=20)
        self._last_blink_time      = None
        self._drowsy_frames_needed = 12
        self._awake_frames_needed  = 8

    def _calibrate(self, ear):
        if self._calibrated:
            return
        if ear > 0.20:
            self._calibration_ears.append(ear)
        if len(self._calibration_ears) >= 60:
            baseline               = np.percentile(list(self._calibration_ears), 70)
            self.ear_open_baseline = round(baseline, 3)
            self.ear_blink_thresh  = round(baseline * 0.72, 3)
            self.ear_drowsy_thresh = round(baseline * 0.62, 3)
            self._calibrated       = True

    @staticmethod
    def _ear(landmarks, indices, w, h):
        pts = [np.array([landmarks[i].x*w, landmarks[i].y*h]) for i in indices]
        A   = np.linalg.norm(pts[1]-pts[5])
        B   = np.linalg.norm(pts[2]-pts[4])
        C   = np.linalg.norm(pts[0]-pts[3])
        return round(float((A+B)/(2.0*C)) if C else 0.0, 4)

    @staticmethod
    def _gaze(landmarks, eye_idx, iris_idx, w, h):
        lc, rc  = landmarks[eye_idx[0]], landmarks[eye_idx[3]]
        eye_w   = abs(rc.x-lc.x)*w
        if eye_w < 1:
            return 'center', 0.0, 0.0
        iris_cx = np.mean([landmarks[i].x for i in iris_idx])*w
        iris_cy = np.mean([landmarks[i].y for i in iris_idx])*h
        eye_cx  = ((lc.x+rc.x)/2)*w
        eye_cy  = ((lc.y+rc.y)/2)*h
        h_r     = (iris_cx-eye_cx)/(eye_w/2)
        v_r     = (iris_cy-eye_cy)/(eye_w/2)
        d       = ('left'  if h_r < -0.18 else
                   'right' if h_r >  0.18 else
                   'up'    if v_r < -0.18 else
                   'down'  if v_r >  0.18 else 'center')
        return d, round(h_r,3), round(v_r,3)

    def _brow_raise(self, landmarks, w, h):
        def dist(bi, ei):
            by = np.mean([landmarks[i].y for i in bi])*h
            ey = np.mean([landmarks[i].y for i in ei])*h
            fh = abs(landmarks[10].y-landmarks[152].y)*h
            return max(0.0, (ey-by)/max(fh,1))
        l = dist(LEFT_BROW_INDICES,  LEFT_EYE_INDICES)
        r = dist(RIGHT_BROW_INDICES, RIGHT_EYE_INDICES)
        return round(float(np.clip(((l+r)/2 - 0.06)/0.10, 0, 1)), 3)

    def update(self, landmarks, w, h):
        left_ear  = self._ear(landmarks, LEFT_EYE_INDICES,  w, h)
        right_ear = self._ear(landmarks, RIGHT_EYE_INDICES, w, h)
        avg_ear   = round((left_ear+right_ear)/2, 4)
        self._calibrate(avg_ear)
        self.ear_history.append(avg_ear)

        # FIX D: use 30th-percentile instead of mean to suppress right-EAR
        # landmark spikes (single frames where right_ear jumps to 1.2-1.5)
        # while still responding quickly to genuine eye closure drops.
        raw_15     = list(self.ear_history)[-15:]
        smooth_ear = round(float(np.percentile(raw_15, 30)), 4)

        blink_detected = False
        now            = time.time()

        if smooth_ear < self.ear_blink_thresh:
            self.frame_counter += 1
            if self.closed_since is None:
                self.closed_since = now
        else:
            if self.frame_counter >= 2:
                self.blink_count += 1
                blink_detected    = True
                self.blink_times.append(now)
                if self._last_blink_time:
                    self.recent_blink_gaps.append(now - self._last_blink_time)
                self._last_blink_time = now
            self.frame_counter = 0
            self.closed_since  = None

        micro_sleep = bool(self.closed_since and (now - self.closed_since) > 1.5)

        if smooth_ear < self.ear_drowsy_thresh:
            self.drowsy_counter += 1
            if self.drowsy_counter >= self._drowsy_frames_needed:
                self.is_drowsy = True
        else:
            self.drowsy_counter = max(0, self.drowsy_counter - 1)
            if self.drowsy_counter == 0:
                self.is_drowsy = False

        blink_rate = len([t for t in self.blink_times if now-t <= 60])

        direction, h_r, v_r = self._gaze(
            landmarks, LEFT_EYE_INDICES, LEFT_IRIS_INDICES, w, h)
        self.gaze_history.append(direction)
        recent         = list(self.gaze_history)[-30:]
        gaze_stability = round(
            sum(1 for g in recent if g=='center') / max(len(recent),1)*100, 1)
        ten_sec        = list(self.gaze_history)[-60:]
        off_screen_pct = round(
            sum(1 for g in ten_sec if g in ('left','right')) / max(len(ten_sec),1)*100, 1)

        brow_raise   = self._brow_raise(landmarks, w, h)
        openness_pct = round(
            min(100, (smooth_ear/max(self.ear_open_baseline,0.01))*100), 1)

        blink_irregularity = 0.0
        if len(self.recent_blink_gaps) >= 5:
            gaps = list(self.recent_blink_gaps)
            blink_irregularity = round(
                float(np.std(gaps) / max(np.mean(gaps), 0.1)), 3)

        return {
            'left_ear':           left_ear,
            'right_ear':          right_ear,
            'avg_ear':            avg_ear,
            'smooth_ear':         smooth_ear,
            'openness_pct':       openness_pct,
            'blink_count':        self.blink_count,
            'blink_detected':     blink_detected,
            'blink_rate':         blink_rate,
            'is_drowsy':          self.is_drowsy,
            'micro_sleep':        micro_sleep,
            'gaze':               {'direction':direction,'h':h_r,'v':v_r},
            'gaze_stability':     gaze_stability,
            'off_screen_pct':     off_screen_pct,
            'brow_raise':         brow_raise,
            'calibrated':         self._calibrated,
            'ear_baseline':       self.ear_open_baseline,
            'blink_irregularity': blink_irregularity,
        }

    def reset(self):
        self.blink_count=self.frame_counter=self.drowsy_counter=0
        self.is_drowsy=False; self.closed_since=None; self._calibrated=False
        self._calibration_ears.clear(); self.blink_times.clear()
        self.ear_history.clear(); self.gaze_history.clear()
        self.recent_blink_gaps.clear(); self._last_blink_time=None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 4 â€” HEAD POSE ESTIMATOR
#  v5.1 FIX E: Yaw sanity limit tightened 75Â° â†’ 65Â° so near-threshold
#              artifacts (yaw 66-74Â° seen in CSV) are also flagged unreliable.
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

MODEL_3D = np.array([
    ( 0.0,    0.0,   0.0),
    ( 0.0,  -63.6, -12.5),
    (-43.3,  32.7, -26.0),
    ( 43.3,  32.7, -26.0),
    (-28.9, -28.9, -24.1),
    ( 28.9, -28.9, -24.1),
], dtype=np.float64)
HEAD_IDX = [1, 152, 33, 263, 61, 291]


class HeadPoseEstimator:
    def __init__(self):
        self._cam            = None
        self._dist           = np.zeros((4,1), dtype=np.float64)
        self._ph             = deque(maxlen=10)
        self._yh             = deque(maxlen=10)
        self._turned_counter = 0
        self._turned_needed  = 8
        self._is_turned      = False

    def estimate(self, landmarks, w, h):
        if self._cam is None or self._cam[0][2] != w/2:
            self._cam = np.array(
                [[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)

        pts = np.array(
            [(landmarks[i].x*w, landmarks[i].y*h) for i in HEAD_IDX],
            dtype=np.float64)
        try:
            ok,rv,tv = cv2.solvePnP(
                MODEL_3D, pts, self._cam, self._dist,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return self._default()

            rm,_              = cv2.Rodrigues(rv)
            _,_,_,_,_,_,euler = cv2.decomposeProjectionMatrix(np.hstack([rm,tv]))
            p = float(euler[0][0])
            y = float(euler[1][0])
            r = float(euler[2][0])

            # FIX E: tightened yaw limit 75Â° â†’ 65Â° â€” CSV showed yaw=66-74Â°
            # still leaking through the old check and affecting head_steadiness
            if abs(p) > 85 or abs(y) > 65:
                return {
                    'pitch':0.0,'yaw':round(y,1),'roll':round(r,1),
                    'label':'upright',
                    'steadiness':50.0,
                    'pose_reliable':False,
                }

            self._ph.append(p)
            self._yh.append(y)
            p = float(np.mean(self._ph))
            y = float(np.mean(self._yh))

            if p > 18:
                raw_label = 'head_drop'
            elif p < -18:
                raw_label = 'forward_lean'
            elif abs(r) > 18:
                raw_label = 'tilted'
            else:
                raw_label = 'upright'

            if abs(y) > 32:
                self._turned_counter += 1
                if self._turned_counter >= self._turned_needed:
                    self._is_turned = True
            else:
                self._turned_counter = max(0, self._turned_counter - 2)
                if self._turned_counter == 0:
                    self._is_turned = False

            label      = 'turned_away' if self._is_turned else raw_label
            steadiness = round(max(0.0, 100 - (abs(p)+abs(y)+abs(r))*1.1), 1)

            return {
                'pitch':round(p,1),'yaw':round(y,1),'roll':round(r,1),
                'label':label,'steadiness':steadiness,
                'pose_reliable':True,
            }
        except:
            return self._default()

    @staticmethod
    def _default():
        return {
            'pitch':0.0,'yaw':0.0,'roll':0.0,
            'label':'upright','steadiness':100.0,
            'pose_reliable':True,
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 5 â€” EMOTION DETECTOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

EMOTION_SCORE = {
    'happy':75.0, 'neutral':65.0, 'surprise':50.0,
    'sad':30.0,   'fear':25.0,   'angry':20.0,   'disgust':15.0,
}


class EmotionDetector:
    def __init__(self, run_every=20):
        self.run_every   = run_every
        self.frame_count = 0
        self.last_result = {}
        self._ok         = None

    def _check(self):
        if self._ok is None:
            try:
                import deepface
                self._ok = True
            except:
                self._ok = False
        return self._ok

    def detect(self, frame):
        self.frame_count += 1
        if self.frame_count % self.run_every != 0:
            return self.last_result
        if not self._check():
            return self._mock()
        try:
            from deepface import DeepFace
            res = DeepFace.analyze(
                frame, actions=['emotion'],
                enforce_detection=False, silent=True)
            if res:
                face     = res[0]
                emotions = face.get('emotion', {})
                dominant = face.get('dominant_emotion', 'neutral')
                total    = sum(emotions.values()) or 1
                self.last_result = {
                    'dominant_emotion': dominant,
                    'scores': {k:round(v/total,4) for k,v in emotions.items()},
                    'face_detected': True,
                }
        except:
            self.last_result = self.last_result or self._mock()
        return self.last_result

    @staticmethod
    def _mock():
        return {
            'dominant_emotion': 'neutral',
            'scores': {e:1/7 for e in
                       ['angry','disgust','fear','happy','sad','surprise','neutral']},
            'face_detected': False,
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 6 â€” COGNITIVE LOAD CLASSIFIER  (v5.1 core changes)
#
#  FIX A â€” overloaded detection:
#    â€¢ Confusion priority gate: avg_confusion >= 3.0 (was 2.5), score 38-60 (was 35-68)
#    â€¢ Overload combo fires at avg_overload >= 2.0 (was 2.5)
#    â€¢ forward_lean no longer required in the 52+ zone
#    â€¢ In 38-52 zone confusion must beat overload by 1.0 margin
#    â€¢ Overload signals: blink > 22, ear < 65, gaze < 50, brow > 0.45
#
#  FIX B â€” fatigued detection:
#    â€¢ Low-blink fatigue: blink < 6/min sustained â†’ fatigued even without EAR
#
#  FIX C â€” confused over-sensitivity:
#    â€¢ Confusion signals: brow > 0.50 (was 0.35), gaze < 40 (was 55),
#      birr > 0.6 (was 0.5), blink > 28 (was 25)
#
#  STATE CLASSIFICATION PRIORITY ORDER (v5.1):
#    1. micro_sleep        â†’ fatigued  (score capped 15)
#    2. is_drowsy          â†’ fatigued  (score capped 30)
#    3. low_blink_fatigue  â†’ fatigued  (score capped 35)   â† NEW
#    4. is_distracted      â†’ distracted (score capped 40)
#    5. avg_confusion >= 3.0 AND score 38-60 â†’ confused
#    6. score >= 65        â†’ focused
#    7. score >= 52        â†’ overloaded (if avg_overload >= 2.0) else light_load
#    8. score >= 38        â†’ confused OR overloaded (confusion must lead by 1.0)
#    9. score <  38        â†’ overloaded
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

WEIGHTS = {
    'ear':             0.20,
    'blink_rate':      0.12,
    'gaze_stability':  0.20,
    'head_steadiness': 0.12,
    'emotion':         0.20,
    'brow':            0.10,
    'blink_pattern':   0.06,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 0.01, "Weights must sum to 1.0"


class CognitiveLoadClassifier:
    def __init__(self):
        self.score_history      = deque(maxlen=120)
        self.state_history      = deque(maxlen=30)
        self.session_start      = time.time()
        self._last_alert_t      = {}
        self._confusion_history = deque(maxlen=20)
        self._overload_history  = deque(maxlen=20)

    def compute(self, eye, pose, emo):
        ear   = eye.get('smooth_ear',         0.30)
        base  = eye.get('ear_baseline',       0.30)
        blink = eye.get('blink_rate',         15)
        brow  = eye.get('brow_raise',         0.0)
        birr  = eye.get('blink_irregularity', 0.0)

        pose_reliable = pose.get('pose_reliable', True)
        pose_label    = pose.get('label', 'upright') if pose_reliable else 'upright'

        # â”€â”€ Signal Scores â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if eye.get('calibrated'):
            ear_score = min(100.0, (ear/max(base,0.01))*100)
        else:
            ear_score = max(0.0, min(100.0, (ear-0.18)/(0.40-0.18)*100))

        if 12 <= blink <= 20:
            blink_score = 100.0
        elif blink < 12:
            blink_score = max(0.0, 100.0-(12-blink)*9)
        else:
            blink_score = max(0.0, 100.0-(blink-20)*6)

        gaze_score = eye.get('gaze_stability', 80.0)
        head_score = pose.get('steadiness', 80.0) if pose_reliable else 65.0
        emo_score  = EMOTION_SCORE.get(emo.get('dominant_emotion','neutral'), 65.0)

        # Suppress brow penalty during forward_lean (camera artifact)
        effective_brow = brow * 0.35 if pose_label == 'forward_lean' else brow
        brow_score     = max(0.0, 100.0 - effective_brow*80)

        blink_pattern_score = max(0.0, 100.0 - birr*60)

        signals = {
            'ear':             round(ear_score,           1),
            'blink_rate':      round(blink_score,         1),
            'gaze_stability':  round(gaze_score,          1),
            'head_steadiness': round(head_score,          1),
            'emotion':         round(emo_score,           1),
            'brow':            round(brow_score,          1),
            'blink_pattern':   round(blink_pattern_score, 1),
        }

        total = max(0.0, min(100.0, sum(signals[k]*WEIGHTS[k] for k in WEIGHTS)))

        eye['brow_adjusted'] = round(effective_brow, 3)

        # â”€â”€ Hard state checks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        is_drowsy    = eye.get('is_drowsy',   False)
        micro_sleep  = eye.get('micro_sleep', False)
        off_screen   = eye.get('off_screen_pct', 0)
        dominant_emo = emo.get('dominant_emotion', 'neutral')

        is_distracted = (
            (pose_label == 'turned_away' and pose_reliable) or
            (off_screen > 60)
        )

        # FIX B: low-blink fatigue â€” blink rate < 6/min sustained over 30+
        # history frames indicates eye strain / early fatigue even without
        # EAR-based drowsiness (common when staring intensely at screen)
        low_blink_fatigue = (blink < 6 and len(self.score_history) > 30)

        # â”€â”€ Confusion combo (5 independent binary signals) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # FIX C: tightened thresholds to reduce over-triggering
        confusion_signals = [
            brow > 0.50,                                         # was 0.35
            gaze_score < 40,                                     # was 55
            birr > 0.60,                                         # was 0.50
            dominant_emo in ('fear','sad','disgust','angry'),
            blink > 28,                                          # was 25
        ]
        self._confusion_history.append(sum(confusion_signals))
        avg_confusion = float(np.mean(list(self._confusion_history)))

        # â”€â”€ Overload combo (5 independent binary signals) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # FIX A: lower thresholds so overload fires more readily
        overload_signals = [
            blink > 22,                                          # was 25
            pose_label == 'forward_lean' and pose_reliable,
            ear_score < 65,                                      # was 60
            gaze_score < 50,                                     # was 45
            brow > 0.45,                                         # was 0.50
        ]
        self._overload_history.append(sum(overload_signals))
        avg_overload = float(np.mean(list(self._overload_history)))

        # â”€â”€ Final state classification (priority order) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if micro_sleep:
            total = min(total, 15.0)
            state = 'fatigued'

        elif is_drowsy:
            total = min(total, 30.0)
            state = 'fatigued'

        else:
            session_time = time.time() - self.session_start
            long_session_fatigue = session_time > 1800  # 30 min

            if long_session_fatigue and total < 60:
                state = 'fatigued'
            elif low_blink_fatigue and total < 55:
                total = min(total, 35.0)
                state = 'fatigued'
            elif is_distracted:
                total = min(total, 40.0)
                state = 'distracted'
            elif total >= 65:
                state = 'focused'
            elif total >= 52 and avg_confusion < 3.0 and avg_overload < 2.0:
                state = 'focused'
            else:
                state = 'high_cognitive_load'

        state = normalize_state(state)
        self.score_history.append(total)
        self.state_history.append(state)

        hist        = list(self.score_history)
        trend_score = 50.0
        if len(hist) >= 20:
            mid         = len(hist)//2
            trend_score = round(min(100, max(0,
                50 + (np.mean(hist[mid:]) - np.mean(hist[:mid]))*2)), 1)

        recent_states  = list(self.state_history)[-10:]
        dominant_state = (max(set(recent_states), key=recent_states.count)
                          if recent_states else state)
        confidence     = round(
            recent_states.count(dominant_state) / max(len(recent_states),1)*100, 0)
        avg = round(float(np.mean(hist)),1) if hist else total

        LABELS = STATE_LABELS
        COLORS = {
            'focused':'#00d4a0',
            'high_cognitive_load':'#e05555',
            'fatigued':'#f0a04b',
            'distracted':'#9b7cf4',
        }

        signals['trend'] = trend_score
        alerts = self._build_alerts(eye, pose, state, blink, off_screen,
                                    micro_sleep, pose_reliable)
        advice = self._advice(state, eye, pose)

        return {
            'score':              round(total,1),
            'state':              state,
            'state_label':        LABELS.get(state,'--'),
            'state_color':        COLORS.get(state,'#888'),
            'avg_score':          avg,
            'signal_scores':      signals,
            'is_drowsy':          is_drowsy,
            'micro_sleep':        micro_sleep,
            'is_distracted':      is_distracted,
            'low_blink_fatigue':  low_blink_fatigue,
            'history':            list(self.score_history)[-40:],
            'session_secs':       int(time.time()-self.session_start),
            'confidence':         int(confidence),
            'trend_score':        trend_score,
            'advice':             advice,
            'alerts':             alerts,
            'confusion_score':    round(avg_confusion, 2),
            'overload_score':     round(avg_overload,  2),
            'pose_reliable':      pose_reliable,
        }

    def _build_alerts(self, eye, pose, state, blink, off_screen,
                      micro_sleep, pose_reliable=True):
        alerts = []
        now    = time.time()

        def push(key, msg, sev, cd=30):
            if now - self._last_alert_t.get(key, 0) > cd:
                alerts.append({'key':key,'msg':msg,'severity':sev})
                self._last_alert_t[key] = now

        if micro_sleep:                          push('microsleep','Micro-sleep detected!','alert',10)
        if eye.get('is_drowsy'):                 push('drowsy','Drowsiness detected','alert',20)
        if eye.get('low_blink_fatigue'):         push('lowblinkfat','Very low blink rate â€” eye fatigue','warn',40)
        if blink < 6:                            push('lowblink',f'Low blink rate ({blink}/min)','warn',40)
        elif blink > 28:                         push('highblink',f'High blink rate ({blink}/min)','warn',40)
        if off_screen > 60:                      push('gaze','Gaze off-screen frequently','warn',30)
        if pose_reliable:
            if pose.get('label') == 'head_drop':    push('headdrop','Head dropping forward','warn',25)
            if pose.get('label') == 'turned_away':  push('turned','Head turned away','warn',25)
            if pose.get('label') == 'forward_lean': push('lean','Leaning into screen','warn',30)
        else:
            push('pose_warn','Pose tracking unstable â€” adjust camera angle','warn',60)
        if eye.get('brow_raise',0) > 0.5:       push('brow','Brow raised - possible mental load','warn',20)
        if eye.get('blink_irregularity',0)>0.6: push('birr','Irregular blinking detected','warn',30)
        if state == 'focused' and not alerts:    push('good','Focus maintained','good',60)
        return alerts

    @staticmethod
    def _advice(state, eye, pose):
        base = {
            'focused':    ('Deep focus','Best learning state. Tackle the hardest concept now.',
                           None,'good'),
            'high_cognitive_load': ('High cognitive load','Too much at once. Focus on one sub-topic.',
                                    '2-minute pause','alert'),
            'fatigued':   ('Fatigue detected','Eyes closing. Take a 5-minute break.',
                           'Break now','alert'),
            'distracted': ('Attention drift','Gaze moved away. Try Pomodoro blocks.',
                           'Reset focus','warn'),
        }
        title,msg,action,severity = base.get(state, base['focused'])
        detail = None
        blink  = eye.get('blink_rate',15)
        if blink < 6:    detail = f'Blink rate very low ({blink}/min). Eye fatigue building up.'
        elif blink < 8:  detail = f'Blink rate low ({blink}/min). Remember to blink.'
        elif blink > 25: detail = f'Blink rate high ({blink}/min). May indicate stress.'
        if pose.get('label') == 'head_drop':    detail = 'Head dropping â€” stand up and stretch.'
        if pose.get('label') == 'forward_lean': detail = 'Leaning forward â€” sit back and breathe.'
        if not pose.get('pose_reliable', True): detail = 'Camera angle causing unreliable tracking.'
        return {'title':title,'message':msg,'action':action,'severity':severity,'detail':detail}

    def reset(self):
        self.score_history.clear(); self.state_history.clear()
        self.session_start = time.time(); self._last_alert_t = {}
        self._confusion_history.clear(); self._overload_history.clear()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 7 â€” DETECTION ENGINE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

mp_face_mesh = mp.solutions.face_mesh

def ml_predict(features):
    try:
        df = pd.DataFrame([features])

        # FORCE ALIGNMENT
        df = df.reindex(columns=FEATURE_SCHEMA, fill_value=0)

        # FEATURE NAME SAFETY (sklearn fix)
        df.columns = FEATURE_SCHEMA

        df_scaled = scaler.transform(df)

        pred = ml_model.predict(df_scaled)[0]
        return normalize_state(ml_encoder.inverse_transform([pred])[0])

    except Exception as e:
        print("ML ERROR:", e)
        return None


class DetectionEngine:
    def __init__(self, camera_index=0, target_fps=20):
        self.camera_index  = camera_index
        self.frame_delay   = 1.0/target_fps
        self.result_queue  = queue.Queue(maxsize=5)
        self._running      = False
        self._thread       = None
        self._lock         = threading.Lock()
        self._frame_bytes  = None
        self.eye_tracker   = EyeTracker()
        self.head_pose     = HeadPoseEstimator()
        self.classifier    = CognitiveLoadClassifier()
        self.emotion       = EmotionDetector(run_every=20)

    def start(self):
        if self._running:
            return
        self._running = True
        self.eye_tracker.reset()
        self.classifier.reset()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_latest(self):
        try:
            return self.result_queue.get_nowait()
        except:
            return {}

    def get_frame(self):
        with self._lock:
            return self._frame_bytes

    def _loop(self):
        fps = 0
        prev_time = time.time()
        current_time = time.time()
        fps = 1 / (current_time - prev_time + 1e-6)
        prev_time = current_time
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print('[Engine] Cannot open camera')
            self._running = False
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        CFG = dict(max_num_faces=1, refine_landmarks=True,
                   min_detection_confidence=0.65,
                   min_tracking_confidence=0.65)

        with mp_face_mesh.FaceMesh(**CFG) as fm:
            while self._running:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                mesh  = fm.process(rgb)
                rgb.flags.writeable = True

                result = {
                    'timestamp':    time.time(),
                    'face_detected':False,
                    'eye':{},'pose':{},'emotion':{},'cognitive':{},
                }

                if mesh.multi_face_landmarks:
                    lm  = mesh.multi_face_landmarks[0].landmark
                    eye = self.eye_tracker.update(lm, w, h)
                    pos = self.head_pose.estimate(lm, w, h)
                    emo = self.emotion.detect(frame)
                    cog = self.classifier.compute(eye, pos, emo)
                    result.update({
                        'face_detected':True,
                        'eye':eye,'pose':pos,'emotion':emo,'cognitive':cog,
                    })
                    frame = self._annotate(frame, eye, pos, cog, w, h, fps if fps else 0)
                else:
                    cv2.putText(frame,'No face detected',(20,40),
                                cv2.FONT_HERSHEY_SIMPLEX,0.65,(80,120,255),2)

                _, buf = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                with self._lock:
                    self._frame_bytes = buf.tobytes()

                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except:
                        pass

                
                fps = 1.0 / max((time.time() - t0), 0.001)
                sleep_t = self.frame_delay - (time.time()-t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
        cap.release()

    @staticmethod
    def _annotate(frame, eye, pose, cog, w, h, fps):
        COLOR_MAP = {
            'focused':    (29,158,117),
            'high_cognitive_load': (85,85,224),
            'fatigued':   (75,160,240),
            'distracted': (147,124,183),
        }
        color = COLOR_MAP.get(cog.get('state',''), (100,100,100))

        cv2.rectangle(frame,(10,10),(320,56),color,-1)
        cv2.putText(frame,
            f"{cog.get('state_label','?')}  "
            f"{cog.get('score',0):.0f}%  "
            f"conf:{cog.get('confidence',0):.0f}%",
            (15,38),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),2)

        cv2.putText(frame,
            f"EAR:{eye.get('smooth_ear',0):.3f}  "
            f"Blinks:{eye.get('blink_count',0)}  "
            f"Rate:{eye.get('blink_rate',0)}/min",
            (10,72),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)

        pose_lbl = pose.get('label','?')
        if not pose.get('pose_reliable',True):
            pose_lbl += '(!)'
        cv2.putText(frame,
            f"Pose:{pose_lbl}  "
            f"Gaze:{eye.get('gaze',{}).get('direction','?')}  "
            f"Brow:{eye.get('brow_raise',0):.2f}",
            (10,86),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)

        cv2.putText(frame,
            f"ConfSig:{cog.get('confusion_score',0):.1f}  "
            f"OvldSig:{cog.get('overload_score',0):.1f}  "
            f"Reliable:{cog.get('pose_reliable',True)}",
            (10,100),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,180,100),1)

        if not eye.get('calibrated'):
            cv2.putText(frame,'Calibrating...',
                        (10,114),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,200,50),1)

        if eye.get('micro_sleep'):
            cv2.rectangle(frame,(0,0),(w,h),(0,0,220),5)
            cv2.putText(frame,'MICRO-SLEEP',
                        (w//2-90,h//2),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
        elif eye.get('is_drowsy'):
            cv2.rectangle(frame,(0,0),(w,h),(0,100,200),3)
        elif eye.get('low_blink_fatigue'):
            cv2.rectangle(frame,(0,0),(w,h),(0,150,220),2)
        cv2.putText(frame, f"FPS:{int(fps)}", (500,20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        return frame


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 8 â€” FLASK APP + SOCKETIO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR,'templates')
STATIC_DIR   = os.path.join(BASE_DIR,'static')

app      = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')
engine   = DetectionEngine(camera_index=0, target_fps=20)
recorder = DatasetRecorder()
voice    = VoiceAlertEngine()
session  = {'active':False,'start':None,'log':[]}


def _log(msg):
    entry = {'time':time.strftime('%H:%M:%S'),'message':msg}
    session['log'].insert(0,entry)
    session['log'] = session['log'][:50]
    socketio.emit('log',entry)


def _build_payload(r):
    cog  = r.get('cognitive',{})
    eye  = r.get('eye',{})
    pose = r.get('pose',{})
    emo  = r.get('emotion',{})
    return {
        'face_detected':      r.get('face_detected',False),
        'score':              cog.get('score',0),
        'state':              cog.get('state',''),
        'state_label':        cog.get('state_label',''),
        'state_color':        cog.get('state_color','#888'),
        'avg_score':          cog.get('avg_score',0),
        'history':            cog.get('history',[]),
        'advice':             cog.get('advice',{}),
        'signal_scores':      cog.get('signal_scores',{}),
        'is_drowsy':          cog.get('is_drowsy',False),
        'micro_sleep':        cog.get('micro_sleep',False),
        'low_blink_fatigue':  cog.get('low_blink_fatigue',False),
        'confidence':         cog.get('confidence',0),
        'alerts':             cog.get('alerts',[]),
        'confusion_score':    cog.get('confusion_score',0),
        'overload_score':     cog.get('overload_score',0),
        'pose_reliable':      cog.get('pose_reliable',True),
        'eye':{
            'left_ear':           eye.get('left_ear',0),
            'right_ear':          eye.get('right_ear',0),
            'smooth_ear':         eye.get('smooth_ear',0),
            'openness_pct':       eye.get('openness_pct',0),
            'blink_count':        eye.get('blink_count',0),
            'blink_rate':         eye.get('blink_rate',0),
            'blink_irregularity': eye.get('blink_irregularity',0),
            'gaze':               eye.get('gaze',{}),
            'gaze_stability':     eye.get('gaze_stability',0),
            'off_screen_pct':     eye.get('off_screen_pct',0),
            'is_drowsy':          eye.get('is_drowsy',False),
            'micro_sleep':        eye.get('micro_sleep',False),
            'low_blink_fatigue':  eye.get('low_blink_fatigue',False),
            'brow_raise':         eye.get('brow_raise',0),
            'brow_adjusted':      eye.get('brow_adjusted',0),
            'calibrated':         eye.get('calibrated',False),
        },
        'pose':{
            'pitch':        pose.get('pitch',0),
            'yaw':          pose.get('yaw',0),
            'roll':         pose.get('roll',0),
            'label':        pose.get('label',''),
            'steadiness':   pose.get('steadiness',0),
            'pose_reliable':pose.get('pose_reliable',True),
        },
        'emotion':{
            'dominant': emo.get('dominant_emotion',''),
            'scores':   emo.get('scores',{}),
        },
        'session_secs':  cog.get('session_secs',0),
        'dataset_stats': recorder.get_stats(),
    }


def _emit_loop():
    while True:
        try:
            if not session.get('active'):
                time.sleep(0.1)
                continue

            r = engine.get_latest()
            if not r:
                time.sleep(0.1)
                continue

            eye  = r.get('eye', {})
            pose = r.get('pose', {})
            emo  = r.get('emotion', {})
            cog  = r.get('cognitive', {})

            # =========================
            # STEP 3: FEATURE BUILDING
            # =========================

            ml_features = build_ml_features(eye, cog, emo)
            ml_features = align_features(ml_features)

            ml_state = ml_predict(ml_features)

            recorder.record(
                eye,
                pose,
                emo,
                cog,
                cog.get('session_secs', 0)
            )


            rule_state = cog.get("state")
            rule_conf  = cog.get("confidence", 50)

            # =========================
            # HYBRID DECISION
            # =========================
            ml_state = normalize_state(ml_state)
            rule_state = normalize_state(rule_state)

            if ml_state == rule_state:
                final_state = rule_state
            elif ml_state in ["fatigued", "high_cognitive_load"]:
                final_state = ml_state
            elif rule_conf < 60:
                final_state = rule_state
            else:
                final_state = rule_state
            final_state = normalize_state(final_state)

            # =========================
            # STEP 4: FINAL PACKAGING
            # (THIS IS WHAT YOU ASKED)
            # =========================

            r["cognitive"]["state"] = final_state
            r["cognitive"]["state_label"] = final_state

            r["eye"] = eye
            r["pose"] = pose
            r["emotion"] = emo

            r["state"] = final_state
            r["confidence"] = rule_conf

            # IMPORTANT: prevent frontend crash
            if r["cognitive"].get("state_label"):
                r["cognitive"]["state_label"] = STATE_LABELS.get(final_state, "Unknown")
            else:
                r["cognitive"]["state_label"] = "Unknown"

            # =========================
            # EMIT TO FRONTEND
            # =========================
            socketio.emit("detection", _build_payload(r))
            socketio.emit("dataset_stats", recorder.get_stats())

        except Exception as e:
            print("EMIT LOOP ERROR:", e)

        time.sleep(0.08)
        


@app.route('/')
def index():
    return send_from_directory(TEMPLATE_DIR,'index.html')

FEATURE_ORDER = [
    "ear_ratio",
    "blink_rate",
    "blink_irregularity",
    "blink_per_10s",
    "gaze_stability",
    "off_screen_pct",
    "head_steadiness",
    "brow_adjusted",
    "fatigue_index",
    "overload_index",
    "focus_stability",
    "fatigue_trend",
    "emo_angry", "emo_disgust", "emo_fear",
    "emo_happy", "emo_sad", "emo_surprise", "emo_neutral"
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:

        df = pd.DataFrame([data])[FEATURE_ORDER]

        print("TRAIN FEATURES:", scaler.feature_names_in_)
        print("INPUT FEATURES:", df.columns.tolist())

        features = {
            "ear_ratio": ear_ratio_value,
            "blink_irregularity": blink_irregularity_value,
            "blink_per_10s": blink_per_10s_value,
            "gaze_stability": gaze_stability_value,
            "off_screen_pct": off_screen_pct_value,
            "head_steadiness": head_steadiness_value,
            "brow_adjusted": brow_adjusted_value,
            "fatigue_index": fatigue_index_value,
            "overload_index": overload_index_value,
            "focus_stability": focus_stability_value,
            "fatigue_trend": fatigue_trend_value,
            "emo_angry": emo_angry,
            "emo_disgust": emo_disgust,
            "emo_fear": emo_fear,
            "emo_happy": emo_happy,
            "emo_sad": emo_sad,
            "emo_surprise": emo_surprise,
            "emo_neutral": emo_neutral
            }

        feature_columns = joblib.load("feature_columns.pkl")

        df = df.reindex(columns=feature_columns, fill_value=0)
       
        df_scaled = scaler.transform(df)

        pred = ml_model.predict(df_scaled)[0]
        state = normalize_state(ml_encoder.inverse_transform([pred])[0])

        return jsonify({"state": state, "state_label": STATE_LABELS.get(state, state)})

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/session/start',methods=['POST'])
def start_session():
    if session['active']:
        return jsonify({'error':'Already active'}),400
    session['active']=True; session['start']=time.time(); session['log']=[]
    recorder.start_session(); engine.start()
    _log('Session started â€” recording to combined_master_dataset.csv')
    return jsonify({'status':'started'})


@app.route('/api/session/stop',methods=['POST'])
def stop_session():
    if not session['active']:
        return jsonify({'error':'No active session'}),400
    session['active']=False; engine.stop()
    new_rows,total = recorder.stop_session()
    _log(f'Session ended. {new_rows} new rows. Total: {total} rows.')
    return jsonify({'status':'stopped','new_rows':new_rows,'total_rows':total})


@app.route('/api/status')
def api_status():
    return jsonify({
        'active':session['active'],
        'uptime':int(time.time()-session['start']) if session['start'] else 0,
    })


@app.route('/api/dataset/stats')
def dataset_stats():
    return jsonify(recorder.get_stats())


@app.route('/api/dataset/download')
def download_dataset():
    if not os.path.exists(MASTER_CSV_PATH):
        return jsonify({'error':'No dataset yet'}),404
    return send_file(MASTER_CSV_PATH,as_attachment=True,
                     download_name='combined_master_dataset.csv',mimetype='text/csv')


@app.route('/api/label',methods=['POST'])
def manual_label():
    """
    POST /api/label  {"label": "high_cognitive_load", "duration": 5}

    Frontend hotkey mapping:
      1=focused  2=distracted  3=high_cognitive_load  4=fatigued
    """
    data     = request.get_json(force=True)
    label    = normalize_state(data.get('label',''))
    duration = float(data.get('duration',5.0))
    if label not in ALL_STATES:
        return jsonify({'error':f'Must be one of: {ALL_STATES}'}),400
    if recorder.force_label(label,duration):
        _log(f'Manual label: [{label}] for {duration}s')
        return jsonify({'status':'ok','label':label,'duration':duration})
    return jsonify({'error':'Recorder not active'}),400


@app.route('/api/voice/toggle',methods=['POST'])
def toggle_voice():
    voice.enabled = not voice.enabled
    return jsonify({'voice_enabled':voice.enabled})


@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            fb = engine.get_frame()
            if fb:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+fb+b'\r\n'
            time.sleep(0.05)
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def on_connect():
    emit('status',{'active':session['active']})


@socketio.on('request_break')
def on_break():
    _log('Break requested.')
    voice.speak_custom('Time for a break. Well done.')
    emit('break_ack',{'message':'Take a 5-minute break!'})


@socketio.on('manual_label')
def on_manual_label(data):
    label    = normalize_state(data.get('label',''))
    duration = float(data.get('duration',5.0))
    if label in ALL_STATES and recorder.force_label(label,duration):
        _log(f'[Hotkey] {label} for {duration}s')
        emit('label_ack',{'label':label,'duration':duration})
    else:
        emit('label_ack',{'error':'Invalid label or recorder not active'})


if __name__=='__main__':
    t = threading.Thread(target=_emit_loop,daemon=True)
    t.start()
    print('\nâ•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—')
    print('â•‘  CogniSense v5.2  â€”  Research-Supported 4-State Detection â•‘')
    print('â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£')
    print('â•‘  http://localhost:5000                               â•‘')
    print('â•‘  Dataset: combined_master_dataset.csv              â•‘')
    print('â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£')
    print('â•‘  v5.1 FIXES:                                        â•‘')
    print('â•‘  States: focused, distracted, high cognitive load, fatigued â•‘')
    print('â•‘  Old labels are mapped into the new research states          â•‘')
    print('â•‘  Burnout risk remains separate from webcam state detection   â•‘')
    print('â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£')
    print('â•‘  MANUAL LABEL HOTKEYS:                              â•‘')
    print('â•‘  1=focused   2=distracted                           â•‘')
    print('â•‘  3=high cognitive load   4=fatigued                 â•‘')
    print('â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£')
    print('â•‘  pip install pyttsx3        (for voice alerts)       â•‘')
    print('â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n')
    socketio.run(app,host='0.0.0.0',port=5000,debug=False)

