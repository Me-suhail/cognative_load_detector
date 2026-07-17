"""
app.py  --  CogniSense Cognitive Load Detector  v3.0
=====================================================
What's new in v3:
  - Single persistent CSV file (master_dataset.csv) — appends every session
  - Voice alerts via pyttsx3 for bad states (drowsy, overloaded, distracted)
  - Alert cooldown system so voice doesn't spam every second
"""

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
from collections import deque, Counter
from flask          import Flask, Response, jsonify, send_from_directory, send_file, render_template
from flask_cors     import CORS
from flask_socketio import SocketIO, emit


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — PERSISTENT DATASET RECORDER
#  All sessions append to ONE master CSV file forever
# ══════════════════════════════════════════════════════════════════════════════

DATASET_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
MASTER_CSV_PATH = os.path.join(DATASET_DIR, 'master_dataset.csv')
os.makedirs(DATASET_DIR, exist_ok=True)
WINDOW_SECONDS = 30
SAVE_INTERVAL = 10


CSV_COLUMNS = [
    'session_id', 'window_start', 'window_end',
    'mean_ear', 'std_ear', 'min_ear', 'max_ear',
    'blink_count', 'blink_rate', 'blink_irregularity',
    'gaze_direction', 'gaze_stability', 'off_screen_pct',
    'head_pitch', 'head_yaw', 'head_roll', 'head_label', 'head_steadiness',
    'focus_score', 'avg_score', 'confidence',
    'ear_baseline','blink_threshold','drowsy_threshold',
    'sig_ear', 'sig_blink_rate', 'sig_gaze', 'sig_head', 'sig_blink_pattern',
    'dominant_state',
]


class DatasetRecorder:

    def __init__(self):
        self._session_id = None
        self._file       = None
        self._writer     = None
        self._window = deque()
        self._last_save  = 0
        self._row_count  = 0      # rows saved in THIS session
        self._total_rows = 0      # total rows in the master file
        self._lock       = threading.Lock()
        self.recording   = False
        self._count_existing_rows()

    def _count_existing_rows(self):
        
        if not os.path.exists(MASTER_CSV_PATH):
            self._total_rows = 0
            return
        try:
                with open(MASTER_CSV_PATH, 'r', encoding='utf-8') as f:
                    self._total_rows = max(0, sum(1 for _ in f) - 1)  # minus header
        except Exception:
                self._total_rows = 0
        print(f'[Dataset] Master file has {self._total_rows} existing rows.')

    def start_session(self):
        self.recording=True
        self._session_id = str(uuid.uuid4())[:8]
        self._window.clear()
        self._row_count  = 0
        self._last_save = time.time()

        file_exists = os.path.exists(MASTER_CSV_PATH)
        self._file  = open(MASTER_CSV_PATH, 'a', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)

        if not file_exists or os.path.getsize(MASTER_CSV_PATH) == 0:
            self._writer.writeheader()
            print("[Dataset] Created new dataset.")
           
        else:

            print(
                f"[Dataset] Appending to dataset ({self._total_rows} rows)"
            )

        self._file.flush()
        return MASTER_CSV_PATH

    def record(self, eye, pose, cog, session_secs):
        if not self.recording :
            return
        now = time.time()
        timestamp = time.time()
        frame = {

            "timestamp": timestamp,

            "session_secs": session_secs,

            "ear": eye.get("smooth_ear", 0),

            "blink_rate": eye.get("blink_rate", 0),

            "blink_count": eye.get("blink_count", 0),

            "blink_irregularity":
                eye.get("blink_irregularity", 0),

            "gaze_direction":
                eye.get("gaze", {}).get(
                    "direction",
                    "center"
                ),

            "gaze_stability":
                eye.get("gaze_stability", 0),

            "off_screen_pct":
                eye.get("off_screen_pct", 0),

            "head_pitch":
                pose.get("pitch", 0),

            "head_yaw":
                pose.get("yaw", 0),

            "head_roll":
                pose.get("roll", 0),

            "head_label":
                pose.get("label", "upright"),

            "head_steadiness":
                pose.get("steadiness", 0),

            "focus_score":
                cog.get("score", 0),

            "avg_score":
                cog.get("avg_score", 0),

            "confidence":
                cog.get("confidence", 0),

            "signals":
                cog.get("signal_scores", {}),

            "state":
                cog.get("state", "focused"),
            "ear_baseline":
                eye.get("ear_baseline", 0),

            "blink_threshold":
                eye.get("blink_threshold", 0),

            "drowsy_threshold":
                eye.get("drowsy_threshold", 0)


        }

        self._window.append(frame)

        while (
            self._window and
            now - self._window[0]["timestamp"] > WINDOW_SECONDS
        ):

            self._window.popleft()

        if timestamp - self._last_save >= SAVE_INTERVAL:
            self._save_window()
            self._last_save = timestamp

    def _extract_window_features(self):

        if len(self._window) == 0:
            return None

        ears = [x["ear"] for x in self._window]

        blink_rates = [f["blink_rate"] for f in self._window]

        blink_irregularity = [
            f["blink_irregularity"]
            for f in self._window
        ]

        gaze_stability = [
            f["gaze_stability"]
            for f in self._window
        ]

        off_screen = [
            f["off_screen_pct"]
            for f in self._window
        ]

        head_pitch = [
            f["head_pitch"]
            for f in self._window
        ]

        head_yaw = [
            f["head_yaw"]
            for f in self._window
        ]

        head_roll = [
            f["head_roll"]
            for f in self._window
        ]

        head_steadiness = [
            f["head_steadiness"]
            for f in self._window
        ]

        focus_scores = [
            f["focus_score"]
            for f in self._window
        ]

        avg_scores = [
            f["avg_score"]
            for f in self._window
        ]

        confidence = [
            f["confidence"]
            for f in self._window
        ]
        ear_baselines = [
            f["ear_baseline"]
            for f in self._window
        ]

        blink_thresholds = [
            f["blink_threshold"]
            for f in self._window
        ]

        drowsy_thresholds = [
            f["drowsy_threshold"]
            for f in self._window
        ]

        signal_ear = []
        signal_blink = []
        signal_gaze = []
        signal_head = []
        signal_pattern = []

        states = []
        gaze_direction = []
        head_labels = []

        for frame in self._window:

            states.append(frame["state"])

            gaze_direction.append(
                frame["gaze_direction"]
            )

            head_labels.append(
                frame["head_label"]
            )

            signals = frame["signals"]

            signal_ear.append(
                signals.get("ear", 0)
            )

            signal_blink.append(
                signals.get("blink_rate", 0)
            )

            signal_gaze.append(
                signals.get("gaze_stability", 0)
            )

            signal_head.append(
                signals.get("head_steadiness", 0)
            )

            signal_pattern.append(
                signals.get("blink_pattern", 0)
            )

        row = {

            "session_id":
                self._session_id,

            "window_start":
                datetime.fromtimestamp(
                    self._window[0]["timestamp"]
                ).strftime("%Y-%m-%d %H:%M:%S"),

            "window_end":
                datetime.fromtimestamp(
                    self._window[-1]["timestamp"]
                ).strftime("%Y-%m-%d %H:%M:%S"),

            "mean_ear":
                round(np.mean(ears),4),

            "std_ear":
                round(np.std(ears),4),

            "min_ear":
                round(np.min(ears),4),
            "max_ear":
                round(np.max(ears),4),

            "blink_count":
                int(self._window[-1]["blink_count"]),

            "blink_rate":
                round(np.mean(blink_rates),2),

            "blink_irregularity":
                round(np.mean(blink_irregularity),3),

            "gaze_direction":
                Counter(gaze_direction).most_common(1)[0][0],

            "gaze_stability":
                round(np.mean(gaze_stability),2),

            "off_screen_pct":
                round(np.mean(off_screen),2),

            "head_pitch":
                round(np.mean(head_pitch),2),

            "head_yaw":
                round(np.mean(head_yaw),2),

            "head_roll":
                round(np.mean(head_roll),2),

            "head_label":
                Counter(head_labels).most_common(1)[0][0],

            "head_steadiness":
                round(np.mean(head_steadiness),2),

            "focus_score":
                round(np.mean(focus_scores),2),

            "avg_score":
                round(np.mean(avg_scores),2),

            "confidence":
                round(np.mean(confidence),1),

            "ear_baseline":
                round(np.mean(ear_baselines),4),

            "blink_threshold":
                round(np.mean(blink_thresholds),4),

            "drowsy_threshold":
                round(np.mean(drowsy_thresholds),4),

            "sig_ear":
                round(np.mean(signal_ear),2),

            "sig_blink_rate":
                round(np.mean(signal_blink),2),

            "sig_gaze":
                round(np.mean(signal_gaze),2),

            "sig_head":
                round(np.mean(signal_head),2),

            "sig_blink_pattern":
                round(np.mean(signal_pattern),2),

            "dominant_state":
                Counter(states).most_common(1)[0][0]

        }
        return row

        
    def _save_window(self):

        row = self._extract_window_features()

        if row is None:
            return

        with self._lock:

            self._writer.writerow(row)

            self._file.flush()

            self._row_count += 1

            self._total_rows += 1

    def stop_session(self):
        self.recording = False
        if self._file is not None:
            self._file.close()
            self._file = None
        print(
            f"[Dataset] Session finished "
            f"({self._row_count} new rows)"
        )
        return self._row_count, self._total_rows

    def get_stats(self):
        return {
            'total_rows':    self._total_rows,
            'session_rows':  self._row_count,
            'window_seconds':WINDOW_SECONDS,
            'save_interval':SAVE_INTERVAL,
            'file':          'dataset/master_dataset.csv',
            'recording':     self.recording,
        }
    def reset(self):

        self.recording = False

        self._window.clear()

        self._session_id = None

        self._last_save = 0

        self._row_count = 0

        if self._file is not None:

            self._file.close()

            self._file = None


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — VOICE ALERT ENGINE
#  Speaks to the student when a bad state is detected
#  Uses pyttsx3 (works offline, no API key needed)
# ══════════════════════════════════════════════════════════════════════════════

VOICE_MESSAGES = {
    'fatigued':   [
        "You look tired. Please take a 5 minute break.",
        "Fatigue detected. Rest your eyes for a few minutes.",
        "Your eyes are closing. Take a short break now.",
    ],
    'high_cognitive_load': [
        "High cognitive load detected. Take a moment to process the information.",
        "You may be overloaded. Try breaking the topic into smaller parts.",
        "Too much information at once. Pause briefly and review what you have learned."
    ],
    'distracted': [
        "You seem distracted. Bring your focus back to the screen.",
        "Attention drift detected. Try to refocus on your study material.",
        "Stay focused. You are almost there.",
    ],
    'confused': [
        "Confusion detected. Try re-reading the last section.",
        "You seem confused. Draw a diagram or simplify the concept.",
    ],
    'micro_sleep': [
        "Wake up! You are falling asleep.",
        "Micro sleep detected. Please take a break immediately.",
    ],
}

# How many seconds to wait before repeating the same alert type
VOICE_COOLDOWN = {
    'fatigued':   90,
    'high_cognitive_load': 60,
    'distracted': 60,
    'confused':   75,
    'micro_sleep':30,
}


class VoiceAlertEngine:
    """
    Runs speech in a background thread so it never blocks the main loop.
    Uses pyttsx3 — works fully offline on Windows/Mac/Linux.
    """

    def __init__(self):
        self._engine      = None
        self._ok          = None
        self._queue       = queue.Queue()
        self._last_spoken = {}        # alert_type → timestamp
        self._msg_index   = {}        # cycles through message variations
        self._thread      = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self.enabled      = True

    def _init_engine(self):
        if self._ok is not None:
            return self._ok
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate',   165)   # speaking speed
            self._engine.setProperty('volume', 0.92)
            self._ok = True
            print('[Voice] pyttsx3 ready.')
        except Exception as e:
            print(f'[Voice] pyttsx3 not available: {e}. Install with: pip install pyttsx3')
            self._ok = False
        return self._ok

    def _worker(self):
        """Background thread that speaks queued messages one by one."""
        while True:
            msg = self._queue.get()
            if self._init_engine() and self._engine:
                try:
                    self._engine.say(msg)
                    self._engine.runAndWait()
                except Exception as e:
                    print(f'[Voice] Error speaking: {e}')

    def speak(self, alert_type: str):
        """Call this with an alert type. Respects cooldown, cycles messages."""
        if not self.enabled:
            return False
        now  = time.time()
        last = self._last_spoken.get(alert_type, 0)
        cool = VOICE_COOLDOWN.get(alert_type, 60)
        if now - last < cool:
            return False

        msgs = VOICE_MESSAGES.get(alert_type, [])
        if not msgs:
            return False

        # Cycle through message variations so it doesn't repeat the same line
        idx = self._msg_index.get(alert_type, 0) % len(msgs)
        msg = msgs[idx]
        self._msg_index[alert_type] = idx + 1
        self._last_spoken[alert_type] = now

        self._queue.put(msg)
        print(f'[Voice] Speaking ({alert_type}): {msg}')
        return True

    def speak_custom(self, text: str):
        """Speak any custom text immediately."""
        if self.enabled:
            self._queue.put(text)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — EYE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

LEFT_EYE_INDICES   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES  = [33,  160, 158, 133, 153, 144]
LEFT_IRIS_INDICES  = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]



class EyeTracker:

    def __init__(self):
        self.blink_count      = 0
        self.frame_counter    = 0
        self.closed_since     = None
        
        self.drowsy_counter   = 0
        self.is_drowsy        = False
        
        self.blink_times      = deque(maxlen=120)
        self.ear_history      = deque(maxlen=120)
        self.gaze_history     = deque(maxlen=60)
        self.recent_blink_gaps =deque(maxlen=20)
        
        self._calibration_ears = deque(maxlen=60)
        self._gaze_v_cal = deque(maxlen=60)
        self._gaze_h_cal=deque(maxlen=60)

        self._calibrated       = False
        self._last_blink_time=None
        
        self.ear_open_baseline = 0.30
        self.ear_blink_thresh  = 0.22
        self.ear_drowsy_thresh = 0.20

        self.gaze_h_baseline = 0.0
        self.gaze_v_baseline = 0.0

        self._drowsy_frames_needed = 12
        

    def _calibrate(self, ear, h_ratio, v_ratio):
        if self._calibrated: return
        if ear > 0.20:
            self._calibration_ears.append(ear)
            self._gaze_h_cal.append(h_ratio)
            self._gaze_v_cal.append(v_ratio)
            
        if len(self._calibration_ears) >= 60:
            baseline = np.percentile(list(self._calibration_ears), 70)
            self.ear_open_baseline = round(baseline, 3)
            self.ear_blink_thresh  = round(baseline * 0.72, 3)
            self.ear_drowsy_thresh = round(baseline * 0.62, 3)
            self._calibrated = True

    @staticmethod
    def _ear(landmarks, indices, w, h):
        pts = [np.array([landmarks[i].x*w, landmarks[i].y*h]) for i in indices]
        A = np.linalg.norm(pts[1]-pts[5])
        B = np.linalg.norm(pts[2]-pts[4])
        C = np.linalg.norm(pts[0]-pts[3])
        return round(float((A+B)/(2.0*C)) if C else 0.0, 4)

    @staticmethod
    def _gaze(landmarks, eye_idx, iris_idx, w, h):
        lc, rc  = landmarks[eye_idx[0]], landmarks[eye_idx[3]]
        eye_w   = abs(rc.x-lc.x)*w
        if eye_w < 1: return 0.0, 0.0
        iris_cx = np.mean([landmarks[i].x for i in iris_idx])*w
        iris_cy = np.mean([landmarks[i].y for i in iris_idx])*h
        eye_cx  = ((lc.x+rc.x)/2)*w
        eye_cy  = ((lc.y+rc.y)/2)*h
        h_r = (iris_cx-eye_cx)/(eye_w/2)
        v_r = (iris_cy-eye_cy)/(eye_w/2)
    
        return round(h_r,3), round(v_r,3)

    def _classify_gaze(self, h_r, v_r):
        h_adj = h_r - self.gaze_h_baseline
        v_adj = v_r - self.gaze_v_baseline

        if h_adj < -0.16:
            return "left"
        elif h_adj > 0.16:
            return "right"
        elif v_adj < -0.18:
            return "up"
        elif v_adj > 0.18:
            return "down"
        return "center"

    def update(self, landmarks, w, h):
        left_ear  = self._ear(landmarks, LEFT_EYE_INDICES,  w, h)
        right_ear = self._ear(landmarks, RIGHT_EYE_INDICES, w, h)
        avg_ear   = round((left_ear+right_ear)/2, 4)

        h_r, v_r = self._gaze(
            landmarks,
            LEFT_EYE_INDICES,
            LEFT_IRIS_INDICES,
            w,
            h
        )
        
        self._calibrate(avg_ear,h_r,v_r)
        direction = self._classify_gaze(h_r,v_r)
        self.ear_history.append(avg_ear)
        smooth_ear = round(float(np.mean(list(self.ear_history)[-5:])), 4)

        blink_detected = False
        now = time.time()
        if smooth_ear < self.ear_blink_thresh:
            self.frame_counter += 1
            if self.closed_since is None: self.closed_since = now
        else:
            if self.frame_counter >= 2:
                self.blink_count += 1; blink_detected = True
                self.blink_times.append(now)

                if self._last_blink_time:
                    self.recent_blink_gaps.append(
                        now - self._last_blink_time
                    )
                self._last_blink_time = now

            self.frame_counter = 0
            self.closed_since = None

        micro_sleep = bool(self.closed_since and (now-self.closed_since) > 1.5)

        if smooth_ear < self.ear_drowsy_thresh:
            self.drowsy_counter += 1
            if self.drowsy_counter >= self._drowsy_frames_needed: self.is_drowsy = True
        else:
            self.drowsy_counter = max(0, self.drowsy_counter-1)
            if self.drowsy_counter == 0: self.is_drowsy = False

        blink_rate     = len([t for t in self.blink_times if now-t <= 60])
        self.gaze_history.append(direction)
        recent         = list(self.gaze_history)[-30:]
        gaze_stability = round(sum(1 for g in recent if g=='center')/max(len(recent),1)*100,1)
        ten_sec        = list(self.gaze_history)[-60:]
        off_screen_pct = round(sum(1 for g in ten_sec if g in ('left','right'))/max(len(ten_sec),1)*100,1)
        blink_irregularity = 0.0
        if len(self.recent_blink_gaps) >= 5:
            gaps = list(self.recent_blink_gaps)

            blink_irregularity = round(
                float(np.std(gaps) / max(np.mean(gaps), 0.1)),
                3
            )
        return {
            'left_ear':left_ear,'right_ear':right_ear,'avg_ear':avg_ear,
            'smooth_ear':smooth_ear,
            'blink_count':self.blink_count,'blink_detected':blink_detected,
            'blink_rate':blink_rate,'is_drowsy':self.is_drowsy,
            'micro_sleep':micro_sleep,'gaze':{'direction':direction,'h':h_r,'v':v_r},
            'gaze_stability':gaze_stability,'off_screen_pct':off_screen_pct,
            'calibrated':self._calibrated,
            'ear_baseline':self.ear_open_baseline,
            'blink_threshold': self.ear_blink_thresh,
            'drowsy_threshold': self.ear_drowsy_thresh,'blink_irregularity':blink_irregularity,
        }

    def reset(self):
        self.blink_count=self.frame_counter=self.drowsy_counter=0
        self.is_drowsy=False; self.closed_since=None; self._calibrated=False; self._last_blink_time=None
        self._calibration_ears.clear(); self._gaze_h_cal.clear(); self._gaze_v_cal.clear(); self.blink_times.clear()
        self.ear_history.clear(); self.gaze_history.clear(); self.recent_blink_gaps.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — HEAD POSE
# ══════════════════════════════════════════════════════════════════════════════

MODEL_3D = np.array([
    (0.0,0.0,0.0),(0.0,-63.6,-12.5),
    (-43.3,32.7,-26.0),(43.3,32.7,-26.0),
    (-28.9,-28.9,-24.1),(28.9,-28.9,-24.1),
], dtype=np.float64)
HEAD_IDX = [1,152,33,263,61,291]


class HeadPoseEstimator:
    def __init__(self):
        self._cam=None; self._dist=np.zeros((4,1),dtype=np.float64)
        self._last_w = None; self._last_h = None
        self._pitch_hist = deque(maxlen=10); self._yaw_hist = deque(maxlen=10)
        self._roll_hist = deque(maxlen=10); self._dist_hist = deque(maxlen=15)
        self._baseline_distance = None

    def _update_camera_matrix(self, w, h):
        if self._cam is None or self._last_w != w or self._last_h != h:
            self._cam = np.array([
                [w, 0, w / 2],
                [0, w, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)

            self._last_w = w
            self._last_h = h

    def estimate(self, landmarks, w, h):
        self._update_camera_matrix(w, h)

        try:
            image_points = np.array(
                [(landmarks[i].x * w, landmarks[i].y * h) for i in HEAD_IDX],
                dtype=np.float64
            )

            ok, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_3D,
                image_points,
                self._cam,
                self._dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not ok:
                return self._default()

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            projection = np.hstack((rotation_matrix, translation_vector))

            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(projection)

            pitch = float(euler[0])
            yaw = float(euler[1])
            roll = float(euler[2])

            distance = float(translation_vector[2])

            self._pitch_hist.append(pitch)
            self._yaw_hist.append(yaw)
            self._roll_hist.append(roll)
            self._dist_hist.append(distance)

            pitch = float(np.mean(self._pitch_hist))
            yaw = float(np.mean(self._yaw_hist))
            roll = float(np.mean(self._roll_hist))
            distance = float(np.mean(self._dist_hist))

            if self._baseline_distance is None and len(self._dist_hist) >= 15:
                self._baseline_distance = distance

            pose_reliable = (
                abs(pitch) < 75 and
                abs(yaw) < 75 and
                abs(roll) < 75 and
                distance > 0
            )

            forward_lean = False
            if self._baseline_distance is not None:
                forward_lean = distance < self._baseline_distance * 0.88

            if abs(yaw) > 25:
                label = "turned_away"
            elif pitch > 22:
                label = "head_drop"
            elif forward_lean:
                label = "forward_lean"
            elif abs(roll) > 20:
                label = "tilted"
            else:
                label = "upright"

            movement_penalty = abs(pitch) + abs(yaw) + abs(roll)
            steadiness = max(0.0, 100 - movement_penalty * 1.05)

            return {
                "pitch": round(pitch, 1),
                "yaw": round(yaw, 1),
                "roll": round(roll, 1),

                "label": label,
                "steadiness": round(steadiness, 1),
                "pose_reliable": pose_reliable,

                "distance_z": round(distance, 2),
                "forward_lean_detected": forward_lean
            }

        except Exception:
            return self._default()

    @staticmethod
    def _default():
        return {'pitch':0.0,'yaw':0.0,'roll':0.0,'label':'upright','steadiness':100.0,'pose_reliable':False,'distance_z':0.0,'forward_lean_detection':False}

    def reset(self):
        self._pitch_hist.clear()
        self._yaw_hist.clear()
        self._roll_hist.clear()
        self._dist_hist.clear()
        self._baseline_distance = None

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — COGNITIVE LOAD CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS={'ear':0.30,'blink_rate':0.18,'gaze_stability':0.18,
         'head_steadiness':0.18,'blink_pattern': 0.16}

assert abs(sum(WEIGHTS.values()) - 1.0) < 0.01


COLORS = {
    'focused': '#00d4a0',
    'high_cognitive_load': '#e05555',
    'fatigued': '#f0a04b',
    'distracted': '#9b7cf4'
}

STATE_LABELS = {
    'focused': 'Focused',
    'distracted': 'Distracted',
    'high_cognitive_load': 'High Cognitive Load',
    'fatigued': 'Fatigued'
}

class CognitiveLoadClassifier:
    def __init__(self):
        self.score_history=deque(maxlen=120); self.state_history=deque(maxlen=30)
        self.session_start=time.time(); self._last_alert_t={}

    def compute(self,eye,pose):
        ear=eye.get('smooth_ear',0.30); base=eye.get('ear_baseline',0.30)
        blink = eye.get('blink_rate', 15); birr = eye.get('blink_irregularity', 0.0)
        gaze_score = eye.get('gaze_stability', 80); off_screen = eye.get('off_screen_pct', 0)
        pose_reliable = pose.get('pose_reliable', True); pose_label = pose.get('label', 'upright')
        head_score = pose.get('steadiness', 80) if pose_reliable else 65
        is_drowsy = eye.get('is_drowsy', False)
        micro_sleep = eye.get('micro_sleep', False)
        
        if eye.get('calibrated'):
            ear_score = min(100.0, (ear / max(base, 0.01)) * 100)
        else:
            ear_score = max(0.0, min(100.0, (ear - 0.18) / (0.40 - 0.18) * 100))

        if 12 <= blink <= 20:
            blink_score = 100.0
        elif blink < 12:
            blink_score = max(0.0, 100 - (12 - blink) * 8)
        else:
            blink_score = max(0.0, 100 - (blink - 20) * 5)

        blink_pattern_score = max(0.0, 100 - birr * 65)

        
        signals={'ear':round(ear_score,1),'blink_rate':round(blink_score,1),
                 'gaze_stability':round(gaze_score,1),'head_steadiness':round(head_score,1),
                 'blink_pattern':round(blink_pattern_score,1)}
        total = sum(signals[k] * WEIGHTS[k] for k in WEIGHTS)
        total = round(max(0, min(100, total)), 1)

        if micro_sleep:
            state = 'fatigued'
            total = min(total, 15)

        elif is_drowsy:
            state = 'fatigued'
            total = min(total, 35)

        elif off_screen > 65 or pose_label == 'turned_away':
            state = 'distracted'
            total = min(total, 45)

        else:
            session_time = time.time() - self.session_start
            long_session = session_time > 1800

            if long_session and total < 55:
                state = 'fatigued'
            elif total >= 75:
                state = 'focused'
            elif total >= 35:
                state = 'high_cognitive_load'
            else:
                state = 'fatigued'

        self.score_history.append(total); self.state_history.append(state)
        hist=list(self.score_history)
        avg_score = round(float(np.mean(hist)), 1) if hist else total
        recent_states = list(self.state_history)[-10:]
        dominant_state = (
            max(set(recent_states), key=recent_states.count)
            if recent_states else state
        )

        confidence = round(
            recent_states.count(dominant_state) /
            max(len(recent_states), 1) * 100,
            0
        )

        alerts = self._build_alerts(
            eye, pose, state, blink, off_screen, micro_sleep
        )
        advice = self._advice(state, eye, pose)

        return {'score':total,'state':state,'state_label':STATE_LABELS[state],
                'state_color':COLORS.get(state,'#888'),'avg_score':avg_score,
                'signal_scores':signals,'is_drowsy':is_drowsy,'micro_sleep':micro_sleep,
                'is_distracted':state == 'distracted','history':list(self.score_history)[-40:],
                'session_secs':int(time.time()-self.session_start),
                'confidence':int(confidence),'alerts':alerts, 'advice':advice}

    def _build_alerts(self,eye,pose,state,blink,off_screen,micro_sleep):
        alerts=[]; now=time.time()
        def push(key,msg,sev,cd=30):
            if now-self._last_alert_t.get(key,0)>cd:
                alerts.append({'key':key,'msg':msg,'severity':sev})
                self._last_alert_t[key]=now
        if micro_sleep:
            push('microsleep','Micro-sleep detected!','alert',10)
        if eye.get('is_drowsy'):
            push('drowsy','Drowsiness detected','alert',20)
        if blink<6:
            push('lowblink',f'Low blink rate ({blink}/min)','warn',40)
        elif blink>28:
            push('highblink',f'High blink rate ({blink}/min)','warn',40)
        if off_screen>65:
            push('gaze','Gaze off-screen frequently','warn',30)
        if pose.get('label')=='head_drop':
            push('headdrop','Head dropping forward','warn',25)
        if state=='focused' and not alerts:
            push('good','Focus maintained','good',60)
        return alerts

    @staticmethod
    def _advice(state,eye,pose):
        advice_map = {
            'focused': {
                'title': 'Deep Focus',
                'message': 'Excellent learning state.',
                'action': None,
                'severity': 'good'
            },
            'high_cognitive_load': {
                'title': 'High Cognitive Load',
                'message': 'Too much information at once.',
                'action': 'Pause for 2 minutes',
                'severity': 'alert'
            },
            'fatigued': {
                'title': 'Fatigue',
                'message': 'Take a short break.',
                'action': 'Rest now',
                'severity': 'alert'
            },
            'distracted': {
                'title': 'Distracted',
                'message': 'Attention drift detected.',
                'action': 'Refocus',
                'severity': 'warn'
            }
        }
        return advice_map.get(state, advice_map['focused'])
    def reset(self):
        self.score_history.clear(); self.state_history.clear()
        self.session_start=time.time(); self._last_alert_t={}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — DETECTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

mp_face_mesh=mp.solutions.face_mesh

class DetectionEngine:
    def __init__(self,camera_index=0,target_fps=20):
        self.latest_result = {}
        self.camera_index=camera_index; self.frame_delay=1.0/target_fps
        self._running=False
        self._thread=None; self._lock=threading.Lock(); self._frame_bytes=None
        self.eye_tracker=EyeTracker(); self.head_pose=HeadPoseEstimator()
        self.classifier=CognitiveLoadClassifier()

    def start(self):
        if self._running: return
        self._running=True; self.eye_tracker.reset(); self.classifier.reset()
        self._thread=threading.Thread(target=self._loop,daemon=True); self._thread.start()

    def stop(self):
        self._running=False
        if self._thread: self._thread.join(timeout=3)

    def get_latest(self):
        with self._lock:
            return self.latest_result.copy()

    def get_frame(self):
        with self._lock: return self._frame_bytes

    def _loop(self):
        cap=cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f'[Engine] Cannot open camera'); self._running=False; return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        fail_count = 0
        CFG=dict(max_num_faces=1,refine_landmarks=True,
                 min_detection_confidence=0.65,min_tracking_confidence=0.65)
        with mp_face_mesh.FaceMesh(**CFG) as fm:
            while self._running:
                t0=time.time(); ret,frame=cap.read()
                if not ret:
                    fail_count += 1

                    if fail_count > 50:
                        print("[Engine] Camera stream lost")
                        break

                    time.sleep(0.05)
                    continue

                fail_count = 0
                frame=cv2.flip(frame,1); h,w=frame.shape[:2]
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                rgb.flags.writeable=False; mesh=fm.process(rgb); rgb.flags.writeable=True
                result={'timestamp':time.time(),'face_detected':False,
                        'eye':{},'pose':{},'cognitive':{}}
                if mesh.multi_face_landmarks:
                    lm=mesh.multi_face_landmarks[0].landmark
                    eye=self.eye_tracker.update(lm,w,h)
                    pos=self.head_pose.estimate(lm,w,h)
                    cog=self.classifier.compute(eye,pos)
                    result.update({'face_detected':True,'eye':eye,'pose':pos,'cognitive':cog})
                    frame=self._annotate(frame,eye,pos,cog,w,h)
                else:
                    cv2.putText(frame,'No face detected',(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.65,(80,120,255),2)
                ok,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,82])
                if ok:
                    with self._lock:
                        self._frame_bytes = buf.tobytes()
                        self.latest_result = result

                elapsed = time.time() - t0
                sleep_time = self.frame_delay - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)
        cap.release()
        self._running = False

    @staticmethod
    def _annotate(frame,eye,pose,cog,w,h):
        C={'focused':(29,158,117),
           'high_cognitive_load':(85,85,224),'fatigued':(75,160,240),'distracted':(147,124,183)}
        color=C.get(cog.get('state',''), (100,100,100))
        cv2.rectangle(frame,(10,10),(240,56),color,-1)
        cv2.putText(frame,f"{cog.get('state_label','?')}  {cog.get('score',0):.0f}%  ({cog.get('confidence',0):.0f}%)",
                    (15,38),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),2)
        cv2.putText(frame,f"EAR:{eye.get('smooth_ear',0):.3f}  Blinks:{eye.get('blink_count',0)}  Rate:{eye.get('blink_rate',0)}/min",
                    (10,72),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
        cv2.putText(frame,f"Pose:{pose.get('label','?')}  Gaze:{eye.get('gaze',{}).get('direction','?')}",
                    (10,86),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
        if not eye.get('calibrated'):
            cv2.putText(frame,'Calibrating...',(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,200,50),1)
        if eye.get('micro_sleep'):
            cv2.rectangle(frame,(0,0),(w,h),(0,0,220),5)
            cv2.putText(frame,'MICRO-SLEEP',(w//2-90,h//2),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
        elif eye.get('is_drowsy'):
            cv2.rectangle(frame,(0,0),(w,h),(0,100,200),3)
        return frame


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR,'templates')
STATIC_DIR   = os.path.join(BASE_DIR,'static')

app      = Flask(__name__,template_folder=TEMPLATE_DIR,static_folder=STATIC_DIR)
CORS(app)
socketio = SocketIO(app,cors_allowed_origins='*',async_mode='threading')
engine   = DetectionEngine(camera_index=0,target_fps=20)
recorder = DatasetRecorder()
voice    = VoiceAlertEngine()
class SessionManager:
    def __init__(self):
        self.reset()
    
    def start(self):
        self.active = True
        self.start_time = time.time()
        self.logs.clear()

    def stop(self):
        self.active = False

    def uptime(self):
        if self.start_time is None:
            return 0
        return int(time.time() - self.start_time)

    def add_log(self, message):
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "message": message
        }

        self.logs.insert(0, entry)
        self.logs = self.logs[:50]

        return entry

    def reset(self):
        self.active = False
        self.start_time = None
        self.logs=[]


session = SessionManager()


def _log(msg):
    entry=session.add_log(msg)
    socketio.emit('log',entry)


def _emit_loop():
    while True:
        if session.active:
            r=engine.get_latest()
            if r:
                cog=r.get('cognitive',{}); eye=r.get('eye',{})
                pose=r.get('pose',{})

                # Save to dataset
                if r.get('face_detected'):
                    try:
                        recorder.record(eye, pose, cog, cog.get('session_secs', 0))
                    except Exception as e:
                        print(f"[Recorder Error] {e}")

                # Voice alerts for bad states
                state=cog.get('state','')
                if eye.get('micro_sleep'):   voice.speak('micro_sleep')
                elif eye.get('is_drowsy'):   voice.speak('fatigued')
                elif state=='high_cognitive_load':    voice.speak('high_cognitive_load')
                elif state=='distracted':    voice.speak('distracted')
                elif state=='fatigued':      voice.speak('fatigued')

                stats=recorder.get_stats()
                socketio.emit('dataset_stats',stats)
                socketio.emit('detection',{
                    'face_detected':r.get('face_detected',False),
                    'score':        cog.get('score',0),
                    'state':        cog.get('state',''),
                    'state_label':  cog.get('state_label',''),
                    'state_color':  cog.get('state_color','#888'),
                    'avg_score':    cog.get('avg_score',0),
                    'history':      cog.get('history',[]),
                    'advice':       cog.get('advice',{}),
                    'signal_scores':cog.get('signal_scores',{}),
                    'confidence':   cog.get('confidence',0),
                    'alerts':       cog.get('alerts',[]),
                    'eye':{'left_ear':eye.get('left_ear',0),'right_ear':eye.get('right_ear',0),
                           'smooth_ear':eye.get('smooth_ear',0),'openness_pct':eye.get('openness_pct',0),
                           'blink_count':eye.get('blink_count',0),'blink_rate':eye.get('blink_rate',0),
                           'gaze':eye.get('gaze',{}),'gaze_stability':eye.get('gaze_stability',0),
                           'off_screen_pct':eye.get('off_screen_pct',0),
                           'is_drowsy':eye.get('is_drowsy',False),'micro_sleep':eye.get('micro_sleep',False),
                           'calibrated':eye.get('calibrated',False)}
                })
        time.sleep(0.08)


@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/session/start',methods=['POST'])
def start_session():
    if session.active: return jsonify({'error':'Already active'}),400
    session.start()
    recorder.start_session(); engine.start()
    _log('Session started. Dataset recording to master_dataset.csv')
    return jsonify({'status':'started'})

@app.route('/api/session/stop',methods=['POST'])
def stop_session():
    if not session.active: return jsonify({'error':'No active session'}),400
    session.stop(); engine.stop()
    new_rows,total=recorder.stop_session()
    _log(f'Session ended. {new_rows} new rows added. Total dataset: {total} rows.')
    return jsonify({'status':'stopped','new_rows':new_rows,'total_rows':total})

@app.route('/api/status')
def status():
    return jsonify({'active':session.active,
                    'uptime':session.uptime()})

@app.route('/api/dataset/stats')
def dataset_stats(): return jsonify(recorder.get_stats())

@app.route('/api/dataset/download')
def download_dataset():
    if not os.path.exists(MASTER_CSV_PATH):
        return jsonify({'error':'No dataset yet'}),404
    return send_file(MASTER_CSV_PATH,as_attachment=True,
                     download_name='master_dataset.csv',mimetype='text/csv')

@app.route('/api/voice/toggle',methods=['POST'])
def toggle_voice():
    voice.enabled = not voice.enabled
    return jsonify({'voice_enabled':voice.enabled})

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            fb=engine.get_frame()
            if fb: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+fb+b'\r\n'
            time.sleep(0.05)
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def on_connect(): emit('status',{'active':session.active})

@socketio.on('request_break')
def on_break():
    _log('Break requested.'); voice.speak_custom('Time for a break. Well done.')
    emit('break_ack',{'message':'Take a 5-minute break!'})


if __name__=='__main__':
    t=threading.Thread(target=_emit_loop,daemon=True); t.start()
    print('\n=========================================')
    print('  CogniSense v3.0')
    print('  http://localhost:5000')
    print('  Dataset: ./dataset/master_dataset.csv')
    print('=========================================')
    print('\n  Install voice alerts: pip install pyttsx3\n')
    socketio.run(app,host='0.0.0.0',port=5000,debug=False)
