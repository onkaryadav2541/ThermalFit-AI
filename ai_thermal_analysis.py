import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'

# --- PHYSICS CALIBRATION ---
T_MIN = 22.0
T_MAX = 35.0
LEAK_THRESHOLD_SCORE = 45

# --- INITIAL CALIBRATION ---
global_offset_x = -40
global_offset_y = 0
scale_factor = 1.0

# --- STABILIZER SETTINGS (THE FIX) ---
STABILITY_STRENGTH = 0.02   # Smooths out medium movements
TELEPORT_DISTANCE = 60      # Allow quick jumps if the face moves far
DEADBAND_RADIUS = 4         # NEW: Ignore ANY movement smaller than 4 pixels (Kills the micro-shake)
SCORE_HISTORY_LEN = 25      # Smooth out the visual bar jumping

# --- LANDMARK DEFINITIONS ---
LANDMARK_IDS = {
    "Nose Bridge": 168,
    "Left Cheek": 118,
    "Right Cheek": 347,
    "Left Chin": 58,
    "Right Chin": 288,
    "Center Chin": 152
}


class PointStabilizer:
    """Uses Exponential Moving Average (EMA) + Deadband to kill jitter."""

    def __init__(self, alpha=STABILITY_STRENGTH):
        self.alpha = alpha
        self.prev_x = None
        self.prev_y = None

    def update(self, new_x, new_y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = new_x, new_y
            return int(new_x), int(new_y)

        # Measure how far MediaPipe is trying to move the dot
        dist = math.hypot(new_x - self.prev_x, new_y - self.prev_y)

        # 1. TELEPORT: Fast head movements snap instantly to prevent lag
        if dist > TELEPORT_DISTANCE:
            self.prev_x, self.prev_y = new_x, new_y
            return int(new_x), int(new_y)

        # 2. DEADBAND (THE FIX): If it's just a tiny jitter, IGNORE IT. Freeze the point.
        if dist < DEADBAND_RADIUS:
            return int(self.prev_x), int(self.prev_y)

        # 3. SMOOTHING: Normal, intentional head movement gets smoothed
        smoothed_x = (self.prev_x * (1 - self.alpha)) + (new_x * self.alpha)
        smoothed_y = (self.prev_y * (1 - self.alpha)) + (new_y * self.alpha)

        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return int(smoothed_x), int(smoothed_y)


# Initialize trackers
sensor_data = {name: {'score': 0, 'history': deque(maxlen=SCORE_HISTORY_LEN)} for name in LANDMARK_IDS}
stabilizers = {name: PointStabilizer() for name in LANDMARK_IDS}
center_stabilizer = PointStabilizer(alpha=0.08)  # Slightly faster for the anchor
session_fit_scores = []
is_locked = False


def calculate_score(intensity):
    if np.isnan(intensity): return 0
    fraction = intensity / 255.0
    real_temp = T_MIN + (fraction * (T_MAX - T_MIN))
    score = ((real_temp - T_MIN) / (T_MAX - T_MIN)) * 100
    return int(np.clip(score, 0, 100))


def draw_dashboard(frame, sensor_data, is_locked, live_fit_score):
    h, w, _ = frame.shape
    # Left Panel
    cv2.rectangle(frame, (0, 0), (320, h), (15, 15, 15), -1)
    cv2.putText(frame, "STABILIZED ANALYTICS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    status_color = (0, 255, 0) if is_locked else (0, 165, 255)
    status_text = "MODE: LOCKED" if is_locked else "MODE: ADJUSTING"

    cv2.putText(frame, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, "WASD: Move | ZX: Scale | Space: Lock", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180),
                1)

    # Offset Info
    cv2.putText(frame, f"X:{global_offset_x} Y:{global_offset_y} S:{scale_factor:.2f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Fit Score
    fit_color = (0, 255, 0) if live_fit_score > 60 else (0, 0, 255)
    cv2.putText(frame, f"FIT SCORE: {live_fit_score}%", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fit_color, 2)

    # Sensor Bars
    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 200 + (i * 50)

        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        bar_len = int(score * 1.5)
        # Background bar
        cv2.rectangle(frame, (110, y_pos - 12), (260, y_pos + 4), (40, 40, 40), -1)
        # Value bar
        cv2.rectangle(frame, (110, y_pos - 12), (110 + bar_len, y_pos + 4), color, -1)
        cv2.putText(frame, f"{score}", (270, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    global is_locked, global_offset_x, global_offset_y, scale_factor

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb_frame)
        total_leak_score = 0
        valid_sensors = 0

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # 1. STABILIZE FACE ANCHOR (Nose bridge)
            raw_nose = face.landmark[1]
            anchor_x, anchor_y = center_stabilizer.update(int(raw_nose.x * w), int(raw_nose.y * h))

            for name, landmark_id in LANDMARK_IDS.items():
                lm = face.landmark[landmark_id]

                # Calculate relative distance from anchor
                dist_x = (lm.x * w) - (raw_nose.x * w)
                dist_y = (lm.y * h) - (raw_nose.y * h)

                # Apply Scale and Global Offset relative to stabilized anchor
                target_x = anchor_x + int(dist_x * scale_factor) + global_offset_x
                target_y = anchor_y + int(dist_y * scale_factor) + global_offset_y

                # 2. STABILIZE INDIVIDUAL SENSOR POINT
                stab_x, stab_y = stabilizers[name].update(target_x, target_y)

                # Clamp to frame boundaries
                stab_x = np.clip(stab_x, 10, w - 10)
                stab_y = np.clip(stab_y, 10, h - 10)

                # Read Thermal/Grayscale Data
                roi = gray_frame[stab_y - 5:stab_y + 5, stab_x - 5:stab_x + 5]
                raw_score = calculate_score(np.mean(roi)) if roi.size > 0 else 0

                # Smooth the data value
                sensor_data[name]['history'].append(raw_score)
                avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                sensor_data[name]['score'] = avg_score

                total_leak_score += avg_score
                valid_sensors += 1

                # Visualization
                color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
                # Draw "connection" line to show tracking
                cv2.line(frame, (anchor_x + global_offset_x, anchor_y + global_offset_y), (stab_x, stab_y),
                         (100, 100, 100), 1)
                cv2.circle(frame, (stab_x, stab_y), 6, color, -1)
                cv2.circle(frame, (stab_x, stab_y), 2, (255, 255, 255), -1)

        # Calculate Fit Score
        live_fit_score = 0
        if valid_sensors > 0:
            avg_heat = total_leak_score / valid_sensors
            live_fit_score = max(0, 100 - int(avg_heat))
            if is_locked: session_fit_scores.append(live_fit_score)

        draw_dashboard(frame, sensor_data, is_locked, live_fit_score)
        cv2.imshow('Calibration Master - Pro Stability', frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_locked = not is_locked

        if not is_locked:
            if key == ord('w'):
                global_offset_y -= 1
            elif key == ord('s'):
                global_offset_y += 1
            elif key == ord('a'):
                global_offset_x -= 1
            elif key == ord('d'):
                global_offset_x += 1
            elif key == ord('z'):
                scale_factor -= 0.01
            elif key == ord('x'):
                scale_factor += 0.01

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()