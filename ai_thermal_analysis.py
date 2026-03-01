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
SCORE_HISTORY_LEN = 25

# --- LANDMARK DEFINITIONS ---
LANDMARK_IDS = {
    "Nose Bridge": 168,
    "Left Cheek": 118,
    "Right Cheek": 347,
    "Left Chin": 58,
    "Right Chin": 288,
    "Center Chin": 152
}


class SmartVelocityStabilizer:
    def __init__(self, min_alpha=0.01, max_alpha=0.3, sensitivity=0.05):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.sensitivity = sensitivity
        self.prev_x = None
        self.prev_y = None

    def update(self, new_x, new_y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = new_x, new_y
            return int(new_x), int(new_y)

        dist = math.hypot(new_x - self.prev_x, new_y - self.prev_y)
        if dist > 80:
            self.prev_x, self.prev_y = new_x, new_y
            return int(new_x), int(new_y)

        dynamic_alpha = self.min_alpha + (dist * self.sensitivity)
        dynamic_alpha = min(self.max_alpha, dynamic_alpha)

        smoothed_x = (self.prev_x * (1 - dynamic_alpha)) + (new_x * dynamic_alpha)
        smoothed_y = (self.prev_y * (1 - dynamic_alpha)) + (new_y * dynamic_alpha)

        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return int(smoothed_x), int(smoothed_y)


# Initialize trackers
sensor_data = {name: {'score': 0, 'history': deque(maxlen=SCORE_HISTORY_LEN)} for name in LANDMARK_IDS}
center_stabilizer = SmartVelocityStabilizer(min_alpha=0.05, max_alpha=0.4)

session_fit_scores = []
is_locked = False

# --- NEW: RIGID LOCK OFFSETS ---
# When locked, we save the exact distance of each dot from the nose bridge
rigid_offsets = {}


def calculate_score(intensity):
    if np.isnan(intensity): return 0
    fraction = intensity / 255.0
    real_temp = T_MIN + (fraction * (T_MAX - T_MIN))
    score = ((real_temp - T_MIN) / (T_MAX - T_MIN)) * 100
    return int(np.clip(score, 0, 100))


def draw_dashboard(frame, sensor_data, is_locked, live_fit_score):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (320, h), (15, 15, 15), -1)
    cv2.putText(frame, "RIGID LOCK ANALYTICS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    status_color = (0, 255, 0) if is_locked else (0, 165, 255)
    status_text = "MODE: RIGID LOCKED" if is_locked else "MODE: ADJUSTING"

    cv2.putText(frame, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, "WASD: Move | ZX: Scale | Space: Lock", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180),
                1)
    cv2.putText(frame, f"X:{global_offset_x} Y:{global_offset_y} S:{scale_factor:.2f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    fit_color = (0, 255, 0) if live_fit_score > 60 else (0, 0, 255)
    cv2.putText(frame, f"FIT SCORE: {live_fit_score}%", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fit_color, 2)

    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 200 + (i * 50)

        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (110, y_pos - 12), (260, y_pos + 4), (40, 40, 40), -1)
        cv2.rectangle(frame, (110, y_pos - 12), (110 + bar_len, y_pos + 4), color, -1)
        cv2.putText(frame, f"{score}", (270, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    global is_locked, global_offset_x, global_offset_y, scale_factor, rigid_offsets

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

            # 1. ALWAYS STABILIZE THE NOSE (The Anchor)
            raw_nose = face.landmark[1]
            anchor_x, anchor_y = center_stabilizer.update(int(raw_nose.x * w), int(raw_nose.y * h))

            for name, landmark_id in LANDMARK_IDS.items():
                lm = face.landmark[landmark_id]

                # MODE: ADJUSTING (Dots can wiggle as MediaPipe guesses)
                if not is_locked:
                    dist_x = (lm.x * w) - (raw_nose.x * w)
                    dist_y = (lm.y * h) - (raw_nose.y * h)

                    target_x = anchor_x + int(dist_x * scale_factor) + global_offset_x
                    target_y = anchor_y + int(dist_y * scale_factor) + global_offset_y

                    # Save the snapshot of where this dot is relative to the nose
                    rigid_offsets[name] = (target_x - anchor_x, target_y - anchor_y)

                # MODE: RIGID LOCKED (Dots are cemented to the anchor)
                else:
                    target_x = anchor_x + rigid_offsets[name][0]
                    target_y = anchor_y + rigid_offsets[name][1]

                stab_x = np.clip(target_x, 10, w - 10)
                stab_y = np.clip(target_y, 10, h - 10)

                # READ THERMAL DATA
                roi = gray_frame[stab_y - 5:stab_y + 5, stab_x - 5:stab_x + 5]
                raw_score = calculate_score(np.mean(roi)) if roi.size > 0 else 0

                sensor_data[name]['history'].append(raw_score)
                avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                sensor_data[name]['score'] = avg_score

                total_leak_score += avg_score
                valid_sensors += 1

                color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
                cv2.line(frame, (anchor_x + global_offset_x, anchor_y + global_offset_y), (stab_x, stab_y),
                         (100, 100, 100), 1)
                cv2.circle(frame, (stab_x, stab_y), 6, color, -1)
                cv2.circle(frame, (stab_x, stab_y), 2, (255, 255, 255), -1)

        live_fit_score = 0
        if valid_sensors > 0:
            avg_heat = total_leak_score / valid_sensors
            live_fit_score = max(0, 100 - int(avg_heat))
            if is_locked: session_fit_scores.append(live_fit_score)

        draw_dashboard(frame, sensor_data, is_locked, live_fit_score)
        cv2.imshow('Rigid Lock Master', frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_locked = not is_locked  # Toggles the Rigid Lock

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