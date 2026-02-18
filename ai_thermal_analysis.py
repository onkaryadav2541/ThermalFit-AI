import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'
LEAK_THRESHOLD_SCORE = 45

# --- INITIAL CALIBRATION ---
# You can adjust these live using keys!
global_offset_x = -40  # Starting guess
global_offset_y = 0
scale_factor = 1.0  # 1.0 = Normal, 0.8 = Smaller Spread, 1.2 = Wider Spread

# --- STABILIZER SETTINGS ---
STABILITY_STRENGTH = 0.05
TELEPORT_DISTANCE = 50

# --- LANDMARK DEFINITIONS ---
LANDMARK_IDS = {
    "Nose Bridge": 168,  # Anchor Point
    "Left Cheek": 118,
    "Right Cheek": 347,
    "Left Chin": 58,
    "Right Chin": 288,
    "Center Chin": 152
}


class PointStabilizer:
    def __init__(self, alpha=STABILITY_STRENGTH):
        self.alpha = alpha
        self.prev_x = None
        self.prev_y = None

    def update(self, new_x, new_y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = new_x, new_y
            return int(new_x), int(new_y)

        dist = math.hypot(new_x - self.prev_x, new_y - self.prev_y)
        if dist > TELEPORT_DISTANCE:
            self.prev_x, self.prev_y = new_x, new_y
            return int(new_x), int(new_y)

        smoothed_x = (self.prev_x * (1 - self.alpha)) + (new_x * self.alpha)
        smoothed_y = (self.prev_y * (1 - self.alpha)) + (new_y * self.alpha)
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return int(smoothed_x), int(smoothed_y)


sensor_data = {name: {'score': 0, 'history': deque(maxlen=20)} for name in LANDMARK_IDS}
stabilizers = {name: PointStabilizer() for name in LANDMARK_IDS}
session_fit_scores = []
is_locked = False


def auto_calibrate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.max(gray)


def calculate_score(intensity, max_pixel_val):
    if np.isnan(intensity): return 0
    if max_pixel_val == 0: return 0
    fraction = intensity / max_pixel_val
    return int(np.clip(fraction * 100, 0, 100))


def draw_dashboard(frame, sensor_data, worst_score, is_locked, live_fit_score):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (320, h), (10, 10, 10), -1)

    cv2.putText(frame, "CALIBRATION MASTER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if not is_locked:
        color = (0, 165, 255)
        text = "MODE: ADJUSTING"
        sub = "WASD=Move | ZX=Scale"
    else:
        color = (0, 255, 0)
        text = "MODE: LOCKED"
        sub = "Recording..."

    cv2.circle(frame, (290, 25), 6, color, -1)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, sub, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Info Display
    cv2.putText(frame, f"X:{global_offset_x} Y:{global_offset_y}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)
    cv2.putText(frame, f"Scale: {scale_factor:.2f}", (140, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    fit_color = (0, 255, 0) if live_fit_score > 60 else (0, 0, 255)
    cv2.putText(frame, f"FIT: {live_fit_score}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fit_color, 2)

    # Bars
    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 180 + (i * 55)
        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (110, y_pos - 10), (110 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (110, y_pos - 10), (250, y_pos + 5), (50, 50, 50), 1)
        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    global is_locked, global_offset_x, global_offset_y, scale_factor

    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    except:
        print("Error: Use Python 3.10")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret: return
    global_max_pixel = auto_calibrate(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Use W/A/S/D to Move. Use Z/X to Resize.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb_frame)
        valid_sensors = 0
        total_leak_score = 0
        worst_score = 0

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # Get Nose Position (Center of the face)
            nose_lm = face.landmark[1]
            face_center_x, face_center_y = int(nose_lm.x * w), int(nose_lm.y * h)

            for name, landmark_id in LANDMARK_IDS.items():
                lm = face.landmark[landmark_id]

                # 1. RAW POSITION
                raw_x = int(lm.x * w)
                raw_y = int(lm.y * h)

                # 2. SCALE CORRECTION (Spread Adjustment)
                # Calculate distance from center
                dist_x = raw_x - face_center_x
                dist_y = raw_y - face_center_y

                # Multiply by Scale Factor (Spread out or Shrink in)
                scaled_x = face_center_x + int(dist_x * scale_factor)
                scaled_y = face_center_y + int(dist_y * scale_factor)

                # 3. POSITION CORRECTION (WASD Offset)
                final_x = scaled_x + global_offset_x
                final_y = scaled_y + global_offset_y

                # 4. STABILIZE
                stab_x, stab_y = stabilizers[name].update(final_x, final_y)

                # Clip
                stab_x = np.clip(stab_x, 6, w - 7)
                stab_y = np.clip(stab_y, 6, h - 7)

                # MEASURE
                roi = gray_frame[stab_y - 6:stab_y + 6, stab_x - 6:stab_x + 6]
                if roi.size > 0:
                    raw_score = calculate_score(np.mean(roi), global_max_pixel)
                else:
                    raw_score = 0

                sensor_data[name]['history'].append(raw_score)
                avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                sensor_data[name]['score'] = avg_score
                total_leak_score += avg_score
                valid_sensors += 1
                if avg_score > worst_score: worst_score = avg_score

                # DRAW
                color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
                # Draw line to center to visualize scale
                cv2.line(frame, (face_center_x + global_offset_x, face_center_y + global_offset_y), (stab_x, stab_y),
                         (50, 50, 50), 1)
                cv2.circle(frame, (stab_x, stab_y), 8, color, -1)
                cv2.circle(frame, (stab_x, stab_y), 2, (255, 255, 255), -1)

        # SCORE
        live_fit_score = 0
        if valid_sensors > 0:
            avg_heat = total_leak_score / valid_sensors
            live_fit_score = max(0, 100 - int(avg_heat))
            if is_locked: session_fit_scores.append(live_fit_score)

        draw_dashboard(frame, sensor_data, worst_score, is_locked, live_fit_score)
        cv2.imshow('Calibration Master', frame)

        # INPUTS
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_locked = not is_locked

        # ADJUSTMENTS
        if not is_locked:
            if key == ord('w'):
                global_offset_y -= 2
            elif key == ord('s'):
                global_offset_y += 2
            elif key == ord('a'):
                global_offset_x -= 2
            elif key == ord('d'):
                global_offset_x += 2
            elif key == ord('z'):
                scale_factor -= 0.05  # Shrink
            elif key == ord('x'):
                scale_factor += 0.05  # Expand

    cap.release()
    cv2.destroyAllWindows()

    print("\nFINAL REPORT")
    if len(session_fit_scores) > 0:
        print(f"FIT SCORE: {int(sum(session_fit_scores) / len(session_fit_scores))}/100")


if __name__ == '__main__':
    main()