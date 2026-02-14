import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Sakshi\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'
LEAK_THRESHOLD_SCORE = 45

# --- POSITION CORRECTION (THE FIX) ---
# If points are too far RIGHT, we use a negative number here to move them LEFT.
# Try -40, -50, or -60 until it hits the face perfectly.
MOVE_ALL_POINTS_LEFT_RIGHT = -40  # Negative = Left, Positive = Right
MOVE_ALL_POINTS_UP_DOWN = 0  # Negative = Up, Positive = Down

# --- STABILIZER SETTINGS ---
STABILITY_STRENGTH = 0.1

# --- LANDMARK DEFINITIONS ---
LANDMARK_IDS = {
    "Nose Bridge": 168,
    "Left Cheek": 118,
    "Right Cheek": 347,
    "Left Chin": 58,
    "Right Chin": 288,
    "Center Chin": 152
}


# --- STABILIZER CLASS ---
class PointStabilizer:
    def __init__(self, alpha=STABILITY_STRENGTH):
        self.alpha = alpha
        self.prev_x = None
        self.prev_y = None

    def update(self, new_x, new_y):
        if self.prev_x is None:
            self.prev_x = new_x
            self.prev_y = new_y
            return int(new_x), int(new_y)
        else:
            smoothed_x = (self.prev_x * (1 - self.alpha)) + (new_x * self.alpha)
            smoothed_y = (self.prev_y * (1 - self.alpha)) + (new_y * self.alpha)
            self.prev_x = smoothed_x
            self.prev_y = smoothed_y
            return int(smoothed_x), int(smoothed_y)


# STATE
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

    cv2.putText(frame, "CENTERED AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if not is_locked:
        color = (0, 165, 255)
        text = "MODE: PREVIEW"
    else:
        color = (0, 255, 0)
        text = "MODE: RECORDING"

    cv2.circle(frame, (290, 25), 6, color, -1)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fit_color = (0, 255, 0) if live_fit_score > 60 else (0, 0, 255)
    cv2.putText(frame, "LIVE FIT SCORE:", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"{live_fit_score}", (200, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, fit_color, 2)
    cv2.line(frame, (10, 140), (310, 140), (50, 50, 50), 1)

    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 180 + (i * 60)
        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (110, y_pos - 15), (110 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (110, y_pos - 15), (250, y_pos + 5), (50, 50, 50), 1)
        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if worst_score > LEAK_THRESHOLD_SCORE:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 0, 255), -1)
        cv2.putText(frame, "LEAK DETECTED", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 255, 0), -1)
        cv2.putText(frame, "SEAL SECURE", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "[SPACE] Record   [Q] Quit", (340, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)


def main():
    global is_locked

    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    except AttributeError:
        print("Error: Use Python 3.10")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret: return
    global_max_pixel = auto_calibrate(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    print("System Loaded. Applying Position Correction...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb_frame)

        worst_score = 0
        total_leak_score = 0
        valid_sensors = 0

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            for name, landmark_id in LANDMARK_IDS.items():
                lm = face.landmark[landmark_id]

                # 1. Get Raw Position
                raw_x = int(lm.x * w)
                raw_y = int(lm.y * h)

                # 2. STABILIZE IT
                stab_x, stab_y = stabilizers[name].update(raw_x, raw_y)

                # 3. APPLY POSITION CORRECTION (Move Left/Right/Up/Down)
                final_x = stab_x + MOVE_ALL_POINTS_LEFT_RIGHT
                final_y = stab_y + MOVE_ALL_POINTS_UP_DOWN

                # Safety Clip
                final_x = np.clip(final_x, 6, w - 7)
                final_y = np.clip(final_y, 6, h - 7)

                # 4. MEASURE HEAT
                roi = gray_frame[final_y - 6:final_y + 6, final_x - 6:final_x + 6]
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

                # 5. DRAW
                color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
                cv2.circle(frame, (final_x, final_y), 8, color, -1)
                cv2.circle(frame, (final_x, final_y), 2, (255, 255, 255), -1)

        live_fit_score = 0
        if valid_sensors > 0:
            avg_heat = total_leak_score / valid_sensors
            live_fit_score = max(0, 100 - int(avg_heat))
            if is_locked:
                session_fit_scores.append(live_fit_score)

        draw_dashboard(frame, sensor_data, worst_score, is_locked, live_fit_score)
        cv2.imshow('Centered Stabilized AI', frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_locked = not is_locked

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("          FINAL AI REPORT")
    print("=" * 50)
    if len(session_fit_scores) > 0:
        final_average = int(sum(session_fit_scores) / len(session_fit_scores))
        print(f"Total Frames Recorded: {len(session_fit_scores)}")
        print(f"FINAL FIT SCORE:       {final_average}/100")
    else:
        print("No data recorded.")
    print("=" * 50)


if __name__ == '__main__':
    main()