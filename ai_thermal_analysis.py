import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'

# PHYSICS SETTINGS
LEAK_THRESHOLD_SCORE = 45

# STABILITY SETTINGS
DEADZONE = 8  # High stability: Ignore movements smaller than 8 pixels
SMOOTH_FACTOR = 0.2  # Very slow smoothing to prevent jitter (0.1 = Rigid)

# RIGID BODY OFFSETS (Distance from Face Center)
# Adjust these numbers to move dots Up/Down/Left/Right relative to center
OFFSETS = {
    "Nose Bridge": (0, -40),  # 0px Left, 40px Up from center
    "Left Cheek": (-50, -10),  # 50px Left, 10px Up
    "Right Cheek": (50, -10),  # 50px Right, 10px Up
    "Left Chin": (-30, 60),  # 30px Left, 60px Down
    "Right Chin": (30, 60)  # 30px Right, 60px Down
}

# STATE VARIABLES
sensor_data = {name: {'score': 0, 'history': deque(maxlen=15)} for name in OFFSETS}
last_center = None
t_min_calibrated = 22.0
t_max_calibrated = 35.0


def auto_calibrate(frame):
    """Scans frame to find the hottest pixel to set T_MAX correctly."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    max_val = np.max(gray)
    min_val = np.min(gray)

    # Calculate approximate temps
    t_max = 35.0  # Assume body temp
    t_min = 22.0  # Room temp

    # Map pixel intensity to this range
    return t_min, t_max, max_val


def calculate_score(intensity, max_pixel_val):
    if np.isnan(intensity): return 0
    # Normalize based on the hottest pixel seen in video
    fraction = intensity / max_pixel_val

    # Score calculation
    score = fraction * 100
    return int(np.clip(score, 0, 100))


def draw_dashboard(frame, sensor_data, worst_leak_name, worst_leak_score, is_locked):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (320, h), (15, 15, 15), -1)

    cv2.putText(frame, "RIGID TRACKING", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Status Light
    color = (0, 255, 0) if is_locked else (0, 165, 255)
    text = "LOCKED" if is_locked else "MOVING"
    cv2.circle(frame, (290, 35), 6, color, -1)
    cv2.putText(frame, text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.line(frame, (10, 75), (310, 75), (100, 100, 100), 1)

    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 120 + (i * 60)

        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (110, y_pos - 15), (110 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (110, y_pos - 15), (250, y_pos + 5), (50, 50, 50), 1)
        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Status Box
    if worst_leak_score > LEAK_THRESHOLD_SCORE:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 0, 255), -1)
        cv2.putText(frame, f"LEAK: {worst_leak_name}", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 255, 0), -1)
        cv2.putText(frame, "SEAL SECURE", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    global last_center

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    # 1. AUTO-CALIBRATION PHASE
    print("Calibrating Temperature Scale...")
    ret, frame = cap.read()
    if not ret: return
    _, _, global_max_pixel = auto_calibrate(frame)
    print(f"Calibration Complete. Max Pixel Intensity: {global_max_pixel}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind

    all_session_scores = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb_frame)

        is_locked = True

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # --- RIGID BODY LOGIC ---
            # Instead of tracking nose/chin separately (which jitter),
            # we track the CENTER OF GRAVITY of the face.

            xs = [lm.x for lm in face.landmark]
            ys = [lm.y for lm in face.landmark]

            avg_x = int((sum(xs) / len(xs)) * w)
            avg_y = int((sum(ys) / len(ys)) * h)

            # Deadzone Check
            should_update = False
            if last_center is None:
                should_update = True
                last_center = (avg_x, avg_y)
            else:
                dist = np.sqrt((avg_x - last_center[0]) ** 2 + (avg_y - last_center[1]) ** 2)
                if dist > DEADZONE:
                    should_update = True
                    is_locked = False

            # Smooth Update
            if should_update:
                old_x, old_y = last_center
                new_x = int((SMOOTH_FACTOR * old_x) + ((1 - SMOOTH_FACTOR) * avg_x))
                new_y = int((SMOOTH_FACTOR * old_y) + ((1 - SMOOTH_FACTOR) * avg_y))
                last_center = (new_x, new_y)

        # --- DRAWING ---
        worst_score = 0
        worst_name = "None"
        current_frame_avg = 0

        if last_center is not None:
            cx, cy = last_center

            # Draw Center Anchor (Yellow Dot)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            for name, (off_x, off_y) in OFFSETS.items():
                # Calculate position based on RIGID OFFSET from center
                dx = cx + off_x
                dy = cy + off_y

                # Clamp
                dx = np.clip(dx, 6, w - 7)
                dy = np.clip(dy, 6, h - 7)

                # Measure
                roi = gray_frame[dy - 6:dy + 6, dx - 6:dx + 6]
                if roi.size > 0:
                    raw_score = calculate_score(np.mean(roi), global_max_pixel)
                else:
                    raw_score = 0

                sensor_data[name]['history'].append(raw_score)
                avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                sensor_data[name]['score'] = avg_score

                current_frame_avg += avg_score
                if avg_score > worst_score:
                    worst_score = avg_score
                    worst_name = name

                # Draw Dots
                color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
                cv2.circle(frame, (dx, dy), 10, color, -1)
                cv2.circle(frame, (dx, dy), 2, (255, 255, 255), -1)

        all_session_scores.append(current_frame_avg / 5)

        draw_dashboard(frame, sensor_data, worst_name, worst_score, is_locked)

        cv2.imshow('AI Thermal Analysis - Rigid Body', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_session_scores) > 0:
        final_avg = int(sum(all_session_scores) / len(all_session_scores))
        print(f"\nFINAL AI SCORE: {100 - final_avg}/100")


if __name__ == '__main__':
    main()