import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Videos\Captures\nutonguy.mp4'

# PHYSICS CALIBRATION
T_MIN = 22.0
T_MAX = 35.0
LEAK_THRESHOLD_SCORE = 45

# STABILITY SETTINGS
SMOOTH_FACTOR = 0.7  # High number (0.8) = Very smooth/slow. Low (0.2) = Fast/Jittery.
MEMORY_FRAMES = 60  # If face is lost, remember position for 60 frames (2 seconds)

# SENSOR SETUP
sensor_config = {
    "Nose Bridge": 6,
    "Left Cheek": 123,
    "Right Cheek": 352,
    "Left Chin": 172,
    "Right Chin": 397
}

# GLOBAL STATE
sensor_data = {name: {'score': 0, 'history': deque(maxlen=10)} for name in sensor_config}
last_known_positions = {name: None for name in sensor_config}  # Stores (x,y)
frames_since_detection = 0


def calculate_score(intensity):
    if np.isnan(intensity): return 0
    fraction = intensity / 255.0
    real_temp = T_MIN + (fraction * (T_MAX - T_MIN))
    score = ((real_temp - T_MIN) / (T_MAX - T_MIN)) * 100
    return int(np.clip(score, 0, 100))


def draw_dashboard(frame, sensor_data, worst_leak_name, worst_leak_score, is_tracking):
    h, w, _ = frame.shape

    # 1. Background Panel
    cv2.rectangle(frame, (0, 0), (320, h), (10, 10, 10), -1)

    # 2. Header
    cv2.putText(frame, "AI THERMAL SENSE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Tracking Status Light
    status_color = (0, 255, 0) if is_tracking else (0, 255, 255)  # Green=Live, Yellow=Memory
    status_text = "LIVE TRACKING" if is_tracking else "MEMORY MODE"
    cv2.circle(frame, (290, 35), 6, status_color, -1)
    cv2.putText(frame, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    cv2.line(frame, (10, 75), (310, 75), (100, 100, 100), 1)

    # 3. List Sensors
    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 120 + (i * 60)

        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Bar Graph
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (110, y_pos - 15), (110 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (110, y_pos - 15), (250, y_pos + 5), (50, 50, 50), 1)

        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 4. Alert Box
    if worst_leak_score > LEAK_THRESHOLD_SCORE:
        alert_color = (0, 0, 255)
        status_text = f"LEAK: {worst_leak_name}"
    else:
        alert_color = (0, 255, 0)
        status_text = "SEAL SECURE"

    cv2.rectangle(frame, (10, h - 80), (310, h - 20), alert_color, -1)
    cv2.putText(frame, status_text, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    global frames_since_detection, last_known_positions

    # AI Config
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    all_session_scores = []

    print("Starting AI Analysis...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb_frame)

        face_found = False

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_found = True
            frames_since_detection = 0  # Reset counter

            # --- LIVE UPDATE ---
            for name, idx in sensor_config.items():
                lm = face_landmarks.landmark[idx]
                target_x, target_y = int(lm.x * w), int(lm.y * h)

                # Retrieve old position
                old_pos = last_known_positions[name]

                if old_pos is None:
                    # First time detection: Jump directly
                    cur_x, cur_y = target_x, target_y
                else:
                    # SMOOTHING LOGIC (Prevents Jitter)
                    old_x, old_y = old_pos
                    cur_x = int((SMOOTH_FACTOR * old_x) + ((1 - SMOOTH_FACTOR) * target_x))
                    cur_y = int((SMOOTH_FACTOR * old_y) + ((1 - SMOOTH_FACTOR) * target_y))

                # Update Memory
                last_known_positions[name] = (cur_x, cur_y)

        else:
            # --- MEMORY MODE ---
            frames_since_detection += 1
            if frames_since_detection < MEMORY_FRAMES:
                # Use old positions if tracked recently
                face_found = True  # Technically false, but we have data to show
            else:
                # Lost for too long
                face_found = False

        # --- DRAWING & MEASURING ---
        worst_score = 0
        worst_name = "None"
        current_frame_avg = 0

        # Only process if we have valid positions (from Live or Memory)
        if last_known_positions["Nose Bridge"] is not None and face_found:

            for name, pos in last_known_positions.items():
                cx, cy = pos

                # Safety Clamp
                cx = np.clip(cx, 6, w - 7)
                cy = np.clip(cy, 6, h - 7)

                # Measure (12x12 box)
                roi = gray_frame[cy - 6:cy + 6, cx - 6:cx + 6]
                if roi.size > 0:
                    raw_score = calculate_score(np.mean(roi))
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
                cv2.circle(frame, (cx, cy), 8, color, -1)  # Big Dot
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)  # Center

        all_session_scores.append(current_frame_avg / 5)

        # Draw Dashboard
        # Pass 'frames_since_detection == 0' to show if we are Live or Memo
        draw_dashboard(frame, sensor_data, worst_name, worst_score, (frames_since_detection == 0))

        cv2.imshow('AI Thermal Analysis - Stabilized', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final Report
    if len(all_session_scores) > 0:
        final_avg = int(sum(all_session_scores) / len(all_session_scores))
        print(f"\nFINAL FIT SCORE: {100 - final_avg}/100")


if __name__ == '__main__':
    main()