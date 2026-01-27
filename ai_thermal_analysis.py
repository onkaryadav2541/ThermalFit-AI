import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Videos\Captures\nutonguy.mp4'

# PHYSICS CALIBRATION
T_MIN = 22.0  # Room Temp (Score 0)
T_MAX = 35.0  # Breath Temp (Score 100)
LEAK_THRESHOLD_SCORE = 45  # Stricter threshold for better accuracy

# SENSOR MEMORY (Fixes "Vanishing" issue)
MEMORY_PERSISTENCE = 30


# ---------------------

def calculate_temp_and_score(intensity):
    # Convert Pixel Brightness -> Celsius -> Score (0-100)
    fraction = intensity / 255.0
    real_temp = T_MIN + (fraction * (T_MAX - T_MIN))
    score = ((real_temp - T_MIN) / (T_MAX - T_MIN)) * 100
    return real_temp, int(np.clip(score, 0, 100))


def draw_dashboard(frame, sensor_data, worst_leak_name, worst_leak_score):
    # Draws the black sidebar and all text.
    # This runs EVERY frame, so it never vanishes.

    # 1. Background
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (320, h), (0, 0, 0), -1)  # Full height sidebar

    # 2. Header
    cv2.putText(frame, "MASK FIT MONITOR", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(frame, (10, 50), (310, 50), (100, 100, 100), 1)

    # 3. List Sensors
    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']

        # Color Logic
        if score > LEAK_THRESHOLD_SCORE:
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green

        y_pos = 90 + (i * 50)

        # Name
        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Bar Graph
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (100, y_pos - 15), (100 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (100, y_pos - 15), (250, y_pos + 5), (50, 50, 50), 1)  # Outline

        # Number - FIXED HERE (Removed FONT_HERSHEY_BOLD)
        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 4. Global Alert Box (Bottom)
    if worst_leak_score > LEAK_THRESHOLD_SCORE:
        alert_color = (0, 0, 255)
        status_text = f"LEAK: {worst_leak_name.upper()}"
    else:
        alert_color = (0, 255, 0)
        status_text = "SEAL SECURE"

    cv2.rectangle(frame, (10, h - 80), (310, h - 20), alert_color, -1)
    cv2.putText(frame, status_text, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    # Setup AI
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Sensor Definitions
    sensor_config = {
        "Nose Bridge": 6,
        "Left Cheek": 123,
        "Right Cheek": 352,
        "Left Chin": 172,
        "Right Chin": 397
    }

    # Initialize Data Storage
    sensor_data = {name: {'score': 0, 'history': deque(maxlen=5)} for name in sensor_config}

    # Stats for Final Report
    all_session_scores = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    delay = int((1000 / fps))

    print("Starting Analysis... Press 'q' to stop and see Final Report.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for name, idx in sensor_config.items():
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)

                    # Only update if the point is actually on screen
                    if 1 < x < w - 1 and 1 < y < h - 1:
                        # 1. Measure Temp
                        roi = gray_frame[y - 1:y + 2, x - 1:x + 2]
                        intensity = np.mean(roi)
                        _, raw_score = calculate_temp_and_score(intensity)

                        # 2. Smooth and Store
                        sensor_data[name]['history'].append(raw_score)
                        avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                        sensor_data[name]['score'] = avg_score

                        # 3. Draw Dots on Face
                        color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
                        cv2.circle(frame, (x, y), 4, color, -1)

        # --- LOGIC: KEEP DASHBOARD ALIVE ---
        worst_score = 0
        worst_name = "None"
        current_frame_avg = 0

        for name, data in sensor_data.items():
            s = data['score']
            current_frame_avg += s
            if s > worst_score:
                worst_score = s
                worst_name = name

        # Save average for the final report
        all_session_scores.append(current_frame_avg / 5)

        # DRAW DASHBOARD
        draw_dashboard(frame, sensor_data, worst_name, worst_score)

        cv2.imshow('Final Master Project - Leak Detection', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- FINAL REPORT GENERATION ---
    if len(all_session_scores) > 0:
        final_avg = int(sum(all_session_scores) / len(all_session_scores))
        print("\n" + "=" * 40)
        print("     FINAL FIT TEST REPORT")
        print("=" * 40)
        print(f" Total Frames Analyzed: {len(all_session_scores)}")
        print(f" Average Fit Score:     {100 - final_avg}/100")  # Invert so 100 is Good
        print("-" * 40)

        if final_avg > LEAK_THRESHOLD_SCORE:
            print(" RESULT: FAILED (Significant Leak Detected)")
        else:
            print(" RESULT: PASSED (Mask Fit is Secure)")
        print("=" * 40 + "\n")


if __name__ == '__main__':
    main()