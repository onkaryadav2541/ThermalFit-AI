import cv2
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\with film tight rituraj.mp4'

# PHYSICS CALIBRATION
T_MIN = 22.0  # Room Temp (Score 0)
T_MAX = 35.0  # Breath Temp (Score 100)
LEAK_THRESHOLD_SCORE = 45

# SENSOR NAMES
point_names = ["Nose Bridge", "Left Cheek", "Right Cheek", "Left Chin", "Right Chin"]
sensor_data = {name: {'score': 0, 'history': deque(maxlen=15)} for name in point_names}
fixed_points = []  # Stores (x, y) coordinates


def calculate_score(intensity):
    if np.isnan(intensity): return 0
    fraction = intensity / 255.0
    real_temp = T_MIN + (fraction * (T_MAX - T_MIN))
    score = ((real_temp - T_MIN) / (T_MAX - T_MIN)) * 100
    return int(np.clip(score, 0, 100))


def draw_dashboard(frame, sensor_data, worst_leak_score, worst_leak_name):
    h, w, _ = frame.shape
    # Sidebar Background
    cv2.rectangle(frame, (0, 0), (320, h), (0, 0, 0), -1)

    # Header
    cv2.putText(frame, "STATIC SENSORS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(frame, (10, 50), (310, 50), (100, 100, 100), 1)

    # Sensors
    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']

        # Color Logic
        if score > LEAK_THRESHOLD_SCORE:
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green

        y_pos = 90 + (i * 50)

        # Draw Name
        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw Bar
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (100, y_pos - 15), (100 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (100, y_pos - 15), (250, y_pos + 5), (50, 50, 50), 1)

        # Draw Number
        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Global Alert Box
    if worst_leak_score > LEAK_THRESHOLD_SCORE:
        alert_color = (0, 0, 255)
        status_text = f"LEAK: {worst_leak_name}"
    else:
        alert_color = (0, 255, 0)
        status_text = "SEAL SECURE"

    cv2.rectangle(frame, (10, h - 80), (310, h - 20), alert_color, -1)
    cv2.putText(frame, status_text, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    global fixed_points
    cap = cv2.VideoCapture(VIDEO_PATH)

    # --- STEP 1: FREEZE & CLICK ---
    # Read the first frame to display for clicking
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read video.")
        return

    print("Video Paused. PLEASE CLICK 5 FIXED POINTS.")
    h, w, _ = first_frame.shape

    def mouse_callback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(fixed_points) < 5:
                # Simply store the coordinate. No AI. No Tracking.
                fixed_points.append((x, y))
                print(f"Point {len(fixed_points)} Fixed at: {x}, {y}")

    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)

    while len(fixed_points) < 5:
        display_frame = first_frame.copy()

        # Instructions
        cv2.putText(display_frame, f"CLICK {5 - len(fixed_points)} POINTS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        cv2.putText(display_frame, f"Next: {point_names[len(fixed_points)]}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # Draw dots where user clicked
        for (x, y) in fixed_points:
            cv2.circle(display_frame, (x, y), 6, (0, 0, 255), -1)

        cv2.imshow('Calibration', display_frame)
        if cv2.waitKey(50) & 0xFF == ord('q'): return

    cv2.destroyWindow('Calibration')

    # --- STEP 2: RUN ANALYSIS (STATIC) ---
    print("Starting Static Analysis...")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    all_session_scores = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        worst_score = 0
        worst_name = "None"
        current_frame_avg = 0

        for i, (x, y) in enumerate(fixed_points):
            name = point_names[i]

            # 1. Measurement (At FIXED coordinates)
            # Safety Check: Ensure point is inside frame
            x = np.clip(x, 2, w - 3)
            y = np.clip(y, 2, h - 3)

            roi = gray_frame[y - 2:y + 3, x - 2:x + 3]

            if roi.size > 0:
                intensity = np.mean(roi)
                raw_score = calculate_score(intensity)
            else:
                raw_score = 0

            # 2. Smooth Data
            sensor_data[name]['history'].append(raw_score)
            avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
            sensor_data[name]['score'] = avg_score

            current_frame_avg += avg_score
            if avg_score > worst_score:
                worst_score = avg_score
                worst_name = name

            # 3. Draw Dot (Static)
            color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

        draw_dashboard(frame, sensor_data, worst_score, worst_name)
        all_session_scores.append(current_frame_avg / 5)

        cv2.imshow('Final Master Project - Static', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_session_scores) > 0:
        final_avg = int(sum(all_session_scores) / len(all_session_scores))
        print(f"\nFINAL REPORT: Score {100 - final_avg}/100")


if __name__ == '__main__':
    main()