import cv2
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Videos\Captures\nutonguy.mp4'

# PHYSICS CALIBRATION
T_MIN = 22.0  # Room Temp (Score 0)
T_MAX = 35.0  # Breath Temp (Score 100)
LEAK_THRESHOLD_SCORE = 45

# SENSOR CONFIGURATION
points = []
point_names = ["Nose Bridge", "Left Cheek", "Right Cheek", "Left Chin", "Right Chin"]
sensor_data = {name: {'score': 0, 'history': deque(maxlen=5)} for name in point_names}


# ---------------------

def click_event(event, x, y, flags, params):
    # User clicks 5 times to set sensors
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 5:
            points.append((x, y))
            # Draw marker
            cv2.circle(params, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(params, str(len(points)), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow('STEP 1: Calibrate Sensors', params)


def calculate_score(intensity):
    # Convert Brightness -> Score (0-100)
    fraction = intensity / 255.0
    real_temp = T_MIN + (fraction * (T_MAX - T_MIN))
    score = ((real_temp - T_MIN) / (T_MAX - T_MIN)) * 100
    return int(np.clip(score, 0, 100))


def draw_dashboard(frame, sensor_data, worst_leak_name, worst_leak_score):
    h, w, _ = frame.shape
    # Sidebar Background
    cv2.rectangle(frame, (0, 0), (320, h), (0, 0, 0), -1)

    # Header
    cv2.putText(frame, "MANUAL FIT MONITOR", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(frame, (10, 50), (310, 50), (100, 100, 100), 1)

    # List Sensors
    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 90 + (i * 50)

        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (100, y_pos - 15), (100 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (100, y_pos - 15), (250, y_pos + 5), (50, 50, 50), 1)
        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Global Alert Box
    if worst_leak_score > LEAK_THRESHOLD_SCORE:
        alert_color = (0, 0, 255)
        status_text = f"LEAK: {worst_leak_name.upper()}"
    else:
        alert_color = (0, 255, 0)
        status_text = "SEAL SECURE"

    cv2.rectangle(frame, (10, h - 80), (310, h - 20), alert_color, -1)
    cv2.putText(frame, status_text, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret: return

    # --- PHASE 1: CLICK SENSORS ---
    print("------------------------------------------------")
    print(" PLEASE CLICK 5 POINTS: Nose, L-Cheek, R-Cheek, L-Chin, R-Chin")
    print("------------------------------------------------")

    cv2.imshow('STEP 1: Calibrate Sensors', frame)
    cv2.setMouseCallback('STEP 1: Calibrate Sensors', click_event, frame)

    while len(points) < 5:
        if cv2.waitKey(100) & 0xFF == ord('q'): return

    cv2.destroyAllWindows()

    # --- PHASE 2: RUN ANALYSIS ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    delay = int(1000 / fps)

    all_session_scores = []
    print("Starting Manual Analysis...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        worst_score = 0
        worst_name = "None"
        current_frame_avg = 0

        # Loop through locked points
        for i, (x, y) in enumerate(points):
            name = point_names[i]

            # Safety check
            if y < 1 or x < 1: continue
            roi = gray_frame[y - 1:y + 2, x - 1:x + 2]
            intensity = np.mean(roi)

            # Calculate
            raw_score = calculate_score(intensity)

            # Smooth
            sensor_data[name]['history'].append(raw_score)
            avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
            sensor_data[name]['score'] = avg_score

            # Track worst
            current_frame_avg += avg_score
            if avg_score > worst_score:
                worst_score = avg_score
                worst_name = name

            # Draw Dot
            color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
            cv2.circle(frame, (x, y), 5, color, -1)

        # Draw Dashboard
        draw_dashboard(frame, sensor_data, worst_name, worst_score)
        all_session_scores.append(current_frame_avg / 5)

        cv2.imshow('Master Project - Manual Mode', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- FINAL REPORT ---
    if len(all_session_scores) > 0:
        final_avg = int(sum(all_session_scores) / len(all_session_scores))
        print("\n" + "=" * 40)
        print("     FINAL FIT TEST REPORT (MANUAL)")
        print("=" * 40)
        print(f" Total Frames Analyzed: {len(all_session_scores)}")
        print(f" Average Fit Score:     {100 - final_avg}/100")
        print("-" * 40)
        if final_avg > LEAK_THRESHOLD_SCORE:
            print(" RESULT: FAILED (Significant Leak Detected)")
        else:
            print(" RESULT: PASSED (Mask Fit is Secure)")
        print("=" * 40 + "\n")


if __name__ == '__main__':
    main()