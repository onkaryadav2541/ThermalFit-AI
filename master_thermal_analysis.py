import cv2
import numpy as np
import traceback
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'

# PHYSICS CALIBRATION
T_MIN = 22.0
T_MAX = 35.0
LEAK_THRESHOLD_SCORE = 45

# VISUAL SETTINGS
DOT_RADIUS = 12  # Big visible dots
SENSING_AREA = 6  # Radius of reading (Total box = 12x12 pixels)

point_names = ["Nose Bridge", "Left Cheek", "Right Cheek", "Left Chin", "Right Chin"]
sensor_data = {name: {'score': 0, 'history': deque(maxlen=10)} for name in point_names}
fixed_points = []


def calculate_score(intensity):
    if np.isnan(intensity): return 0
    fraction = intensity / 255.0
    real_temp = T_MIN + (fraction * (T_MAX - T_MIN))
    score = ((real_temp - T_MIN) / (T_MAX - T_MIN)) * 100
    return int(np.clip(score, 0, 100))


def draw_dashboard(frame, sensor_data, worst_leak_score, worst_leak_name):
    h, w, _ = frame.shape
    # Draw Panel
    cv2.rectangle(frame, (0, 0), (320, h), (10, 10, 10), -1)
    cv2.putText(frame, "STATIC SENSORS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(frame, (10, 50), (310, 50), (100, 100, 100), 1)

    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        # Color Logic
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)

        y_pos = 90 + (i * 60)

        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Bar Graph
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (100, y_pos - 15), (100 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (100, y_pos - 15), (250, y_pos + 5), (50, 50, 50), 1)

        # Score Number
        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Status Box
    if worst_leak_score > LEAK_THRESHOLD_SCORE:
        alert_color = (0, 0, 255)
        status_text = f"LEAK: {worst_leak_name}"
    else:
        alert_color = (0, 255, 0)
        status_text = "SEAL SECURE"

    cv2.rectangle(frame, (10, h - 80), (310, h - 20), alert_color, -1)
    cv2.putText(frame, status_text, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    try:
        global fixed_points
        cap = cv2.VideoCapture(VIDEO_PATH)

        # --- PHASE 1: CLICK FIXED POINTS ---
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Video not found.")
            return

        h, w, _ = first_frame.shape
        print(f"Video Paused. CLICK 5 FIXED POINTS.")

        def mouse_callback(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN and len(fixed_points) < 5:
                # Just store the X,Y coordinate. No AI.
                fixed_points.append((x, y))
                print(f"Fixed Point {len(fixed_points)} at {x},{y}")

        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)

        while len(fixed_points) < 5:
            disp = first_frame.copy()
            cv2.putText(disp, f"CLICK {5 - len(fixed_points)} POINTS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)
            cv2.putText(disp, f"Next: {point_names[len(fixed_points)]}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            for (fx, fy) in fixed_points:
                cv2.circle(disp, (fx, fy), DOT_RADIUS, (0, 0, 255), -1)

            cv2.imshow('Calibration', disp)
            if cv2.waitKey(50) & 0xFF == ord('q'): return
        cv2.destroyWindow('Calibration')

        # --- PHASE 2: STATIC ANALYSIS ---
        print("REWINDING VIDEO")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = int(1000 / fps)

        # DATA FOR FINAL REPORT
        leak_duration_counters = {name: 0 for name in point_names}
        total_frames_analyzed = 0
        all_session_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video Finished.")
                break

            h, w, _ = frame.shape
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            worst_score = 0
            worst_name = "None"
            current_frame_avg = 0

            for i, (fx, fy) in enumerate(fixed_points):
                name = point_names[i]

                # Use the FIXED coordinates
                cur_x, cur_y = fx, fy

                # Safety Clip
                cur_x = np.clip(cur_x, SENSING_AREA, w - SENSING_AREA - 1)
                cur_y = np.clip(cur_y, SENSING_AREA, h - SENSING_AREA - 1)

                # Measurement
                roi = gray_frame[cur_y - SENSING_AREA: cur_y + SENSING_AREA,
                      cur_x - SENSING_AREA: cur_x + SENSING_AREA]

                if roi.size > 0:
                    raw_score = calculate_score(np.mean(roi))
                else:
                    raw_score = 0

                sensor_data[name]['history'].append(raw_score)
                avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                sensor_data[name]['score'] = avg_score

                # --- STATS TRACKING ---
                if avg_score > LEAK_THRESHOLD_SCORE:
                    leak_duration_counters[name] += 1

                current_frame_avg += avg_score
                if avg_score > worst_score:
                    worst_score = avg_score
                    worst_name = name

                # Draw Dot
                color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
                cv2.circle(frame, (cur_x, cur_y), DOT_RADIUS, color, -1)
                cv2.circle(frame, (cur_x, cur_y), 3, (255, 255, 255), -1)

            draw_dashboard(frame, sensor_data, worst_score, worst_name)

            total_frames_analyzed += 1
            all_session_scores.append(current_frame_avg / 5)

            cv2.imshow('Final Master Project - Static', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

        # --- PRINT FINAL REPORT ---
        print("\n" + "=" * 50)
        print("          FINAL LEAK REPORT")
        print("=" * 50)

        failed_locations = []

        if total_frames_analyzed > 0:
            for name in point_names:
                frames_leaked = leak_duration_counters[name]
                percent_leaked = (frames_leaked / total_frames_analyzed) * 100

                if percent_leaked > 5:  # If leaked more than 5% of time
                    status = "FAIL - LEAK DETECTED"
                    failed_locations.append(name)
                    color_code = "\033[91m"  # Red text code (works in some terminals)
                else:
                    status = "PASS - SECURE"
                    color_code = "\033[92m"  # Green text code

                print(f"{name:.<25} {status} ({int(percent_leaked)}% of time)")

            print("-" * 50)
            if failed_locations:
                print(f"CRITICAL LEAKS FOUND AT:")
                for loc in failed_locations:
                    print(f" -> {loc}")

                final_score = int(sum(all_session_scores) / len(all_session_scores))
                print(f"\nOVERALL FIT SCORE: {100 - final_score}/100")
            else:
                print("RESULT: PERFECT FIT! No significant leaks detected.")

        print("=" * 50)

    except Exception as e:
        print("\nCRASH DETECTED")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == '__main__':
    main()