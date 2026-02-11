import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Sakshi\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'
LEAK_THRESHOLD_SCORE = 45

# --- CRITICAL: THE EXACT FACE ANCHOR POINTS (MediaPipe IDs) ---
# These numbers represent specific points on the 3D face mesh.
# They will rotate and move perfectly with your head.
SENSOR_LANDMARKS = {
    "Nose Bridge": 6,  # Center of nose
    "Left Cheek": 118,  # Middle of left cheek
    "Right Cheek": 347,  # Middle of right cheek
    "Left Chin": 58,  # Jawline left
    "Right Chin": 288,  # Jawline right
    "Center Chin": 152  # Bottom of chin
}

# STATE
sensor_data = {name: {'score': 0, 'history': deque(maxlen=20)} for name in SENSOR_LANDMARKS}
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
    cv2.rectangle(frame, (0, 0), (320, h), (15, 15, 15), -1)

    # Header
    cv2.putText(frame, "THERMAL FIT AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if not is_locked:
        color = (0, 165, 255)
        text = "MODE: PREVIEW"
        sub = "Press SPACE to Record"
    else:
        color = (0, 255, 0)
        text = "MODE: RECORDING"
        sub = "Gathering Data..."

    cv2.circle(frame, (290, 25), 6, color, -1)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, sub, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Fit Score
    fit_color = (0, 255, 0) if live_fit_score > 60 else (0, 0, 255)
    cv2.putText(frame, "LIVE SCORE:", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"{live_fit_score}", (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, fit_color, 2)
    cv2.line(frame, (10, 130), (310, 130), (100, 100, 100), 1)

    # Sensor Bars
    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 170 + (i * 60)
        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (110, y_pos - 15), (110 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (110, y_pos - 15), (250, y_pos + 5), (50, 50, 50), 1)
        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Alert Box
    if worst_score > LEAK_THRESHOLD_SCORE:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 0, 255), -1)
        cv2.putText(frame, "LEAK DETECTED", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 255, 0), -1)
        cv2.putText(frame, "SEAL SECURE", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    global is_locked

    # 1. SETUP MEDIA PIPE FACE MESH
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret: return
    global_max_pixel = auto_calibrate(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("System Loaded. Dots are now GLUED to landmarks.")

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

        # --- TRACKING LOGIC ---
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            for name, landmark_id in SENSOR_LANDMARKS.items():
                # GET EXACT LANDMARK POSITION
                # This logic ties the dot to the specific skin point (118, 347, etc)
                lm = face.landmark[landmark_id]
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Keep dot inside screen
                cx = np.clip(cx, 6, w - 7)
                cy = np.clip(cy, 6, h - 7)

                # MEASURE HEAT
                roi = gray_frame[cy - 6:cy + 6, cx - 6:cx + 6]
                if roi.size > 0:
                    raw_score = calculate_score(np.mean(roi), global_max_pixel)
                else:
                    raw_score = 0

                # UPDATE DATA
                sensor_data[name]['history'].append(raw_score)
                avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                sensor_data[name]['score'] = avg_score

                total_leak_score += avg_score
                valid_sensors += 1

                if avg_score > worst_score: worst_score = avg_score

                # DRAW DOTS
                color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
                cv2.circle(frame, (cx, cy), 8, color, -1)  # The dot
                cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)  # Center white dot
                # Optional: Draw line to text so we know which is which
                # cv2.putText(frame, name, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # CALCULATE LIVE SCORE
        live_fit_score = 0
        if valid_sensors > 0:
            avg_heat = total_leak_score / valid_sensors
            live_fit_score = max(0, 100 - int(avg_heat))

            if is_locked:
                session_fit_scores.append(live_fit_score)

        draw_dashboard(frame, sensor_data, worst_score, is_locked, live_fit_score)
        cv2.imshow('Sticky Thermal AI', frame)

        # CONTROLS
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_locked = not is_locked

    cap.release()
    cv2.destroyAllWindows()

    # --- FINAL REPORT ---
    print("\n" + "=" * 50)
    print("          FINAL AI REPORT")
    print("=" * 50)
    if len(session_fit_scores) > 0:
        final_average = int(sum(session_fit_scores) / len(session_fit_scores))
        print(f"Total Frames Recorded: {len(session_fit_scores)}")
        print(f"FINAL FIT SCORE:       {final_average}/100")

        if final_average > 85:
            print("RESULT: PASS (Excellent)")
        elif final_average > 70:
            print("RESULT: PASS (Good)")
        else:
            print("RESULT: FAIL (Leaks)")
    else:
        print("No data recorded. (Press SPACE to record next time)")
    print("=" * 50)


if __name__ == '__main__':
    main()