import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'
LEAK_THRESHOLD_SCORE = 45

# --- ADJUSTMENT VARIABLES (WASD) ---
# These allow you to nudge the dots if the AI is slightly off
offset_x = 0
offset_y = 0
scale_factor = 1.0

# --- LANDMARK DEFINITIONS ---
# These are the "Anchors" on the face (Nose, Cheeks, Chin)
LANDMARK_IDS = {
    "Nose Bridge": 168,  # Between eyes (Very stable)
    "Left Cheek": 118,  # Cheekbone
    "Right Cheek": 347,  # Cheekbone
    "Left Chin": 58,  # Jawline
    "Right Chin": 288,  # Jawline
    "Center Chin": 152  # Bottom of chin
}

# STATE
sensor_data = {name: {'score': 0, 'history': deque(maxlen=20)} for name in LANDMARK_IDS}
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

    cv2.putText(frame, "HYBRID THERMAL AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if not is_locked:
        color = (0, 165, 255)
        text = "MODE: ADJUST (WASD)"
        sub = "Move dots to seal line"
    else:
        color = (0, 255, 0)
        text = "MODE: LOCKED"
        sub = "Recording Data..."

    cv2.circle(frame, (290, 25), 6, color, -1)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, sub, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Fit Score
    fit_color = (0, 255, 0) if live_fit_score > 60 else (0, 0, 255)
    cv2.putText(frame, "LIVE FIT SCORE:", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"{live_fit_score}", (200, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, fit_color, 2)
    cv2.line(frame, (10, 140), (310, 140), (100, 100, 100), 1)

    # Sensor Bars
    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 180 + (i * 60)
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

    # Draw Controls Info
    cv2.putText(frame, f"Offset X:{offset_x} Y:{offset_y}", (340, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)


def main():
    global is_locked, offset_x, offset_y, scale_factor

    # 1. SETUP MEDIA PIPE (Must use Python 3.10!)
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    except AttributeError:
        print("CRITICAL ERROR: MediaPipe not installed properly or using Python 3.13.")
        print("Please switch to Python 3.10.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret: return
    global_max_pixel = auto_calibrate(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    print("System Loaded. Use WASD to nudge the dots.")

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

            # Draw "Spiderweb" center for visual reference
            nose_lm = face.landmark[1]  # Tip of nose
            center_x, center_y = int(nose_lm.x * w), int(nose_lm.y * h)

            # Loop through all 6 points
            for name, landmark_id in LANDMARK_IDS.items():
                lm = face.landmark[landmark_id]

                # 1. Base Position (Sticks to head)
                base_x = int(lm.x * w)
                base_y = int(lm.y * h)

                # 2. Manual Adjustment (WASD nudge)
                # We apply the offset relative to the face center to keep scaling correct
                final_x = base_x + offset_x
                final_y = base_y + offset_y

                # Keep dot inside screen
                final_x = np.clip(final_x, 6, w - 7)
                final_y = np.clip(final_y, 6, h - 7)

                # 3. MEASURE HEAT
                roi = gray_frame[final_y - 6:final_y + 6, final_x - 6:final_x + 6]
                if roi.size > 0:
                    raw_score = calculate_score(np.mean(roi), global_max_pixel)
                else:
                    raw_score = 0

                # Update Data
                sensor_data[name]['history'].append(raw_score)
                avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                sensor_data[name]['score'] = avg_score
                total_leak_score += avg_score
                valid_sensors += 1

                if avg_score > worst_score: worst_score = avg_score

                # 4. DRAW VISUALS
                color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)

                # Draw Line to Center (Visual Spiderweb)
                cv2.line(frame, (center_x, center_y), (final_x, final_y), (50, 50, 50), 1)

                # Draw Dot
                cv2.circle(frame, (final_x, final_y), 8, color, -1)
                cv2.circle(frame, (final_x, final_y), 2, (255, 255, 255), -1)

        # CALCULATE LIVE SCORE
        live_fit_score = 0
        if valid_sensors > 0:
            avg_heat = total_leak_score / valid_sensors
            live_fit_score = max(0, 100 - int(avg_heat))

            if is_locked:
                session_fit_scores.append(live_fit_score)

        draw_dashboard(frame, sensor_data, worst_score, is_locked, live_fit_score)
        cv2.imshow('Hybrid Sticky AI', frame)

        # CONTROLS
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_locked = not is_locked

        # Manual Nudge Controls
        if not is_locked:
            if key == ord('w'):
                offset_y -= 2
            elif key == ord('s'):
                offset_y += 2
            elif key == ord('a'):
                offset_x -= 2
            elif key == ord('d'):
                offset_x += 2

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
    else:
        print("No data recorded.")
    print("=" * 50)


if __name__ == '__main__':
    main()