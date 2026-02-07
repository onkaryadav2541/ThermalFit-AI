import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'
LEAK_THRESHOLD_SCORE = 45

# --- INTERACTIVE VARIABLES ---
GLOBAL_OFFSET_X = 0
GLOBAL_OFFSET_Y = 20
SCALE_FACTOR = 1.0

BASE_OFFSETS = {
    "Nose Bridge": (0, -20),
    "Left Cheek": (-40, 10),
    "Right Cheek": (40, 10),
    "Left Chin": (-25, 60),
    "Right Chin": (25, 60),
    "Center Chin": (0, 70)
}

# STATE
sensor_data = {name: {'score': 0, 'history': deque(maxlen=20)} for name in BASE_OFFSETS}
last_center = None
is_setup_mode = True  # Start in "Fast/Setup" mode


def auto_calibrate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.max(gray)


def calculate_score(intensity, max_pixel_val):
    if np.isnan(intensity): return 0
    fraction = intensity / max_pixel_val
    return int(np.clip(fraction * 100, 0, 100))


def draw_dashboard(frame, sensor_data, worst_leak, worst_score, is_setup, live_fit_score):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (320, h), (15, 15, 15), -1)

    # Header
    cv2.putText(frame, "THERMAL FIT AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Status Light
    if is_setup:
        color = (0, 165, 255)  # Orange
        text = "MODE: SETUP (WASD)"
        smooth_text = "Tracking: FAST"
    else:
        color = (0, 255, 0)  # Green
        text = "MODE: LOCKED"
        smooth_text = "Tracking: HEAVY"

    cv2.circle(frame, (290, 25), 6, color, -1)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, smooth_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Fit Score
    fit_color = (0, 255, 0) if live_fit_score > 60 else (0, 0, 255)
    cv2.putText(frame, "FIT SCORE:", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"{live_fit_score}", (200, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, fit_color, 2)

    cv2.line(frame, (10, 130), (310, 130), (100, 100, 100), 1)

    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 170 + (i * 60)

        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (110, y_pos - 15), (110 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (110, y_pos - 15), (250, y_pos + 5), (50, 50, 50), 1)
        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Overall Alert
    if worst_score > LEAK_THRESHOLD_SCORE:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 0, 255), -1)
        cv2.putText(frame, "LEAK DETECTED", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 255, 0), -1)
        cv2.putText(frame, "SEAL SECURE", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Instructions Overlay (Bottom)
    cv2.putText(frame, "[SPACE] Lock/Unlock   [WASD] Adjust", (340, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)


def main():
    global last_center, GLOBAL_OFFSET_X, GLOBAL_OFFSET_Y, SCALE_FACTOR, is_setup_mode

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame = cap.read()
    if not ret: return
    global_max_pixel = auto_calibrate(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = face_mesh.process(rgb_frame)

        # --- TRACKING LOGIC ---
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            xs = [lm.x for lm in face.landmark]
            ys = [lm.y for lm in face.landmark]
            target_x = int((sum(xs) / len(xs)) * w)
            target_y = int((sum(ys) / len(ys)) * h)

            # Determine Smoothing Power
            if is_setup_mode:
                # FAST MODE: Follows face quickly so you can position dots
                smooth = 0.6
            else:
                # LOCKED MODE: Extremely heavy smoothing. NO JITTER.
                smooth = 0.05

            if last_center is None:
                last_center = (target_x, target_y)
            else:
                # Constant Smoothing (No Deadzone "Snapping")
                new_x = int((smooth * last_center[0]) + ((1 - smooth) * target_x))
                new_y = int((smooth * last_center[1]) + ((1 - smooth) * target_y))
                last_center = (new_x, new_y)

        # --- DRAW & MEASURE ---
        worst_score = 0
        total_leak = 0

        if last_center is not None:
            cx, cy = last_center

            # Show Anchor Point
            anchor_color = (0, 165, 255) if is_setup_mode else (0, 255, 0)
            cv2.circle(frame, (cx, cy), 5, anchor_color, -1)

            for name, (bx, by) in BASE_OFFSETS.items():
                final_x = cx + int(bx * SCALE_FACTOR) + GLOBAL_OFFSET_X
                final_y = cy + int(by * SCALE_FACTOR) + GLOBAL_OFFSET_Y

                final_x = np.clip(final_x, 6, w - 7)
                final_y = np.clip(final_y, 6, h - 7)

                roi = gray_frame[final_y - 6:final_y + 6, final_x - 6:final_x + 6]
                if roi.size > 0:
                    raw_score = calculate_score(np.mean(roi), global_max_pixel)
                else:
                    raw_score = 0

                sensor_data[name]['history'].append(raw_score)
                avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                sensor_data[name]['score'] = avg_score
                total_leak += avg_score

                if avg_score > worst_score:
                    worst_score = avg_score

                color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)

                # Draw Visuals
                cv2.circle(frame, (final_x, final_y), 10, color, -1)
                cv2.circle(frame, (final_x, final_y), 3, (255, 255, 255), -1)

        # Calculate Fit Score
        fit_score = 0
        if len(BASE_OFFSETS) > 0:
            avg_leak = total_leak / len(BASE_OFFSETS)
            fit_score = max(0, 100 - int(avg_leak))

        draw_dashboard(frame, sensor_data, "Leak", worst_score, is_setup_mode, fit_score)
        cv2.imshow('AI Thermal Analysis - [SPACE] to Lock', frame)

        # --- CONTROLS ---
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_setup_mode = not is_setup_mode  # TOGGLE LOCK

        # Adjust only if in SETUP mode (Prevent accidental shifts when locked)
        if is_setup_mode:
            if key == ord('w'):
                GLOBAL_OFFSET_Y -= 2
            elif key == ord('s'):
                GLOBAL_OFFSET_Y += 2
            elif key == ord('a'):
                GLOBAL_OFFSET_X -= 2
            elif key == ord('d'):
                GLOBAL_OFFSET_X += 2
            elif key == ord('z'):
                SCALE_FACTOR -= 0.05
            elif key == ord('x'):
                SCALE_FACTOR += 0.05

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()