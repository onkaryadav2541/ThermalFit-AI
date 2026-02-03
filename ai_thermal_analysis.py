import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'

# PHYSICS
T_MIN = 22.0
T_MAX = 35.0
LEAK_THRESHOLD_SCORE = 45

# --- STABILIZATION SETTINGS (THE FIX) ---
DEADZONE = 5  # Pixels. If face moves less than this, dots stay FROZEN.
VERTICAL_OFFSET = 15  # Push dots down by 15px to hit the mask, not the skin.
SMOOTH_FACTOR = 0.5  # 0.1 = Rigid, 0.9 = Loose.

# SENSOR INDICES (Standard Face Mesh)
sensor_config = {
    "Nose Bridge": 5,  # Lower nose
    "Left Cheek": 123,
    "Right Cheek": 352,
    "Left Chin": 152,  # Bottom of chin
    "Right Chin": 378
}

# STATE
sensor_data = {name: {'score': 0, 'history': deque(maxlen=10)} for name in sensor_config}
current_dots = {name: (0, 0) for name in sensor_config}  # Where we draw
last_face_center = None


def calculate_score(intensity):
    if np.isnan(intensity): return 0
    fraction = intensity / 255.0
    real_temp = T_MIN + (fraction * (T_MAX - T_MIN))
    score = ((real_temp - T_MIN) / (T_MAX - T_MIN)) * 100
    return int(np.clip(score, 0, 100))


def draw_dashboard(frame, sensor_data, worst_leak_name, worst_leak_score, is_locked):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (320, h), (10, 10, 10), -1)

    cv2.putText(frame, "AI STABILIZED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Status Light
    color = (0, 255, 0) if is_locked else (0, 165, 255)  # Green=Locked(Stable), Orange=Moving
    text = "STABLE LOCK" if is_locked else "RE-ALIGNING..."
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

    if worst_leak_score > LEAK_THRESHOLD_SCORE:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 0, 255), -1)
        cv2.putText(frame, f"LEAK: {worst_leak_name}", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 255, 0), -1)
        cv2.putText(frame, "SEAL SECURE", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    global last_face_center

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    all_session_scores = []

    print("Starting Stabilized AI...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb_frame)

        is_locked = True  # Default to locked (stable)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # 1. Calculate Face Center (Nose Tip)
            nose_tip = face.landmark[4]
            cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)

            # 2. Deadzone Logic
            should_update = False
            if last_face_center is None:
                should_update = True
            else:
                # Distance moved
                dist = np.sqrt((cx - last_face_center[0]) ** 2 + (cy - last_face_center[1]) ** 2)
                if dist > DEADZONE:
                    should_update = True
                    is_locked = False  # We are moving

            # 3. Update Positions if moved enough
            if should_update:
                last_face_center = (cx, cy)
                for name, idx in sensor_config.items():
                    lm = face.landmark[idx]
                    tx, ty = int(lm.x * w), int(lm.y * h)

                    # Apply Offset (Push down onto mask)
                    ty += VERTICAL_OFFSET

                    # Apply Smoothing
                    old_x, old_y = current_dots[name]
                    if old_x == 0:  # First frame
                        current_dots[name] = (tx, ty)
                    else:
                        nx = int((SMOOTH_FACTOR * old_x) + ((1 - SMOOTH_FACTOR) * tx))
                        ny = int((SMOOTH_FACTOR * old_y) + ((1 - SMOOTH_FACTOR) * ty))
                        current_dots[name] = (nx, ny)

        # --- DRAWING & MEASURING ---
        worst_score = 0
        worst_name = "None"
        current_frame_avg = 0

        # Check if initialized
        if current_dots["Nose Bridge"][0] != 0:
            for name, (dx, dy) in current_dots.items():

                # Safety Clamp
                dx = np.clip(dx, 6, w - 7)
                dy = np.clip(dy, 6, h - 7)

                # Measure
                roi = gray_frame[dy - 6:dy + 6, dx - 6:dx + 6]
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
                cv2.circle(frame, (dx, dy), 8, color, -1)
                cv2.circle(frame, (dx, dy), 2, (255, 255, 255), -1)  # Pinpoint

        all_session_scores.append(current_frame_avg / 5)

        draw_dashboard(frame, sensor_data, worst_name, worst_score, is_locked)

        cv2.imshow('AI Thermal Analysis - Stabilized', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_session_scores) > 0:
        final_avg = int(sum(all_session_scores) / len(all_session_scores))
        print(f"\nFINAL AI SCORE: {100 - final_avg}/100")


if __name__ == '__main__':
    main()