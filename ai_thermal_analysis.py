import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'
LEAK_THRESHOLD_SCORE = 45

# --- ANTI-DRIFT SETTINGS ---
# 1. Sticky Lock: Face must move > 15 pixels to break the lock.
#    This prevents "slow drifting" while sitting still.
DEADZONE = 15

# 2. Heavy Weight: When it does move, it moves like a tank (slowly).
SMOOTH_FACTOR = 0.05

# --- INTERACTIVE VARIABLES ---
GLOBAL_OFFSET_X = 0
GLOBAL_OFFSET_Y = 20
SCALE_FACTOR = 1.0

# Base positions relative to Face Center
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


def auto_calibrate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.max(gray)


def calculate_score(intensity, max_pixel_val):
    if np.isnan(intensity): return 0
    fraction = intensity / max_pixel_val
    return int(np.clip(fraction * 100, 0, 100))


def draw_dashboard(frame, sensor_data, worst_leak, worst_score, is_locked):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (320, h), (15, 15, 15), -1)

    # Header
    cv2.putText(frame, "ANTI-DRIFT SYSTEM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Status Light
    if is_locked:
        color = (0, 255, 0)  # Green
        text = "STATUS: GLUED"
    else:
        color = (0, 165, 255)  # Orange
        text = "STATUS: MOVING..."

    cv2.circle(frame, (290, 25), 6, color, -1)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Controls
    cv2.putText(frame, "[WASD] Move   [ZX] Scale", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.line(frame, (10, 95), (310, 95), (100, 100, 100), 1)

    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 140 + (i * 60)

        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        bar_len = int(score * 1.5)
        cv2.rectangle(frame, (110, y_pos - 15), (110 + bar_len, y_pos + 5), color, -1)
        cv2.rectangle(frame, (110, y_pos - 15), (250, y_pos + 5), (50, 50, 50), 1)
        cv2.putText(frame, f"{score}", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if worst_score > LEAK_THRESHOLD_SCORE:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 0, 255), -1)
        cv2.putText(frame, f"LEAK: {worst_leak}", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (10, h - 80), (310, h - 20), (0, 255, 0), -1)
        cv2.putText(frame, "SEAL SECURE", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    global last_center, GLOBAL_OFFSET_X, GLOBAL_OFFSET_Y, SCALE_FACTOR

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

        is_locked = True  # Default assumption: We are glued in place

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # Use Center of Gravity (Average of all points)
            xs = [lm.x for lm in face.landmark]
            ys = [lm.y for lm in face.landmark]
            avg_x, avg_y = int((sum(xs) / len(xs)) * w), int((sum(ys) / len(ys)) * h)

            if last_center is None:
                last_center = (avg_x, avg_y)
            else:
                # --- DRIFT PROTECTION LOGIC ---
                dist = np.sqrt((avg_x - last_center[0]) ** 2 + (avg_y - last_center[1]) ** 2)

                # If movement is < 15 pixels, we IGNORE IT completely.
                # The dots stay "Glued" to the old position.
                if dist > DEADZONE:
                    is_locked = False
                    # Only if we move A LOT, do we slowly transition
                    new_x = int((SMOOTH_FACTOR * last_center[0]) + ((1 - SMOOTH_FACTOR) * avg_x))
                    new_y = int((SMOOTH_FACTOR * last_center[1]) + ((1 - SMOOTH_FACTOR) * avg_y))
                    last_center = (new_x, new_y)

                # Else: last_center remains UNCHANGED.
                # This prevents the "slow creep" or "shifting by itself".

        # --- DRAW DOTS ---
        worst_score = 0
        worst_name = "None"

        if last_center is not None:
            cx, cy = last_center
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Anchor

            for name, (bx, by) in BASE_OFFSETS.items():
                # Apply Offsets
                final_x = cx + int(bx * SCALE_FACTOR) + GLOBAL_OFFSET_X
                final_y = cy + int(by * SCALE_FACTOR) + GLOBAL_OFFSET_Y

                final_x = np.clip(final_x, 6, w - 7)
                final_y = np.clip(final_y, 6, h - 7)

                # Measure
                roi = gray_frame[final_y - 6:final_y + 6, final_x - 6:final_x + 6]
                if roi.size > 0:
                    raw_score = calculate_score(np.mean(roi), global_max_pixel)
                else:
                    raw_score = 0

                sensor_data[name]['history'].append(raw_score)
                avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                sensor_data[name]['score'] = avg_score
                if avg_score > worst_score:
                    worst_score = avg_score
                    worst_name = name

                # Draw
                color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
                cv2.circle(frame, (final_x, final_y), 10, color, -1)
                cv2.circle(frame, (final_x, final_y), 3, (255, 255, 255), -1)

        draw_dashboard(frame, sensor_data, worst_name, worst_score, is_locked)
        cv2.imshow('AI Anti-Drift - WASD to Adjust', frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
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