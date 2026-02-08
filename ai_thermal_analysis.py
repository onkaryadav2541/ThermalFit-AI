import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'
LEAK_THRESHOLD_SCORE = 45

# --- MAPPING DOTS TO FACE MESH INDICES ---
# These indices correspond to the MediaPipe Face Mesh map
LANDMARK_MAP = {
    "Nose Bridge": 6,
    "Left Cheek": 234,
    "Right Cheek": 454,
    "Left Chin": 132,
    "Right Chin": 361,
    "Center Chin": 152
}

# INTERACTIVE ADJUSTMENTS
GLOBAL_OFFSET_X = 0
GLOBAL_OFFSET_Y = 0
is_setup_mode = True

sensor_data = {name: {'score': 0, 'history': deque(maxlen=20)} for name in LANDMARK_MAP}


def calculate_score(intensity, max_pixel_val):
    if np.isnan(intensity) or max_pixel_val == 0: return 0
    fraction = intensity / max_pixel_val
    return int(np.clip(fraction * 100, 0, 100))


def draw_dashboard(frame, sensor_data, worst_score, is_setup, live_fit_score):
    h, w, _ = frame.shape
    # Side Panel
    cv2.rectangle(frame, (0, 0), (320, h), (15, 15, 15), -1)
    cv2.putText(frame, "THERMAL FIT AI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    status_color = (0, 165, 255) if is_setup else (0, 255, 0)
    status_text = "MODE: ADJUSTING" if is_setup else "MODE: LOCKED"
    cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Fit Score
    cv2.putText(frame, f"FIT SCORE: {live_fit_score}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, (name, data) in enumerate(sensor_data.items()):
        score = data['score']
        color = (0, 0, 255) if score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        y_pos = 170 + (i * 60)
        cv2.putText(frame, name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(frame, (110, y_pos - 15), (110 + int(score * 1.5), y_pos + 5), color, -1)


def main():
    global GLOBAL_OFFSET_X, GLOBAL_OFFSET_Y, is_setup_mode

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret: return

    # Auto-calibration for "hot" spots
    global_max_pixel = np.max(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = face_mesh.process(rgb_frame)

        worst_score = 0
        total_leak = 0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            for name, index in LANDMARK_MAP.items():
                # Get the actual landmark coordinate
                lm = face_landmarks.landmark[index]

                # Convert normalized to pixel coordinates
                # Adding Global Offsets for fine-tuning the mask seal location
                px_x = int(lm.x * w) + GLOBAL_OFFSET_X
                px_y = int(lm.y * h) + GLOBAL_OFFSET_Y

                px_x = np.clip(px_x, 10, w - 10)
                px_y = np.clip(px_y, 10, h - 10)

                # Sample Thermal Data (ROI)
                roi = gray_frame[px_y - 5:px_y + 5, px_x - 5:px_x + 5]
                raw_score = calculate_score(np.mean(roi), global_max_pixel) if roi.size > 0 else 0

                # Smoothing the score
                sensor_data[name]['history'].append(raw_score)
                avg_score = int(sum(sensor_data[name]['history']) / len(sensor_data[name]['history']))
                sensor_data[name]['score'] = avg_score
                total_leak += avg_score
                worst_score = max(worst_score, avg_score)

                # Draw point on face
                dot_color = (0, 0, 255) if avg_score > LEAK_THRESHOLD_SCORE else (0, 255, 0)
                cv2.circle(frame, (px_x, px_y), 6, dot_color, -1)
                cv2.circle(frame, (px_x, px_y), 2, (255, 255, 255), -1)

        # Calculate Final Stats
        fit_score = max(0, 100 - (total_leak // len(LANDMARK_MAP)))
        draw_dashboard(frame, sensor_data, worst_score, is_setup_mode, fit_score)

        cv2.imshow('Automatic Thermal Mask Fit', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_setup_mode = not is_setup_mode

        if is_setup_mode:
            if key == ord('w'):
                GLOBAL_OFFSET_Y -= 2
            elif key == ord('s'):
                GLOBAL_OFFSET_Y += 2
            elif key == ord('a'):
                GLOBAL_OFFSET_X -= 2
            elif key == ord('d'):
                GLOBAL_OFFSET_X += 2

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()