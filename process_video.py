import cv2
import mediapipe as mp
import numpy as np
import os
import shutil

# --- CONFIGURATION ---
# Use 'r' before the string to handle backslashes and spaces correctly
VIDEO_FOLDER = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings'
OUTPUT_DIR = 'dataset'
FRAMES_PER_SECOND = 2  # Extract 2 images per second

# PHYSICS THRESHOLDS
LEAK_THRESHOLD = 45
T_MIN = 22.0
T_MAX = 35.0


def calculate_score(intensity):
    fraction = intensity / 255.0
    real_temp = T_MIN + (fraction * (T_MAX - T_MIN))
    score = ((real_temp - T_MIN) / (T_MAX - T_MIN)) * 100
    return int(np.clip(score, 0, 100))


def process_batch():
    # 1. Setup Folders (Clear old data to start fresh)
    if os.path.exists(OUTPUT_DIR):
        print("Cleaning old dataset...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(f"{OUTPUT_DIR}/leak")
    os.makedirs(f"{OUTPUT_DIR}/no_leak")

    # 2. Setup AI
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    # 3. Get list of video files
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print(f"ERROR: No videos found in {VIDEO_FOLDER}")
        return

    print(f"Found {len(video_files)} videos. Starting Batch Processing...")
    print("-" * 50)

    total_saved = 0

    # 4. Loop through every video
    for vid_index, filename in enumerate(video_files):
        video_path = os.path.join(VIDEO_FOLDER, filename)
        print(f"[{vid_index + 1}/{len(video_files)}] Processing: {filename}...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30

        frame_interval = int(fps / FRAMES_PER_SECOND)
        count = 0
        vid_saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            if count % frame_interval == 0:
                # Resize for consistency (optional, but good for training)
                # frame = cv2.resize(frame, (640, 480))

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    h, w, _ = frame.shape
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Check Nose(6) and Cheeks(123, 352)
                    check_points = [6, 123, 352]
                    total_score = 0

                    for idx in check_points:
                        lm = landmarks.landmark[idx]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cx, cy = np.clip(cx, 2, w - 3), np.clip(cy, 2, h - 3)

                        roi = gray[cy - 2:cy + 3, cx - 2:cx + 3]
                        if roi.size > 0:
                            total_score += calculate_score(np.mean(roi))

                    avg_score = total_score / len(check_points)

                    # Sort
                    if avg_score > LEAK_THRESHOLD:
                        label = "leak"
                    else:
                        label = "no_leak"

                    # Save with unique name: video_name + frame_number
                    safe_vid_name = os.path.splitext(filename)[0].replace(" ", "_")
                    save_name = f"{OUTPUT_DIR}/{label}/{safe_vid_name}_frame_{vid_saved_count}.jpg"
                    cv2.imwrite(save_name, frame)
                    vid_saved_count += 1
                    total_saved += 1

            count += 1

        cap.release()
        print(f"   -> Extracted {vid_saved_count} frames.")

    print("-" * 50)
    print(f"BATCH COMPLETE! Total images in dataset: {total_saved}")
    print(f"You can now run 'python train_model.py'")


if __name__ == "__main__":
    process_batch()