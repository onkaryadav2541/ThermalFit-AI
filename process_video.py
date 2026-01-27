import cv2
import os

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Videos\Captures\nutonguy.mp4'  # Put your video filename here
OUTPUT_FOLDER = 'dataset_raw'  # Where the images will go
FRAMES_PER_SECOND = 1  # How many frames to capture per second


# ---------------------

def extract_frames(video_path, output_folder, capture_rate):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cam = cv2.VideoCapture(video_path)
    original_fps = cam.get(cv2.CAP_PROP_FPS)

    hop = round(original_fps / capture_rate)

    current_frame = 0
    saved_count = 0

    print(f"Processing {video_path}...")
    print(f"Original FPS: {original_fps}. Capturing 1 frame every {hop} frames.")

    while True:
        ret, frame = cam.read()
        if ret:
            # Check if this frame is the one we want to save
            if current_frame % hop == 0:
                # Create a filename: dataset_raw/frame_0.jpg, frame_1.jpg, etc.
                name = os.path.join(output_folder, f'frame_{saved_count}.jpg')

                # OPTIONAL: Resize image to make training faster (e.g., 224x224)
                # frame = cv2.resize(frame, (224, 224))

                cv2.imwrite(name, frame)
                saved_count += 1

            current_frame += 1
        else:
            break

    cam.release()
    print(f"Done! Extracted {saved_count} images to '{output_folder}'")


# Run the function
extract_frames(VIDEO_PATH, OUTPUT_FOLDER, FRAMES_PER_SECOND)