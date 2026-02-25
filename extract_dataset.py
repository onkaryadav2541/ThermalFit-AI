import cv2
import os
import glob

# --- CONFIGURATION ---
# Put the path to the folder containing ALL your videos here:
INPUT_FOLDER_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings'

# Where you want the pictures saved:
OUTPUT_FOLDER_PATH = r'C:\Users\Onkar\PycharmProjects\Quantative_testing_method_for_respiratory_mask\dataset_frames'

# Extract 1 frame every 15 frames (Extracts 2 pictures per second for a 30fps video)
FRAME_SKIP = 15


def extract_all_videos():
    # 1. Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)
        print(f"Created output folder: {OUTPUT_FOLDER_PATH}")

    # 2. Find all .mp4 videos in the input folder
    search_path = os.path.join(INPUT_FOLDER_PATH, '*.mp4')
    video_files = glob.glob(search_path)

    if not video_files:
        print(f"Error: No .mp4 files found in {INPUT_FOLDER_PATH}")
        return

    print(f"Found {len(video_files)} videos. Starting extraction...")
    total_saved = 0

    # 3. Loop through every video one by one
    for video_path in video_files:
        # Get just the name of the video without the folder path or .mp4
        video_name = os.path.basename(video_path).replace('.mp4', '')

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        video_saved_count = 0

        print(f"Processing: {video_name}...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Video is finished

            # Only save every Nth frame
            if frame_count % FRAME_SKIP == 0:
                # Name it clearly: VideoName_frame_00015.jpg
                filename = f"{video_name}_frame_{frame_count:05d}.jpg"
                filepath = os.path.join(OUTPUT_FOLDER_PATH, filename)

                cv2.imwrite(filepath, frame)
                video_saved_count += 1
                total_saved += 1

            frame_count += 1

        cap.release()
        print(f"  -> Extracted {video_saved_count} frames from {video_name}")

    print("\n" + "=" * 40)
    print("EXTRACTION COMPLETE")
    print(f"Total frames saved across all videos: {total_saved}")
    print(f"Saved to: {OUTPUT_FOLDER_PATH}")
    print("=" * 40)


if __name__ == '__main__':
    extract_all_videos()