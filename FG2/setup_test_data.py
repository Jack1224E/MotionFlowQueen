import cv2
import argparse
import os
import sys

def extract_frames(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        sys.exit(1)

    output_dir = "input_frames"
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    # Extract 3 consecutive frames
    for i in range(3):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {i+1} from video.")
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 3:
        cv2.imwrite(os.path.join(output_dir, "frame1.png"), frames[0])
        cv2.imwrite(os.path.join(output_dir, "frame2.png"), frames[1]) # Middle frame (ground truth)
        cv2.imwrite(os.path.join(output_dir, "frame3.png"), frames[2])
        print(f"Successfully extracted frames to '{output_dir}/'.")
    else:
        print("Failed to extract 3 consecutive frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3 consecutive frames from a video.")
    parser.add_argument("video_path", help="Path to the input video file.")
    args = parser.parse_args()
    
    extract_frames(args.video_path)
