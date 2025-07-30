"""
Video Frame Extraction, Duplicate Removal, and Image Preprocessing for Anomaly Detection

This script performs the following tasks:
1. Extracts frames from videos at a specified FPS and saves them as images.
2. Removes duplicate or nearly identical frames based on pixel change thresholds.
3. Crops and applies a mask to images for preprocessing before anomaly detection.

Functions:
- `extract_frames(video_folder, output_folder, fps)`: Extracts frames from videos.
- `remove_dups(folder_path, threshold)`: Removes duplicate frames based on pixel change.
- `crop_mask_imgs(input_folder, output_folder)`: Crops and masks images for processing.

Dependencies:
- OpenCV (cv2) for video and image manipulation.
- PIL (Pillow) for image handling.
- NumPy for numerical operations.
- shutil for file handling.
"""

import os
import cv2
import argparse
import numpy as np
from PIL import Image
import shutil

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Preprocessing Utility for Anomaly Detection")

        # Action flags
        parser.add_argument('--extract-frames', action='store_true', help="Extract frames from videos")
        parser.add_argument('--remove-dups', action='store_true', help="Remove duplicate images")
        parser.add_argument('--crop-mask', action='store_true', help="Crop and mask images")

        # Common I/O
        parser.add_argument('--video-folder', type=str, help="Folder containing videos")
        parser.add_argument('--input-folder', type=str, help="Folder of images/videos to process")
        parser.add_argument('--output-folder', type=str, help="Folder to save processed images")
        parser.add_argument('--mask-path', type=str, default="imgs/bin_mask_opt.jpg", help="Path to mask image")

        # Optional params
        parser.add_argument('--fps', type=int, default=24, help="FPS for frame extraction")
        parser.add_argument('--threshold', type=float, default=0.01, help="Threshold for duplicate detection")

        self.args = parser.parse_args()

class ImagePrepper:
    def __init__(self, config: Config):
        self.cfg = config.args

    def extract_frames(self):
        video_folder = self.cfg.video_folder
        output_folder = self.cfg.output_folder
        fps = self.cfg.fps

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_count = 0
        for video_file in os.listdir(video_folder):
            if not video_file.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                print(f"Skipping non-video file: {video_file}")
                continue

            video_path = os.path.join(video_folder, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open: {video_file}")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(round(video_fps / fps))

            print(f"Processing {video_file} at {fps} FPS...")
            frame_count = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    filename = f"frame_{image_count:04d}.jpg"
                    path = os.path.join(output_folder, filename)
                    cv2.imwrite(path, frame)
                    image_count += 1
                    extracted_count += 1
                frame_count += 1
            cap.release()
            print(f"Extracted {extracted_count} frames from {video_file}")

    def remove_dups(self):
        folder_path = self.cfg.input_folder
        threshold = self.cfg.threshold
        output_folder = os.path.join(folder_path, "unique_imgs")
        os.makedirs(output_folder, exist_ok=True)

        files = sorted(os.listdir(folder_path))
        previous_image = None
        dup_count = 0

        for filename in files:
            file_path = os.path.join(folder_path, filename)
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            org_img = Image.open(file_path)
            current_image = org_img.convert('L')
            current_array = np.array(current_image).astype(np.int16)

            if previous_image is not None:
                diff = np.abs(current_array - previous_image).astype(np.uint8)
                change_fraction = np.sum(diff > 20) / diff.size
                if change_fraction < threshold:
                    dup_count += 1
                    continue

            previous_image = current_array
            shutil.copy(file_path, os.path.join(output_folder, filename))

        print(f"Removed {dup_count} duplicate images.")

    def crop_mask_imgs(self):
        input_folder = self.cfg.input_folder
        output_folder = self.cfg.output_folder
        mask_path = self.cfg.mask_path
        roi = [525, 215, 150, 400]

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        mask = cv2.imread(mask_path)
        count = 0

        print(f"Cropping and masking images from: {input_folder}")
        for filename in os.listdir(input_folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(input_folder, filename)
            img = cv2.imread(path)
            cropped = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            masked = cv2.bitwise_and(cropped, mask)

            out_name = f"masked_{count:04d}.jpg"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, masked)
            count += 1

        print(f"Processed {count} images.")

def main():
    config = Config()
    processor = ImagePrepper(config)

    args = config.args
    if args.extract_frames:
        processor.extract_frames()
    elif args.remove_dups:
        processor.remove_dups()
    elif args.crop_mask:
        processor.crop_mask_imgs()
    else:
        print("Choose tool to be used: --extract-frames, --remove-dups, or --crop-mask")

if __name__ == "__main__":
    main()