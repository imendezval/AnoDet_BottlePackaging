import cv2
import os
from PIL import Image
import numpy as np
import shutil

def extract_frames(video_folder, output_folder, fps=24):
    """
    Extracts frames from all videos in the given folder at the specified FPS
    and saves them into the output folder.

    :param video_folder: Path to the folder containing video files.
    :param output_folder: Path to the folder to save extracted frames.
    :param fps: Number of frames per second to extract.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_count = 0
    # Iterate over all files in the video folder
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)

        # Check if the file is a video (basic check based on extension)
        if not video_file.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            print(f"Skipping non-video file: {video_file}")
            continue

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_file}")
            continue

        # Get the video frame rate
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(video_fps / fps))

        print(f"Processing {video_file}: Extracting frames at {fps} FPS")

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame if it matches the desired interval
            if frame_count % frame_interval == 0:
                frame_filename = f"nd_{image_count:04d}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1
                image_count += 1 

            frame_count += 1

        cap.release()
        print(f"Finished processing {video_file}. Extracted {extracted_count} frames.")


def remove_dups(folder_path, threshold=0.01):
    non_dupe_folder = os.path.join(folder_path, "unique_imgs")
    os.makedirs(non_dupe_folder, exist_ok=True)

    files = sorted(os.listdir(folder_path))

    previous_image = None
    dup_num = 0

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        output_path = os.path.join(non_dupe_folder, filename)
        if os.path.isfile(file_path):
            org_img = Image.open(file_path)
            current_image = org_img.convert('L')
            current_array = np.array(current_image).astype(np.int16) #type int necessary for diff calculation

            if previous_image is not None:
                diff = np.abs(current_array - previous_image).astype(np.uint8)
                
                change_fraction = np.sum(diff > 20) / diff.size
                
                if change_fraction < threshold:
                    #print(f"Removing still frame: {file_path}") # No remove - if few pixels change a lot
                    #os.remove(file_path)           # if not enough pixels change a lot, remove
                    dup_num += 1
                    continue
            org_img.save(output_path)
            previous_image = current_array
            shutil.copy(file_path, non_dupe_folder)

    print(dup_num)


def crop_mask_imgs(input_folder, output_folder):

    mask = cv2.imread("C:/data/git/repo/Bottle_AnoDet/imgs/bin_mask_opt.jpg")
    roi = [525, 215, 150, 400]

    if not os.path.exists(output_folder):
            os.makedirs(output_folder)   
        
    extracted_count = 0

    print(f"Processing folder: {input_folder}")
    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)

        extracted_count += 1
        if not image_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img = cv2.imread(image_path)
        img_cropped = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        masked_img = cv2.bitwise_and(img_cropped, mask)

        img_name = f"fallen_before_{extracted_count:04d}.jpg"
        img_path_output = os.path.join(output_folder, img_name)
        cv2.imwrite(img_path_output, masked_img)

    return masked_img


"""
input_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/eval"  # Replace with your input folder path
output_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/eval/masked"  # Replace with your output folder path
crop_mask_imgs(input_folder,output_folder)

input_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/cut/no_deckel"  # Replace with your input folder path
output_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/sampled/no_deckel"  # Replace with your output folder path
extract_frames(input_folder, output_folder, fps=10)
"""

input_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/sampled/no_anomaly/rect_crop"
remove_dups(input_folder)