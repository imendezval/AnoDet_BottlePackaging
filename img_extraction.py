import cv2
import os

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

    # Iterate over all files in the video folder
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)

        # Check if the file is a video (basic check based on extension)
        if not video_file.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            print(f"Skipping non-video file: {video_file}")
            continue

        # Create a subfolder for frames of this video
        video_name = os.path.splitext(video_file)[0]
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

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
                frame_filename = f"fa_{extracted_count:04d}.jpg"
                frame_path = os.path.join(video_output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1

            frame_count += 1

        cap.release()
        print(f"Finished processing {video_file}. Extracted {extracted_count} frames.")

def prep_img(img_path):

    mask = cv2.imread("C:/data/git/repo/Bottle_AnoDet/imgs/bin_mask.png")
    img = cv2.imread(img_path)

    roi = [600, 210, 170, 410]
    img_cropped = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    cv2.imshow("cropped", img_cropped)
    cv2.waitKey(0)
    masked_img = cv2.bitwise_and(img_cropped, mask)
    return masked_img


if __name__ == "__main__":
    
    input_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/sampled/no_anomaly"  # Replace with your input folder path
    output_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/masked/fallen_after"  # Replace with your output folder path

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   
    
    extracted_count = 0
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing folder: {folder_name}")
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)

            extracted_count += 1
            if not image_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            masked_img = prep_img(image_path)
            img_name = f"fa_masked_{extracted_count:04d}.jpg"
            img_path_output = os.path.join(output_folder, img_name)
            cv2.imwrite(img_path_output, masked_img)
    """
    input_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/cut/no_anomaly_cut"  # Replace with your input folder path
    output_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/sampled/no_anomaly"  # Replace with your output folder path
    extract_frames(input_folder, output_folder, fps=10)
    """
    


"""
input_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/cut/fallen_after_cut"  # Replace with your input folder path
output_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/sampled/fallen_after"  # Replace with your output folder path
extract_frames(input_folder, output_folder, fps=24)
"""
