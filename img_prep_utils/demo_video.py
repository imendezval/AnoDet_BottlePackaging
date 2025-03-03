"""
Demo video creator from recorded footage using trained EfficientNetV2-S model

This script takes a recorded video from the camera, and outputs a demo of the 
live classification using a trained EfficientNetV2-S model, clearly showcasing 
how anomalies are classified, and how the cropping and masking are applied.
"""
import cv2

# Input and output video files
input_video = "recorded_video.mp4"
output_video = "output_edit.mp4"

# Parameters
roi = [525, 215, 150, 400]
box_start_time = 5
mask_start_time = 15
mask_end_time = 29
fade_duration = 2
box_color = (255, 0, 255)  # BGR
box_thickness = 2
fps = 30

# Load the binary mask
mask = cv2.imread("imgs/bin_mask_opt.jpg")

mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
_, mask_binary = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY_INV)

# Open the input video
cap = cv2.VideoCapture(input_video)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or fps

# Calculate frame numbers for effects
box_start_frame = int(box_start_time * video_fps)
mask_start_frame = int(mask_start_time * video_fps)
mask_end_frame = int(mask_end_time * video_fps)
fade_frames = int(fade_duration * video_fps)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, video_fps, (frame_width, frame_height))

# Process each frame
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Draw the rectangle (always visible after fade-in)
    if frame_idx >= box_start_frame:
        start_point = (roi[0], roi[1])  # Top-left corner of the ROI
        end_point = (roi[0] + roi[2], roi[1] + roi[3])  # Bottom-right corner
        frame = cv2.rectangle(frame, start_point, end_point, box_color, box_thickness)

    # Fade-in and fade-out effect for the mask
    if mask_start_frame <= frame_idx < mask_start_frame + fade_frames:
        # Fade-in progress
        alpha = (frame_idx - mask_start_frame) / fade_frames
    elif mask_start_frame + fade_frames <= frame_idx < mask_end_frame - fade_frames:
        # Mask fully visible
        alpha = 1
    elif mask_end_frame - fade_frames <= frame_idx < mask_end_frame:
        # Fade-out progress
        alpha = (mask_end_frame - frame_idx) / fade_frames
    else:
        alpha = 0

    # Apply the mask if alpha > 0
    if alpha > 0:
        x, y, w, h = roi
        roi_area = frame[y:y + h, x:x + w]
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)

        # Blend the mask into the ROI with transparency
        mask_region = cv2.addWeighted(roi_area, 1 - alpha, mask, alpha, 0)
        roi_area[mask_binary == 0] = mask_region[mask_binary == 0]
        frame[y:y + h, x:x + w] = roi_area

    # Write the frame to the output video
    out.write(frame)
    frame_idx += 1

# Release resources
cap.release()
out.release()
print("Processing complete. Output saved as", output_video)
