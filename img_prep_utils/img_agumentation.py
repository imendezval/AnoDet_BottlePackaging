"""
Image Augmentation with Brightness and Contrast Adjustments.

This script was specifically created to augument the instance where the
conveyor belt is empty - since only 40 instances where found in the  
>6000 image dataset, even though it is the most common instance by far.
It works for all other images too.

This script performs the following tasks:
1. Reads images from an input folder.
2. Generates multiple augmented versions of each image by randomly adjusting brightness and contrast.
3. Reapplies the mask to the augmented images, since black region is affected
4. Saves the augmented images into an output folder.

Functions:
- `adjust_brightness_contrast(image, alpha, beta)`: Adjusts brightness and contrast of an image.

Dependencies:
- OpenCV (cv2) for image processing.
- NumPy for numerical operations.
- OS for file management.
"""

import os
import cv2
import numpy as np

# Define input and output folders
input_folder = "imgs/empty_rail"  # Change this
output_folder = "imgs/empty_rail/augumented"  # Change this

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to randomly adjust brightness and contrast
def adjust_brightness_contrast(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# List all images in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
mask = cv2.imread("imgs/bin_mask_opt.jpg")
# Loop through each image
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error loading {image_file}")
        continue
    
    # Create 7 variations
    for i in range(4):
        alpha = np.random.uniform(0.8, 1.2)  # Contrast (scaling factor)
        beta = np.random.randint(-40, 40)  # Brightness (added value)
        
        aug_img = adjust_brightness_contrast(image, alpha, beta)
        aug_img_masked = cv2.bitwise_and(aug_img, mask)
        
        # Save the augmented image
        new_filename = f"aug_{i}_{os.path.splitext(image_file)[0]}.jpg"
        cv2.imwrite(os.path.join(output_folder, new_filename), aug_img_masked)