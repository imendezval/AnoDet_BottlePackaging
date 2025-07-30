"""
Showcases how the "remove_dups" function in the util_functions/img_prep file works:
it takes and displays 2 images, computes the difference between them using the proposed
pixel-based algorithm, and displays a third image clearly showing the result of 
the algorithm (changed pixels).
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def mark_pixel_differences(grayscale1, grayscale2, diff_threshold=25, **kwargs):
    """
    Mark pixel differences above a certain threshold between two images.
    
    Parameters:
    - img1, img2: Grayscale images as numpy arrays.
    - diff_threshold: Pixel intensity difference to mark as significant.

    Returns:
    - diff_image: Image where differences above the threshold are marked.
    """
    # Compute absolute pixel differences
    diff = cv2.absdiff(grayscale1, grayscale2)
    #diff_max = np.max(diff, axis=2)
    change_fraction = np.sum(diff > diff_threshold) / diff.size
    print(change_fraction)
    
    # Create a mask for pixels exceeding the threshold
    mask = diff > diff_threshold
    
    # Create an output image to visualize differences
    diff_image = np.zeros_like(img1)  # Initialize blank image
    diff_image[mask] = 255  # Highlight pixels exceeding threshold

    return diff_image

# Load the two images
image1_path = "C:/data/git/repo/Bottle_AnoDet/imgs/masked/fallen_after/fallen_after_masked_0258.jpg"
image2_path = "C:/data/git/repo/Bottle_AnoDet/imgs/masked/fallen_after/fallen_after_masked_0259.jpg"

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Convert to grayscale for pixel-wise comparison
grayscale1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
grayscale2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Mark pixel differences
diff_image = mark_pixel_differences(grayscale1, grayscale2, diff_threshold=25)



# Display the result
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Image 1")
plt.imshow(img1, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Image 2")
plt.imshow(img2, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Pixel Differences (Threshold: 25)")
plt.imshow(diff_image, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()