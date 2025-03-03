import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def pixel_above_thresh(img, grauwerte_threshold=25):

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a mask for pixels exceeding the threshold
    mask = grayscale > grauwerte_threshold
    
    # Create an output image to visualize differences
    diff_image = np.zeros_like(img)  # Initialize blank image
    diff_image[mask] = 255  # Highlight pixels exceeding threshold

    masked_img = cv2.bitwise_and(img, diff_image)

    return diff_image, masked_img

# Load the two images
image1_path = "imgs/sampled/wrong_lid/crop/nd_0021.jpg"
img = cv2.imread(image1_path)
diff_image, masked_img = pixel_above_thresh(img, grauwerte_threshold=150)





# Display the result
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title("Image 1")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Pixel Differences (Threshold: 215)")
plt.imshow(diff_image, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Masked Image")
plt.imshow(masked_img, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()