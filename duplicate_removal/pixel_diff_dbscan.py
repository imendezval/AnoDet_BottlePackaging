import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import cv2

def mark_pixel_differences(img1, img2, diff_threshold=25, **kwargs):
    """
    Mark pixel differences above a certain threshold between two images.
    
    Parameters:
    - img1, img2: Grayscale images as numpy arrays.
    - diff_threshold: Pixel intensity difference to mark as significant.

    Returns:
    - diff_image: Image where differences above the threshold are marked.
    """
    # Compute absolute pixel differences
    diff = np.abs(img1 - img2)
    #diff_max = np.max(diff, axis=2)
    change_fraction = np.sum(diff > diff_threshold) / diff.size
    print(change_fraction)
    
    # Create a mask for pixels exceeding the threshold
    mask = diff > diff_threshold
    
    # Create an output image to visualize differences
    diff_image = np.zeros_like(img1)  # Initialize blank image
    diff_image[mask] = 255  # Highlight pixels exceeding threshold

    diff = cv2.absdiff(img1, img2)
    diff_binary = diff > diff_threshold
    y, x = np.where(diff_binary)
    data = np.column_stack((x, y))
    h, w = diff.shape
    clustered_image = np.zeros((h, w, 3), dtype=np.uint8)
    if data.size != 0:
        eps = kwargs.get("eps", 5)
        min_samples = kwargs.get("min_samples", 10)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = db.labels_
        unique_labels = set(labels) 
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))[:, :3] * 255
        for cluster, color in zip(unique_labels, colors):
            if cluster != -1:  # Ignore noise
                clustered_image[y[labels == cluster], x[labels == cluster]] = color

    return diff_image, clustered_image

# Load the two images
image1 = np.array(Image.open("C:/data/git/repo/Bottle_AnoDet/imgs/masked/fallen_after/fallen_after_masked_0258.jpg").convert('L'))  # Grayscale
image2 = np.array(Image.open("C:/data/git/repo/Bottle_AnoDet/imgs/masked/fallen_after/fallen_after_masked_0259.jpg").convert('L'))  # Grayscale


"""
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Convert to grayscale for pixel-wise comparison
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Compute the absolute difference
diff = cv2.absdiff(gray1, gray2)
"""
# Mark pixel differences
diff_image, clustered_image = mark_pixel_differences(image1, image2, diff_threshold=25)



# Display the result
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.title("Image 1")
plt.imshow(image1, cmap='gray')
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Image 2")
plt.imshow(image2, cmap='gray')
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Pixel Differences (Threshold: 25)")
plt.imshow(diff_image, cmap='gray')
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("DBScan Clustering")
plt.imshow(clustered_image)
plt.axis("off")

plt.tight_layout()
plt.show()