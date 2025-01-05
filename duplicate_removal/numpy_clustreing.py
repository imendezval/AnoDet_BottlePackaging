import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import label

def classify_images_with_clusters(img1, img2, diff_threshold=25, cluster_size_threshold=50):
    """
    Classify images as duplicates or non-duplicates based on clusters of pixel changes.
    
    Parameters:
    - img1, img2: RGB images as numpy arrays.
    - diff_threshold: Threshold for pixel intensity differences.
    - cluster_size_threshold: Minimum size of pixel clusters to classify as non-duplicate.
    
    Returns:
    - is_duplicate: True if images are classified as duplicates, False otherwise.
    - diff_image: Visual representation of pixel differences with colored clusters.
    - cluster_sizes: Sizes of each cluster.
    """
    # Compute absolute differences in RGB space
    diff = np.abs(img1 - img2)  # Shape: (H, W, 3)
    
    # Calculate the maximum difference across RGB channels
    max_diff = np.max(diff, axis=2)  # Shape: (H, W)
    
    # Threshold to create a binary mask of significant changes
    mask = max_diff > diff_threshold
    
    # Label connected components in the mask
    labeled_array, num_features = label(mask)  # `labeled_array` assigns a unique ID to each cluster
    
    # Measure cluster sizes
    cluster_sizes = []
    for i in range(1, num_features + 1):  # Start from 1 to exclude the background (label 0)
        cluster_size = np.sum(labeled_array == i)
        cluster_sizes.append(cluster_size)
    
    # Create an empty image for visualizing the clusters
    diff_image = np.zeros_like(img1)  # Initialize an image with all black pixels
    
    # Color the clusters: Red for small clusters, Green for large clusters
    for i in range(1, num_features + 1):
        cluster_mask = (labeled_array == i)
        if cluster_sizes[i-1] > cluster_size_threshold:
            diff_image[cluster_mask] = [0, 255, 0]  # Color large clusters green
        else:
            diff_image[cluster_mask] = [255, 0, 0]  # Color small clusters red

    # Check if any large clusters exist
    has_large_cluster = np.any(np.array(cluster_sizes) > cluster_size_threshold)
    
    # Classify images
    is_duplicate = not has_large_cluster  # Duplicate if no large clusters found

    return is_duplicate, diff_image, cluster_sizes

# Load the two images
image1 = np.array(Image.open("C:/data/git/repo/Bottle_AnoDet/imgs/masked/fallen_after/fallen_after_masked_0258.jpg")).astype(np.int16)  # RGB
image2 = np.array(Image.open("C:/data/git/repo/Bottle_AnoDet/imgs/masked/fallen_after/fallen_after_masked_0259.jpg")).astype(np.int16)  # RGB

# Classify images based on clusters of pixel changes
is_duplicate, diff_image, cluster_sizes = classify_images_with_clusters(
    image1, image2, diff_threshold=25, cluster_size_threshold=2000
)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Image 1")
plt.imshow(image1)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Image 2")
plt.imshow(image2)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Diff Image (Duplicate: {is_duplicate})")
plt.imshow(diff_image)
plt.axis("off")

plt.tight_layout()
plt.show()

# Print all cluster sizes
print("Cluster Sizes:")
for idx, size in enumerate(cluster_sizes):
    print(f"Cluster {idx + 1}: {size} pixels")

# Print classification result
print(f"Classified as Duplicate: {is_duplicate}")
