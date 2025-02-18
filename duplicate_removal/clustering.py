import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

def detect_clusters(diff_image, method="dbscan", threshold=25, **kwargs):
    """
    Detect clusters of pixel differences in the image using different clustering methods.

    Parameters:
        diff_image (numpy.ndarray): The difference image (grayscale).
        method (str): Clustering method to use ("kmeans", "dbscan", "watershed", "hierarchical").
        threshold (int): Pixel intensity difference threshold.
        kwargs: Additional parameters for clustering methods.

    Returns:
        clustered_image (numpy.ndarray): RGB image with color-coded clusters.
    """
    h, w = diff_image.shape
    binary_diff = diff_image > threshold  # Threshold the diff image
    y, x = np.where(binary_diff)         # Coordinates of changed pixels
    data = np.column_stack((x, y))       # [x, y]

    clustered_image = np.zeros((h, w, 3), dtype=np.uint8)

    if method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 5)
        intensity = diff_image[y, x]     # Include intensity as a feature
        data_with_intensity = np.column_stack((data, intensity))

        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_with_intensity)
        labels = kmeans.labels_
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))[:, :3] * 255
        for cluster in range(n_clusters):
            clustered_image[y[labels == cluster], x[labels == cluster]] = colors[cluster]

    elif method == "dbscan":
        eps = kwargs.get("eps", 5)
        min_samples = kwargs.get("min_samples", 10)

        # DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = db.labels_
        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))[:, :3] * 255
        for cluster, color in zip(unique_labels, colors):
            if cluster != -1:  # Ignore noise
                clustered_image[y[labels == cluster], x[labels == cluster]] = color

    elif method == "watershed":
        # Create markers for Watershed
        _, binary = cv2.threshold(diff_image, threshold, 255, cv2.THRESH_BINARY)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, markers = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
        markers = np.uint8(markers)
        markers = cv2.connectedComponents(markers)[1]
        markers = cv2.watershed(cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR), markers)

        # Color the clusters
        unique_labels = np.unique(markers)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))[:, :3] * 255
        for label, color in zip(unique_labels, colors):
            if label > 0:  # Ignore background
                clustered_image[markers == label] = color

    elif method == "hierarchical":
        distance_threshold = kwargs.get("distance_threshold", 10)

        # Hierarchical clustering
        Z = linkage(data, method='ward')  # Ward's method for clustering
        labels = fcluster(Z, t=distance_threshold, criterion='distance')
        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))[:, :3] * 255
        for cluster, color in zip(unique_labels, colors):
            clustered_image[y[labels == cluster], x[labels == cluster]] = color

    return clustered_image


# Visualize results for all methods
def visualize_clustering(diff_image, threshold=25):
    methods = ["kmeans", "dbscan", "watershed", "hierarchical"]
    plt.figure(figsize=(16, 10))

    for i, method in enumerate(methods, 1):
        clustered_image = detect_clusters(diff_image, method=method, threshold=threshold, n_clusters=3, eps=5, min_samples=5, distance_threshold=10)
        plt.subplot(1, len(methods), i)
        plt.imshow(clustered_image)
        plt.title(method.capitalize())
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Example Usage
def process_images(image1_path, image2_path, diff_threshold=25):
    # Load the two images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Convert to grayscale for pixel-wise comparison
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Visualize clustering
    visualize_clustering(diff, threshold=diff_threshold)


# Paths to example images
image1_path = "C:/data/git/repo/Bottle_AnoDet/imgs/masked/fallen_after/fallen_after_masked_0258.jpg"  # Replace with the correct path
image2_path = "C:/data/git/repo/Bottle_AnoDet/imgs/masked/fallen_after/fallen_after_masked_0259.jpg"  # Replace with the correct path

# Process and visualize
process_images(image1_path, image2_path, diff_threshold=25)
