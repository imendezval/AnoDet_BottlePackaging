import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def calculate_abs_distance(contour):
    # Find the leftmost and rightmost points
    leftmost_point = min(contour, key=lambda x: x[0][0])  # point with smallest x value
    rightmost_point = max(contour, key=lambda x: x[0][0])  # point with largest x value
    
    left_x = 75 - leftmost_point[0][0]
    right_x = 75 - rightmost_point[0][0]
    x_distance = min(abs(left_x-37.5), abs(right_x-37.5))

    upmost_point = min(contour, key=lambda x: x[0][1])  # point with smallest x value
    downtmost_point = max(contour, key=lambda x: x[0][1])  # point with largest x value
    
    up_y = 65 - upmost_point[0][1]
    down_y = 65 - downtmost_point[0][1]
    y_distance = min(abs(up_y-32.5), abs(down_y-32.5))

    distance = math.sqrt(y_distance**2 + x_distance**2)

    return distance



def find_area(filename):
    # Load the image
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # 1️⃣ Original Image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 5, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # 2️⃣ Use Canny Edge Detection 
    blurred = cv2.GaussianBlur(image, (7, 7), 0)  # Adjust kernel size if needed
    edges = cv2.Canny(blurred, 20, 70)

    plt.subplot(1, 5, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges Detected")
    plt.axis("off")

    # 3️⃣ Apply Otsu's Thresholding
    _, thresh = cv2.threshold(edges, 50, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Closing to fill small gaps
    kernel = np.ones((9, 9), np.uint8)  # Adjust kernel size if necessary
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    plt.subplot(1, 5, 3)
    plt.imshow(closed_thresh, cmap='gray')
    plt.title("Thresholded + Closing")
    plt.axis("off")

    # 4️⃣ Find Contours
    contours, _ = cv2.findContours(closed_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization
    if len(contours) > 0:
        cv2.drawContours(contour_image, [contours[0]], -1, (0, 255, 0), 2)  # Largest (Green)
    if len(contours) > 1:
        cv2.drawContours(contour_image, [contours[1]], -1, (0, 0, 255), 2)  # Second Largest (Blue)
    if len(contours) > 2:
        cv2.drawContours(contour_image, [contours[2]], -1, (255, 0, 0), 2)  # Third Largest (Red)

    # 5️⃣ Fit Ellipse around largest contour

    contour_values = []

    for contour in contours:
        abs_sum = calculate_abs_distance(contour)
        print(abs_sum)
        contour_values.append((contour, abs_sum))
    
    # Find the contour with the smallest value of the distance from the origin
    contour_values.sort(key=lambda x: x[1])  # Sort based on the sum
    closest_contour = contour_values[0][0] if contour_values else None


    smallest_contour_area = cv2.contourArea(closest_contour) if closest_contour is not None else 0
    print(f"Smallest Contour Area: {smallest_contour_area} pixels")

    # Compute total image area
    image_area = image.shape[0] * image.shape[1]
    print(image.shape)

    # Compute ratio of smallest contour area to total image area
    smallest_contour_ratio = smallest_contour_area / image_area if image_area > 0 else 0

    print(f"Ratio of Smallest Contour Area to Image: {smallest_contour_ratio:.6f}")


    if closest_contour is not None and len(closest_contour) >= 5:  # Need at least 5 points to fit an ellipse
        ellipse = cv2.fitEllipse(closest_contour)
        cv2.ellipse(contour_image, ellipse, (0, 255, 255), 2)  # Draw ellipse

    plt.subplot(1, 5, 4)
    plt.imshow(contour_image)
    plt.title("Contours + Ellipse")
    plt.axis("off")

    # Show all steps
    plt.tight_layout()
    plt.show()


file_name = "C:/Users/Arian/OneDrive - Hochschule Heilbronn/data_seminararbeit/wrong_lid/cropped/nd_0001.jpg"
#file_name = "C:/Users/Arian/OneDrive - Hochschule Heilbronn/data_seminararbeit/wrong_lid/cropped/nd_0035.jpg"
#file_name = "C:/Users/Arian/OneDrive - Hochschule Heilbronn/data_seminararbeit/wrong_lid/cropped/nd_0075.jpg"
#file_name = "C:/Users/Arian/OneDrive - Hochschule Heilbronn/data_seminararbeit/wrong_lid/cropped/nd_0114.jpg"

find_area(file_name)