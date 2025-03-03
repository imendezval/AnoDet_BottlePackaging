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


def calculate_ellipse_area(ellipse):
    """
    Calculate the area of an ellipse fitted to a given contour.
    
    :param contour: Contour points (numpy array of shape Nx1x2).
    :return: Area of the fitted ellipse.
    """


    (center_x, center_y), (major_axis, minor_axis), angle = ellipse

    # Semi-major and semi-minor axes
    a = major_axis / 2
    b = minor_axis / 2
    #print(f"A:{a}")
    #print(f"B:{b}")

    # Calculate ellipse area
    area = np.pi * a * b
    return area, b/a


def find_area(filename):
    # Load the image
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Original Image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 5, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # Use Canny Edge Detection 
    blurred = cv2.GaussianBlur(image, (7, 7), 0)  # Adjust kernel size if needed
    edges = cv2.Canny(blurred, 20, 70)

    plt.subplot(1, 5, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges Detected")
    plt.axis("off")

    # Apply Otsu's Thresholding
    _, thresh = cv2.threshold(edges, 50, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Closing to fill small gaps
    kernel = np.ones((9, 9), np.uint8)  # Adjust kernel size if necessary
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    plt.subplot(1, 5, 3)
    plt.imshow(closed_thresh, cmap='gray')
    plt.title("Thresholded + Closing")
    plt.axis("off")

    # Find Contours
    contours, _ = cv2.findContours(closed_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization
    if len(contours) > 0:
        cv2.drawContours(contour_image, [contours[0]], -1, (0, 255, 0), 2)  # Largest (Green)
    if len(contours) > 1:
        cv2.drawContours(contour_image, [contours[1]], -1, (0, 0, 255), 2)  # Second Largest (Blue)
    if len(contours) > 2:
        cv2.drawContours(contour_image, [contours[2]], -1, (255, 0, 0), 2)  # Third Largest (Red)

    # Fit Ellipse around largest contour

    contour_values = []

    for contour in contours:
        abs_sum = calculate_abs_distance(contour)
        #print(abs_sum)
        contour_values.append((contour, abs_sum))
    
    # Find the contour with the smallest value of the distance from the origin
    contour_values.sort(key=lambda x: x[1])  # Sort based on the sum
    closest_contour = contour_values[0][0] if contour_values else None


    smallest_contour_area = cv2.contourArea(closest_contour) if closest_contour is not None else 0
    #print(f"Smallest Contour Area: {smallest_contour_area} pixels")

    # Compute total image area
    image_area = image.shape[0] * image.shape[1]
    #print(image.shape)

    # Compute ratio of smallest contour area to total image area
    smallest_contour_ratio = smallest_contour_area / image_area if image_area > 0 else 0

    #print(f"Ratio of Smallest Contour Area to Image: {smallest_contour_ratio:.6f}")


    if closest_contour is not None and len(closest_contour) >= 5:  # Need at least 5 points to fit an ellipse
        ellipse = cv2.fitEllipse(closest_contour)
        cv2.ellipse(contour_image, ellipse, (0, 255, 255), 2)  # Draw ellipse
        #print(ellipse)
        smallest_ellipse_area, diff = calculate_ellipse_area(ellipse)
        print(f"Smallest Contour Area: {smallest_ellipse_area} pixels")
        smallest_ellipse_ratio = smallest_ellipse_area / image_area if image_area > 0 else 0
        print(f"Ratio Area to Image: {smallest_ellipse_ratio:.6f}")
        print(f"Diff B/A {diff} \n")
        
    
    plt.subplot(1, 5, 4)
    plt.imshow(contour_image)
    plt.title("Contours + Ellipse")
    plt.axis("off")

    # Show all steps
    plt.tight_layout()
    plt.show()


well_placed = "imgs/sampled/wrong_lid/crop/nd_0018.jpg"

miss_1 = "imgs/sampled/wrong_lid/crop/nd_0054.jpg"
miss_2 = "imgs/sampled/wrong_lid/crop/nd_0096.jpg"
miss_3 = "imgs/sampled/wrong_lid/crop/nd_0132.jpg"

find_area(well_placed)