import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd


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

    # Calculate ellipse area
    area = np.pi * a * b
    return area, b/a, center_y, angle


def find_area(filename):
    # Load the image
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Use Canny Edge Detection 
    blurred = cv2.GaussianBlur(image, (7, 7), 0)  # Adjust kernel size if needed
    edges = cv2.Canny(blurred, 20, 70)

    # Apply Otsu's Thresholding
    _, thresh = cv2.threshold(edges, 50, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Closing to fill small gaps
    kernel = np.ones((9, 9), np.uint8)  # Adjust kernel size if necessary
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

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

    image_area = image.shape[0] * image.shape[1]
    smallest_contour_area_ratio = smallest_contour_area / image_area if image_area > 0 else 0
    #print(f"Ratio of Smallest Contour Area to Image: {smallest_contour_ratio:.6f}")

    if closest_contour is not None and len(closest_contour) >= 5:  # Need at least 5 points to fit an ellipse
        ellipse = cv2.fitEllipse(closest_contour)
        cv2.ellipse(contour_image, ellipse, (0, 255, 255), 2)  # Draw ellipse
        smallest_ellipse_area, diff, centre_y, angle = calculate_ellipse_area(ellipse)
        smallest_ellipse_area_ratio = smallest_ellipse_area / image_area

        #return smallest_ellipse_area, smallest_ellipse_area_ratio, \
        #    smallest_contour_area, smallest_contour_area_ratio, diff, centre_y, angle

        print(f"centre_y {centre_y}")
        print(f"angle {angle}")
        print(f"smallest_ellipse_area_ratio {smallest_ellipse_area_ratio}")
        print(f"smallest_contour_area_ratio {smallest_contour_area_ratio}")
        print(f"diff {diff}")
        print(f"smallest_ellipse_area {smallest_ellipse_area}")
        print(f"smallest_contour_area {smallest_contour_area}\n")

"""
data = []
input_folder = "C:/data/git/repo/Bottle_AnoDet/imgs/sampled/no_anomaly/rect_crop/unique_imgs"
for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)

        ellipse_area, area_ratio, ellipse_area2, \
            area_ratio2, diff, centre_y, angle = find_area(image_path)
        
        data.append({
        "filename": filename,
        "centre_y": centre_y,
        "ellipse_area": ellipse_area,
        "area_ratio": area_ratio,
        "ellipse_area2": ellipse_area2,
        "area_ratio2": area_ratio2,
        "diff": diff,
        "angle": angle
        })

df = pd.DataFrame(data)
output_path = "C:/data/git/repo/Bottle_AnoDet/inner_circle_df.xlsx"
df.to_excel(output_path, index=False)
"""

file_name1 = "imgs/sampled/wrong_lid/crop/nd_0018.jpg"
file_name2 = "imgs/sampled/wrong_lid/crop/nd_0019.jpg"
file_name3 = "imgs/sampled/wrong_lid/crop/nd_0020.jpg"
file_name4 = "imgs/sampled/wrong_lid/crop/nd_0021.jpg"
file_name5 = "imgs/sampled/wrong_lid/crop/nd_0022.jpg"
file_name6 = "imgs/sampled/wrong_lid/crop/nd_0054.jpg"
file_name7 = "imgs/sampled/wrong_lid/crop/nd_0096.jpg"
file_name8 = "imgs/sampled/wrong_lid/crop/nd_0132.jpg"

find_area(file_name1)
find_area(file_name2)
find_area(file_name3)
find_area(file_name4)
find_area(file_name5)
find_area(file_name6)
find_area(file_name7)
find_area(file_name8)