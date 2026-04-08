# Author: Rui Gu
# This script processes heatmap images to detect and highlight red and green regions.
# Dependencies:
# - OpenCV: pip install opencv-python
# - NumPy: pip install numpy
# Ensure these dependencies are installed before running the script.

import cv2
import numpy as np
import os

# Path to the heatmap images
heatmap_image_path = "Enter the path to the heatmap images: "    # Path to the folder with heatmap images
output_path = "Enter the path to save output images: "           # Path to save processed images

# Create the output directory (if it doesn't exist)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# List all files ending with .png or .jpg in the heatmap directory
image_files = [f for f in os.listdir(heatmap_image_path) if f.endswith('.png') or f.endswith('.jpg')]

for image_file in image_files:
    # Load the heatmap image
    heatmap_img = cv2.imread(os.path.join(heatmap_image_path, image_file))

    # Check if the image was loaded successfully
    if heatmap_img is None:
        print(f"Failed to load heatmap: {image_file}")
        continue

    # Convert the heatmap image from BGR to HSV color space
    hsv_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red and green colors
    # Red color range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Green color range (a narrower range to exclude common colors like leaves)
    lower_green = np.array([40, 100, 150])  # Increased brightness and saturation lower bounds to exclude noise
    upper_green = np.array([70, 255, 255])  # Focused on pure bright green

    # Create masks for red and green regions
    red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2  # Combine masks for two red ranges

    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

    # Use morphological operations to remove small noisy regions
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours for red and green regions
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an image to save the marked heatmap regions
    marked_img = heatmap_img.copy()

    # Draw red contours on the original image
    cv2.drawContours(marked_img, contours_red, -1, (0, 0, 255), 2)  # Red contours

    # Draw green contours on the original image
    cv2.drawContours(marked_img, contours_green, -1, (0, 255, 0), 2)  # Green contours

    # Save the image with marked regions
    output_file_path = os.path.join(output_path, f"highlighted_{image_file}")
    cv2.imwrite(output_file_path, marked_img)

    # Print the processing result
    print(f"Processed: {image_file}, result saved at {output_file_path}")