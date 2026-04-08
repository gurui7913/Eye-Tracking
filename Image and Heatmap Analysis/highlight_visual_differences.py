# Author: Rui Gu
# This script processes original images and corresponding heatmaps to highlight differences.
# The code is entirely written by the author and does not include borrowed components.

# Dependencies:
# - OpenCV: pip install opencv-python
# - NumPy: pip install numpy
# Ensure these dependencies are installed before running the script.

import cv2
import numpy as np
import os

# File paths for the first subject
original_image_path = "/path/to/original_images"   # Path to original images
heatmap_image_path = "/path/to/heatmaps"           # Path to heatmap images
output_path = "/path/to/output"                    # Path to save output images

# Create output folder if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# List all image files ending with .png or .jpg
image_files = [f for f in os.listdir(original_image_path) if f.endswith((".png", ".jpg"))]

for image_file in image_files:
    # Load the original image
    original_img = cv2.imread(os.path.join(original_image_path, image_file))

    # Try to load the heatmap image, checking for different extensions
    heatmap_img = None
    for ext in [".png", ".jpg", ".jpeg"]:
        potential_path = os.path.join(heatmap_image_path, image_file.replace(".png", ext).replace(".jpg", ext))
        if os.path.exists(potential_path):
            heatmap_img = cv2.imread(potential_path)
            break

    # Check if images were loaded successfully
    if original_img is None or heatmap_img is None:
        print(f"Unable to load heatmap: {image_file}")
        continue

    # Crop the heatmap image
    height, width = heatmap_img.shape[:2]
    if width > 800:
        crop_left = (width - 800) // 2
        crop_right = crop_left + 800
    else:
        crop_left = 0
        crop_right = width

    if height > 450:
        crop_top = (height - 450) // 2
        crop_bottom = crop_top + 450
    else:
        crop_top = 0
        crop_bottom = height

    heatmap_img = heatmap_img[crop_top:crop_bottom, crop_left:crop_right]

    # Get the dimensions of the original image
    original_height, original_width = original_img.shape[:2]

    # Resize the heatmap to match the dimensions of the original image
    resized_heatmap = cv2.resize(heatmap_img, (original_width, original_height), interpolation=cv2.INTER_AREA)

    # Compute the absolute difference between the two images
    difference = cv2.absdiff(resized_heatmap, original_img)

    # Convert the difference image to grayscale
    gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to extract significant difference regions
    _, threshold_diff = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

    # Generate a heatmap from the thresholded difference
    heatmap = cv2.applyColorMap(threshold_diff, cv2.COLORMAP_JET)

    # Overlay the heatmap onto the original image
    combined_img = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)

    # Save the result to the output folder
    output_file_path = os.path.join(output_path, f"highlighted_{image_file}")
    cv2.imwrite(output_file_path, combined_img)

    # Output processing result
    print(f"Processed: {image_file}, result saved to {output_file_path}")