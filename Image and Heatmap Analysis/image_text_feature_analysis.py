# Author: Rui Gu
# Portions of this code, such as the use of CLIP for text feature extraction, are adapted from:
# https://github.com/openai/CLIP.git
# The code has been modified to suit the specific requirements of this project.

# This script extracts text features from descriptions using the CLIP model.
# Dependencies:
# - PyTorch: pip install torch
# - OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git
# - Pillow: pip install pillow
# Ensure these dependencies are installed before running the script.


import os
import torch
import clip
from PIL import Image
import numpy as np

# Set paths
original_image_path = "Enter the path to the original images: "      # Path to original images
focus_area_path = "Enter the path to the focus area images: "        # Path to focus area images
output_feature_path = "Enter the path to save output features: "     # Path to save output features

# Create output folder
if not os.path.exists(output_feature_path):
    os.makedirs(output_feature_path)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# List all original images
original_files = [f for f in os.listdir(original_image_path) if f.endswith(('.jpg', '.png'))]

# Start processing each original image
for original_file in original_files:
    # Get the corresponding heatmap file name
    base_name = os.path.splitext(original_file)[0]

    # Search for the corresponding heatmap image, supporting different extensions
    possible_extensions = ['.jpg', '.png', '.jpeg']
    heatmap_img_path = None

    for ext in possible_extensions:
        potential_path = os.path.join(focus_area_path, base_name + ext)
        if os.path.exists(potential_path):
            heatmap_img_path = potential_path
            break

    if heatmap_img_path is None:
        print(f"No corresponding heatmap found: {original_file}")
        continue

    # Load the original image and its corresponding heatmap
    try:
        original_img = Image.open(os.path.join(original_image_path, original_file))
        focus_img = Image.open(heatmap_img_path)
    except Exception as e:
        print(f"Unable to load images (PIL), error: {e}")
        continue

    # Preprocess the original image and heatmap using CLIP
    original_preprocessed = preprocess(original_img).unsqueeze(0).to(device)
    focus_preprocessed = preprocess(focus_img).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        original_features = model.encode_image(original_preprocessed)
        focus_features = model.encode_image(focus_preprocessed)

    # Apply different color weights to heatmap features
    # Note: This assumes different color regions have already been extracted and weighted (example: red weighted higher)
    weight = 1.5 if 'red' in base_name else 1.0  # Simple example, apply weights based on color regions in practice
    combined_features = (original_features + weight * focus_features) / 2

    # Save features
    feature_output_path = os.path.join(output_feature_path, f"{base_name}_features.pt")
    torch.save(combined_features.cpu(), feature_output_path)

    print(f"Processed image: {original_file}, features saved at: {feature_output_path}")

print("Feature extraction and combined analysis for all images completed.")
