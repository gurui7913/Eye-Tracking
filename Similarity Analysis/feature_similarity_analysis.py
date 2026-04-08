# Author: Rui Gu
# This script calculates the cosine similarity between image features and text features.
# Dependencies:
# - PyTorch: pip install torch
# - NumPy: pip install numpy
# - Pandas: pip install pandas
# - scikit-learn: pip install scikit-learn
# Ensure these dependencies are installed before running the script.

import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Define paths
image_features_path = "Enter the path to the image features: "      # Path to folder with image features
text_features_path = "Enter the path to the text features: "        # Path to folder with text features
output_csv_path = "Enter the path to save the output CSV: "         # Path to save the similarity results CSV

# List all image and text feature files
image_files = [f for f in os.listdir(image_features_path) if f.endswith('_features.pt')]
text_files = [f for f in os.listdir(text_features_path) if f.endswith('_features.pt')]

# Ensure that image and text features are matched in the same order
image_files.sort()
text_files.sort()

# List to store results
similarity_results = []

# Check if filenames match
for image_file, text_file in zip(image_files, text_files):
    if image_file.split('_')[0] != text_file.split('_')[0]:
        print(f"Filename mismatch: Image feature {image_file} and Text feature {text_file}")
        continue

    # Load features
    image_feature = torch.load(os.path.join(image_features_path, image_file))
    text_feature = torch.load(os.path.join(text_features_path, text_file))

    # Ensure features are 2D tensors
    if len(image_feature.shape) == 1:
        image_feature = image_feature.unsqueeze(0)
    if len(text_feature.shape) == 1:
        text_feature = text_feature.unsqueeze(0)

    # Compute cosine similarity
    similarity = cosine_similarity(image_feature.detach().numpy(), text_feature.detach().numpy())

    # Output similarity
    similarity_value = similarity[0][0]
    print(f"Image: {image_file} and Text: {text_file} Cosine Similarity: {similarity_value:.4f}")

    # Save to results list
    similarity_results.append({
        "image_file": image_file,
        "text_file": text_file,
        "similarity": similarity_value
    })

# Save similarity results to a CSV file
df = pd.DataFrame(similarity_results)
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df.to_csv(output_csv_path, index=False)

print(f"All image and text feature similarity computations are complete. Results have been saved to {output_csv_path}")
