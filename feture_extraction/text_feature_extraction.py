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

import torch
import clip
from PIL import Image
import os

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths for text description files and output
descriptions_path = "Enter the path to the description text files: "  # Path to the folder containing text descriptions
output_path = "Enter the path to save output text features: "         # Path to save the extracted text features

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Iterate through all description files
for description_file in os.listdir(descriptions_path):
    if not description_file.endswith(".txt"):
        continue

    # Read the description file
    with open(os.path.join(descriptions_path, description_file), "r") as f:
        descriptions = f.readlines()

    # Process each description
    for i, description in enumerate(descriptions):
        description = description.strip()

        # Convert the text to feature vectors
        text_tokens = clip.tokenize([description]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)

        # Save the feature vectors
        text_feature_path = os.path.join(output_path, f"{description_file}_desc_{i}_features.pt")
        torch.save(text_features.cpu(), text_feature_path)

        print(f"Processed text description: {description_file}, features saved at: {text_feature_path}")

print("Feature extraction for all text descriptions is complete.")
