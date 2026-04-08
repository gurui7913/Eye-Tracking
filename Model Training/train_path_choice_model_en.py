# Author: Rui Gu
# This script combines visual and text features, balances data using SMOTE, and trains a Random Forest model for classification.
# Dependencies:
# - PyTorch: pip install torch
# - NumPy: pip install numpy
# - Pandas: pip install pandas
# - Scikit-learn: pip install scikit-learn
# - imbalanced-learn: pip install imbalanced-learn
# Ensure these dependencies are installed before running the script.

import torch
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 1: Combine visual features from three sets
visual_features_folders = [
    'Enter the path to the first visual feature folder: ',
    'Enter the path to the second visual feature folder: ',
    'Enter the path to the third visual feature folder: '
]

all_visual_features = []
for visual_features_folder in visual_features_folders:
    visual_feature_files = [f for f in os.listdir(visual_features_folder) if f.endswith('.pt')]
    for visual_feature_file in visual_feature_files:
        visual_feature_path = os.path.join(visual_features_folder, visual_feature_file)
        visual_feature_tensor = torch.load(visual_feature_path)
        all_visual_features.append(visual_feature_tensor.numpy())

all_visual_features = np.concatenate(all_visual_features, axis=0)
print("Combined visual features shape:", all_visual_features.shape)

# Step 2: Combine text features from three sets
text_features_folders = [
    'Enter the path to the first text feature folder: ',
    'Enter the path to the first text feature folder: ',
    'Enter the path to the first text feature folder: '
]

all_text_features = []
for text_features_folder in text_features_folders:
    text_feature_files = [f for f in os.listdir(text_features_folder) if f.endswith('.pt')]
    for text_feature_file in text_feature_files:
        text_feature_path = os.path.join(text_features_folder, text_feature_file)
        text_feature_tensor = torch.load(text_feature_path)
        all_text_features.append(text_feature_tensor.numpy())

all_text_features = np.concatenate(all_text_features, axis=0)
print("Combined text features shape:", all_text_features.shape)

# Step 3: Combine visual and text features
input_features = np.concatenate((all_visual_features, all_text_features), axis=1)
print("Final feature matrix shape after combining visual and text features:", input_features.shape)

# Step 4: Combine labels
labels_files = [
    'Enter the path to the first labels CSV file: ',
    'Enter the path to the second labels CSV file: ',
    'Enter the path to the third labels CSV file: '
]

all_labels = []
for labels_file in labels_files:
    labels = pd.read_csv(labels_file)['label']
    all_labels.append(labels)

all_labels = pd.concat(all_labels, axis=0).reset_index(drop=True)
print("Total number of combined labels:", len(all_labels))

# Step 5: Use SMOTE to oversample and balance the data
smote = SMOTE(random_state=42, k_neighbors=4)
X_resampled, y_resampled = smote.fit_resample(input_features, all_labels)
print("Resampled data shape:", X_resampled.shape, "Number of labels:", len(y_resampled))

# Step 6: Define Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

# Step 7: Use cross-validation to evaluate the Random Forest model
cross_val_scores = cross_val_score(model, X_resampled, y_resampled, cv=3)
print("Cross-validation accuracy scores:", cross_val_scores)
print("Mean cross-validation accuracy:", cross_val_scores.mean())

# Step 8: Split the data into training and testing sets, further train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest model accuracy (using test set):", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Random Forest confusion matrix:\n", conf_matrix)

