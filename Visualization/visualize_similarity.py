# Author: Rui Gu
# This script visualizes the cosine similarity between images and text as a bar chart.
# Dependencies:
# - Pandas: pip install pandas
# - Matplotlib: pip install matplotlib
# Ensure these dependencies are installed before running the script.

import pandas as pd
import matplotlib.pyplot as plt

# Load the similarity_results.csv file
similarity_file_path = "Enter the path to the similarity results CSV file: "    # Path to similarity results
similarity_data = pd.read_csv(similarity_file_path)

# Use the correct column names
images = similarity_data['image_file']
similarities = similarity_data['similarity']

# Set global font
plt.rcParams['font.family'] = 'Arial'  # You can change this to your preferred font
plt.rcParams['font.size'] = 12  # Adjust font size

# Create the visualization chart
fig = plt.figure(figsize=(10, 5), facecolor='white')  # Set the entire chart background color to white
ax = fig.add_subplot()

# Set the background color of the axes to white
ax.set_facecolor('white')

# Create a bar chart with custom color
ax.bar(images, similarities, color='#fbbb9b')  # Use #fbbb9b as the bar color
ax.set_xlabel('Images', color='#fbbb9b')  # Use #fbbb9b for the axis label color
ax.set_ylabel('Cosine Similarity', color='#fbbb9b')
ax.set_title('Image and Text Similarity', color='#fbbb9b')

# Modify the tick label colors for x and y axes
plt.xticks(rotation=90, color='#fbbb9b')  # Set the x-axis labels to #fbbb9b
plt.yticks(color='#fbbb9b')  # Set the y-axis labels to #fbbb9b

# Set the color of the axis lines
ax.spines['bottom'].set_color('#fbbb9b')
ax.spines['left'].set_color('#fbbb9b')
ax.spines['top'].set_color('#fbbb9b')
ax.spines['right'].set_color('#fbbb9b')

# Set the width of the axis lines
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

# Modify the tick line colors
ax.tick_params(axis='x', colors='#fbbb9b')  # Set x-axis tick line color to #fbbb9b
ax.tick_params(axis='y', colors='#fbbb9b')  # Set y-axis tick line color to #fbbb9b

# Adjust the chart layout to prevent overlapping
plt.tight_layout()

# Save the chart as a PNG file
output_chart_path = "Enter the path to save the similarity chart (PNG): "     # Path to save the output chart
plt.savefig(output_chart_path)

# Display the chart
plt.show()
