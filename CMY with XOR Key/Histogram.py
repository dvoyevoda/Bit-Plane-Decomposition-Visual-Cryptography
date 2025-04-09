import cv2
import matplotlib.pyplot as plt
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate grayscale or RGB histogram of an image.")
parser.add_argument("image_path", help="Path to the input image file")
args = parser.parse_args()

# Check if file exists
if not os.path.exists(args.image_path):
    print(f"Error: File '{args.image_path}' not found.")
    exit(1)

# Load the image
img = cv2.imread(args.image_path)
if img is None:
    print(f"Error: Failed to load image '{args.image_path}'.")
    exit(1)

# Determine if image is grayscale or color
if len(img.shape) == 2:
    # Grayscale image
    plt.hist(img.ravel(), 256, [0, 256], color='black')
    plt.title('Grayscale Histogram')
    plt.