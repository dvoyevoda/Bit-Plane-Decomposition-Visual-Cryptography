from PIL import Image
import numpy as np
import os
import BinaryMetrics  # Import the metrics module

# Define the input and output file names
input_filename = "input.png"
reconstructed_filename = "reconstructed.png"

# Create an output directory to store all generated images
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Open the input image and convert it to grayscale ('L' mode)
img = Image.open(input_filename).convert("L")
img_array = np.array(img)

# Initialize a dictionary to hold the binary (0 or 255) images for each bit level
bit_images = {}

# Process each bit level (0 = least significant, 7 = most significant)
for bit in range(8):
    # Extract the bit mask for the current bit: result is 0 or 1 per pixel.
    bit_mask = (img_array >> bit) & 1
    
    # Multiply by 255 to get a binary image (0 or 255)
    bit_image = (bit_mask * 255).astype(np.uint8)
    
    # Save the binary bit image in the output directory (e.g., "bit_0.png", ..., "bit_7.png")
    bit_filename = f"bit_{bit}.png"
    bit_path = os.path.join(output_dir, bit_filename)
    Image.fromarray(bit_image).save(bit_path)
    
    # Store the binary mask (0 or 1) for later reconstruction.
    bit_images[bit] = bit_mask

# Reconstruct the original image from the bit-level images
# The reconstruction sums up each bit weighted by 2^bit.
reconstructed = np.zeros_like(img_array, dtype=np.uint8)
for bit in range(8):
    reconstructed += (bit_images[bit] * (2 ** bit)).astype(np.uint8)

# Save the reconstructed image in the output directory
reconstructed_path = os.path.join(output_dir, reconstructed_filename)
Image.fromarray(reconstructed).save(reconstructed_path)

print("Bit-level images and the reconstructed image have been saved in:", output_dir)

# Compute accuracy metrics
psnr_value = BinaryMetrics.psnr(img_array, reconstructed)
# For MNCC, convert arrays back to PIL images
mncc_value = BinaryMetrics.normxcorr2D(Image.fromarray(img_array), Image.fromarray(reconstructed))

print(f"PSNR: {psnr_value:.2f} dB")
print(f"MNCC: {mncc_value:.4f}")