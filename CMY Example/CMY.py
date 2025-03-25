from PIL import Image
import numpy as np
import os
import BinaryMetrics  # Import the metrics module

# Define input and output filenames
input_filename = "input.png"
final_reconstructed_filename = "reconstructed_color.png"

# Create an output directory to store all generated images
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories for each channel (C, M, Y)
channels = ['C', 'M', 'Y']
channel_dirs = {}
for ch in channels:
    channel_dir = os.path.join(output_dir, ch)
    os.makedirs(channel_dir, exist_ok=True)
    channel_dirs[ch] = channel_dir

# Open the input image in RGB mode
img = Image.open(input_filename).convert("RGB")
img_array = np.array(img)

# Extract R, G, B channels from the RGB image
R_channel = img_array[:, :, 0]
G_channel = img_array[:, :, 1]
B_channel = img_array[:, :, 2]

# Convert from RGB to CMY:
# C = 255 - R, M = 255 - G, Y = 255 - B
C_channel = 255 - R_channel
M_channel = 255 - G_channel
Y_channel = 255 - B_channel

# Create "original colored" CMY channel images.
# For display, we tint each channel with its representative color:
#   - Cyan is represented as (0, C, C)
#   - Magenta as (M, 0, M)
#   - Yellow as (Y, Y, 0)
original_color_channels = {
    'C': np.stack((np.zeros_like(C_channel), C_channel, C_channel), axis=-1),
    'M': np.stack((M_channel, np.zeros_like(M_channel), M_channel), axis=-1),
    'Y': np.stack((Y_channel, Y_channel, np.zeros_like(Y_channel)), axis=-1)
}

# Store the intensity (grayscale) CMY channels in a dictionary for processing
cmy_channels = {'C': C_channel, 'M': M_channel, 'Y': Y_channel}

# Dictionary to store the reconstructed channels (intensity images)
reconstructed_channels = {}

# Process each CMY channel individually
for channel_name, channel_array in cmy_channels.items():
    # Save the original colored channel image in its respective folder.
    # This image retains the "color" for the channel (e.g., in the C folder, the image appears cyan).
    color_orig_filename = f"{channel_name}_color_original.png"
    color_orig_path = os.path.join(channel_dirs[channel_name], color_orig_filename)
    Image.fromarray(original_color_channels[channel_name]).save(color_orig_path)
    
    # Dictionary to hold each bit-plane mask for reconstruction later
    bit_images = {}

    # For each bit level (0 = least significant, 7 = most significant)
    for bit in range(8):
        # Extract the bit mask for this bit (each pixel becomes 0 or 1)
        bit_mask = (channel_array >> bit) & 1
        
        # Multiply by 255 so the bit-plane is viewable (0 or 255)
        bit_image = (bit_mask * 255).astype(np.uint8)
        
        # Save the bit-level image in the channel-specific folder
        bit_filename = f"{channel_name}_bit_{bit}.png"
        bit_path = os.path.join(channel_dirs[channel_name], bit_filename)
        Image.fromarray(bit_image).save(bit_path)
        
        # Save the bit mask for reconstruction
        bit_images[bit] = bit_mask

    # Reconstruct the channel by summing the bit planes weighted by 2^bit
    reconstructed_channel = np.zeros_like(channel_array, dtype=np.uint8)
    for bit in range(8):
        reconstructed_channel += (bit_images[bit] * (2 ** bit)).astype(np.uint8)
    
    # Save the reconstructed channel image in its folder
    recon_filename = f"{channel_name}_reconstructed.png"
    recon_path = os.path.join(channel_dirs[channel_name], recon_filename)
    Image.fromarray(reconstructed_channel).save(recon_path)
    
    # Store the reconstructed channel for final combination
    reconstructed_channels[channel_name] = reconstructed_channel

# Combine the reconstructed CMY channels to form the final CMY image.
reconstructed_CMY = np.stack((reconstructed_channels['C'],
                              reconstructed_channels['M'],
                              reconstructed_channels['Y']),
                              axis=-1)

# Convert the reconstructed CMY image back to RGB using:
#   R = 255 - C, G = 255 - M, B = 255 - Y
reconstructed_RGB = 255 - reconstructed_CMY

# Save the final reconstructed color image in the output folder.
final_reconstructed_path = os.path.join(output_dir, final_reconstructed_filename)
Image.fromarray(reconstructed_RGB).save(final_reconstructed_path)

print("Bit-level images for each channel, original colored CMY channel images, and the final reconstructed color image have been saved in:", output_dir)

# Compute accuracy metrics (PSNR and MNCC) for each CMY channel
for channel_name in channels:
    original_channel = cmy_channels[channel_name]
    recon_channel = reconstructed_channels[channel_name]
    psnr_value = BinaryMetrics.psnr(original_channel, recon_channel)
    mncc_value = BinaryMetrics.normxcorr2D(Image.fromarray(original_channel), Image.fromarray(recon_channel))
    print(f"Channel {channel_name} - PSNR: {psnr_value:.2f} dB, MNCC: {mncc_value:.4f}")