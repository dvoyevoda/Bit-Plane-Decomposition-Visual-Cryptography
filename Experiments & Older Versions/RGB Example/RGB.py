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

# Create subdirectories for each channel (R, G, B)
channels = ['R', 'G', 'B']
channel_dirs = {}
for ch in channels:
    channel_dir = os.path.join(output_dir, ch)
    os.makedirs(channel_dir, exist_ok=True)
    channel_dirs[ch] = channel_dir

# Open the input image in RGB mode
img = Image.open(input_filename).convert("RGB")
img_array = np.array(img)

# Extract R, G, B channels as 2D arrays (intensity values)
R_channel = img_array[:, :, 0]
G_channel = img_array[:, :, 1]
B_channel = img_array[:, :, 2]

# Create original colored channel images that retain the color
original_color_channels = {
    'R': np.stack((R_channel, np.zeros_like(R_channel), np.zeros_like(R_channel)), axis=-1),
    'G': np.stack((np.zeros_like(G_channel), G_channel, np.zeros_like(G_channel)), axis=-1),
    'B': np.stack((np.zeros_like(B_channel), np.zeros_like(B_channel), B_channel), axis=-1)
}

# Store intensity channels in a dictionary for bit-plane processing
rgb_channels = {'R': R_channel, 'G': G_channel, 'B': B_channel}

# Dictionary to store the reconstructed channels (intensity images)
reconstructed_channels = {}

# Process each channel individually
for channel_name, channel_array in rgb_channels.items():
    # Save the original colored channel image in its respective folder.
    # This image retains the channel’s color (e.g., only blue in the Blue folder).
    color_orig_filename = f"{channel_name}_color_original.png"
    color_orig_path = os.path.join(channel_dirs[channel_name], color_orig_filename)
    Image.fromarray(original_color_channels[channel_name]).save(color_orig_path)
    
    # Dictionary to hold bit masks for reconstruction
    bit_images = {}

    # For each bit level (0 = LSB, 7 = MSB)
    for bit in range(8):
        # Extract the bit mask for this bit (values will be 0 or 1)
        bit_mask = (channel_array >> bit) & 1
        
        # Multiply by 255 so the bit-plane can be visualized (0 or 255)
        bit_image = (bit_mask * 255).astype(np.uint8)
        
        # Save the bit-level image in the channel-specific folder
        bit_filename = f"{channel_name}_bit_{bit}.png"
        bit_path = os.path.join(channel_dirs[channel_name], bit_filename)
        Image.fromarray(bit_image).save(bit_path)
        
        # Store the bit mask for later reconstruction
        bit_images[bit] = bit_mask

    # Reconstruct the channel by summing the bit planes weighted by 2^bit
    reconstructed_channel = np.zeros_like(channel_array, dtype=np.uint8)
    for bit in range(8):
        reconstructed_channel += (bit_images[bit] * (2 ** bit)).astype(np.uint8)
    
    # Optionally, save the reconstructed channel image in its subfolder
    recon_filename = f"{channel_name}_reconstructed.png"
    recon_path = os.path.join(channel_dirs[channel_name], recon_filename)
    Image.fromarray(reconstructed_channel).save(recon_path)
    
    # Store the reconstructed channel for final combination
    reconstructed_channels[channel_name] = reconstructed_channel

# Combine the reconstructed R, G, and B channels to form the final color image
reconstructed_RGB = np.stack((reconstructed_channels['R'],
                              reconstructed_channels['G'],
                              reconstructed_channels['B']),
                              axis=-1)

# Save the final reconstructed color image in the output directory
final_reconstructed_path = os.path.join(output_dir, final_reconstructed_filename)
Image.fromarray(reconstructed_RGB).save(final_reconstructed_path)

print("Bit-level images for each channel, original colored channel images, and the final reconstructed color image have been saved in:", output_dir)

# Compute accuracy metrics for each channel (comparing intensity channels)
for channel_name in channels:
    original_channel = rgb_channels[channel_name]
    recon_channel = reconstructed_channels[channel_name]
    psnr_value = BinaryMetrics.psnr(original_channel, recon_channel)
    mncc_value = BinaryMetrics.normxcorr2D(Image.fromarray(original_channel), Image.fromarray(recon_channel))
    print(f"Channel {channel_name} - PSNR: {psnr_value:.2f} dB, MNCC: {mncc_value:.4f}")