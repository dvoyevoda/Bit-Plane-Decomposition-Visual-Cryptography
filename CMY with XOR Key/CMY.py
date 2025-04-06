import os
import numpy as np
from PIL import Image
import argparse
import math
from BinaryMetrics import psnr, normxcorr2D  # Import PSNR and MNCC functions

def load_image(image_path):
    """Load an image and convert it to a numpy array."""
    return Image.open(image_path)

def save_image(image_array, path):
    """Save a numpy array as an image."""
    Image.fromarray(image_array).save(path)

def generate_random_key(shape):
    """Generate a random noise key with the given shape."""
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)

def load_or_generate_key(expected_shape, key_path="image_key.png"):
    """
    Loads the image key if it exists and has the correct dimensions.
    If the key file doesn't exist or its dimensions don't match,
    generates a random noise key with the expected shape, saves it,
    and returns it.
    """
    if not os.path.exists(key_path):
        print(f"Key image file '{key_path}' not found. Generating a new random key...")
        key_array = generate_random_key(expected_shape)
        save_image(key_array, key_path)
        return key_array

    try:
        key_image = Image.open(key_path)
        key_array = np.array(key_image)
        if key_array.shape != expected_shape:
            print("Key image dimensions do not match the share. Generating a new random key...")
            key_array = generate_random_key(expected_shape)
            save_image(key_array, key_path)
    except Exception as e:
        print(f"Error loading key image ({e}). Generating a new random key...")
        key_array = generate_random_key(expected_shape)
        save_image(key_array, key_path)
    return key_array

def apply_xor(share, key):
    """
    Apply a pixel-wise XOR between the share and the key.
    Both must be numpy arrays of the same shape.
    """
    if share.shape != key.shape:
        raise ValueError("Share and key must have the same dimensions for XOR operation.")
    return np.bitwise_xor(share, key)

def encrypt_share(share, key_path="image_key.png"):
    """Encrypt a share by XOR-ing it with the key image."""
    key = load_or_generate_key(share.shape, key_path)
    encrypted_share = apply_xor(share, key)
    return encrypted_share

def decrypt_share(encrypted_share, key_path="image_key.png"):
    """
    Decrypt an encrypted share by XOR-ing it with the key image.
    Since XOR is its own inverse, this recovers the original share.
    """
    key = load_or_generate_key(encrypted_share.shape, key_path)
    original_share = apply_xor(encrypted_share, key)
    return original_share

def split_into_bit_layers(image_array):
    """
    Splits an image (as a numpy array) into 8 bit layers for each color channel.
    Returns a list (per channel) of lists (bit layers 0 to 7).
    """
    height, width, channels = image_array.shape
    bit_layers = []
    for c in range(channels):
        channel = image_array[:, :, c]
        layers = [(channel >> bit) & 1 for bit in range(8)]
        bit_layers.append(layers)
    return bit_layers

def combine_bit_layers_channel(bit_layers):
    """
    Reconstructs a single channel image from its 8 binary bit layers.
    """
    channel = np.zeros_like(bit_layers[0], dtype=np.uint8)
    for bit in range(8):
        channel |= (bit_layers[bit].astype(np.uint8) << bit)
    return channel

def tint_image(grayscale, color):
    """
    Converts a grayscale image to a tinted full-color image for the specified channel.
    For Cyan (C): output image has 0 in the Red channel and grayscale in Green and Blue.
    For Magenta (M): output image has 0 in the Green channel and grayscale in Red and Blue.
    For Yellow (Y): output image has 0 in the Blue channel and grayscale in Red and Green.
    """
    h, w = grayscale.shape
    tinted = np.zeros((h, w, 3), dtype=np.uint8)
    if color == "C":
        tinted[:, :, 0] = 0
        tinted[:, :, 1] = grayscale
        tinted[:, :, 2] = grayscale
    elif color == "M":
        tinted[:, :, 0] = grayscale
        tinted[:, :, 1] = 0
        tinted[:, :, 2] = grayscale
    elif color == "Y":
        tinted[:, :, 0] = grayscale
        tinted[:, :, 1] = grayscale
        tinted[:, :, 2] = 0
    else:
        tinted = np.stack([grayscale, grayscale, grayscale], axis=2)
    return tinted

def process_image(image_path, output_folder="output", key_path="image_key.png"):
    """
    Process the input image:
      1. Split it into 8-bit layers for each color channel.
      2. For each channel, create two folders:
         - One for the original 8-bit layers
         - One for the XOR-encrypted 8-bit layers
      3. Reconstruct the image for each channel from the decrypted bit layers,
         convert it to a tinted image, and save it in the channel's folder.
      4. Combine the reconstructed channels into a final image, save it in the main
         output folder, and print the PSNR and MNCC levels compared to the original.
    """
    image = load_image(image_path)
    image_array = np.array(image)

    os.makedirs(output_folder, exist_ok=True)
    bit_layers_all = split_into_bit_layers(image_array)
    # Map channel indices to CMY colors
    color_map = {0: "C", 1: "M", 2: "Y"}
    reconstructed_channels = []

    for channel_index, layers in enumerate(bit_layers_all):
        channel_name = color_map.get(channel_index, f"channel_{channel_index}")
        channel_folder = os.path.join(output_folder, channel_name)
        bit_layers_folder = os.path.join(channel_folder, "bit_layers")
        xor_layers_folder = os.path.join(channel_folder, "bit_layers_xor")
        os.makedirs(bit_layers_folder, exist_ok=True)
        os.makedirs(xor_layers_folder, exist_ok=True)

        encrypted_layers = []
        for bit_index, layer in enumerate(layers):
            share = (layer * 255).astype(np.uint8)
            original_filename = os.path.join(bit_layers_folder, f"bit_{bit_index}.png")
            save_image(share, original_filename)
            encrypted_share = encrypt_share(share, key_path)
            encrypted_filename = os.path.join(xor_layers_folder, f"bit_{bit_index}.png")
            save_image(encrypted_share, encrypted_filename)
            encrypted_layers.append(encrypted_share)

        decrypted_layers = []
        for bit_index, enc_layer in enumerate(encrypted_layers):
            dec_layer = decrypt_share(enc_layer, key_path)
            binary_layer = (dec_layer > 127).astype(np.uint8)
            decrypted_layers.append(binary_layer)

        # Reconstruct the grayscale channel from bit layers and store it
        reconstructed_channel = combine_bit_layers_channel(decrypted_layers)
        reconstructed_channels.append(reconstructed_channel)
        # Tint the grayscale channel for full-color representation
        tinted_channel = tint_image(reconstructed_channel, channel_name)
        recon_filename = os.path.join(channel_folder, f"reconstructed_{channel_name}.png")
        save_image(tinted_channel, recon_filename)
        print(f"Channel {channel_name} processed. Reconstructed tinted image saved as {recon_filename}.")

    # Combine the three channels into a final reconstructed image
    if len(reconstructed_channels) == 3:
        final_reconstructed = np.stack(reconstructed_channels, axis=2)
    else:
        final_reconstructed = np.stack([reconstructed_channels[0]] * 3, axis=2)

    final_filename = os.path.join(output_folder, "final_reconstructed.png")
    save_image(final_reconstructed, final_filename)
    print(f"Final reconstructed image saved as {final_filename}.")

    # Compute and print PSNR and MNCC metrics using functions from BinaryMetrics.py
    psnr_value = psnr(image_array, final_reconstructed)
    # normxcorr2D expects PIL images, so convert the arrays
    original_pil = Image.fromarray(image_array.astype(np.uint8))
    reconstructed_pil = Image.fromarray(final_reconstructed.astype(np.uint8))
    mncc_value = normxcorr2D(original_pil, reconstructed_pil)
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"MNCC: {mncc_value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Cryptography with XOR Key Encryption for CMY channels")
    parser.add_argument("image", help="Path to the input image file")
    parser.add_argument("--output", default="output", help="Folder to save output directories for CMY channels")
    parser.add_argument("--key", default="image_key.png", help="File name for the image key (should be in the project folder)")
    args = parser.parse_args()

    process_image(args.image, args.output, args.key)