import io
import hashlib
import numpy as np
from PIL import Image
import pywt
from scipy.fftpack import dct, idct
import pyqrcode
import math
import struct  # For packing dimensions

# --- Configuration ---
WAVELET_TYPE = "db4"  # Wavelet type (e.g., 'haar', 'db1', 'db4', 'bior4.4')
DCT_BLOCK_SIZE = 8  # Block size for DCT
EMBED_SUBBAND = "HL"  # Subband to embed into ('LH', 'HL', or 'HH')
# Choose mid-frequency coefficients indices within the 8x8 DCT block
# Example: A few coefficients not too close to DC (0,0) or high freq.
# Using more coefficients increases capacity but might reduce robustness/invisibility.
# Let's use one coefficient per block for simplicity in this example.
DCT_COEFF_INDICES = [(3, 4)]  # (row, col) - Use a single coefficient per block
# Embedding strength - adjusts how much coefficients are modified.
# Needs tuning: too low -> extraction fails, too high -> visible artifacts.
ALPHA = 20.0

# --- Helper Functions ---


def dct2(block):
    """2D DCT"""
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct2(block):
    """2D Inverse DCT"""
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def rgb2ycbcr(im_rgb):
    """Convert RGB Pillow Image to YCbCr Numpy array (float)"""
    im_np = np.array(im_rgb).astype(float)
    r, g, b = im_np[:, :, 0], im_np[:, :, 1], im_np[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def ycbcr2rgb(im_ycbcr):
    """Convert YCbCr Numpy array (float) back to RGB Pillow Image"""
    y, cb, cr = im_ycbcr[:, :, 0], im_ycbcr[:, :, 1], im_ycbcr[:, :, 2]
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    # Clip values and convert to uint8
    im_np = np.stack([r, g, b], axis=-1)
    im_np = np.clip(im_np, 0, 255).astype(np.uint8)
    return Image.fromarray(im_np)


def int_to_bits(n, num_bits):
    """Convert an integer to a list of bits (0s and 1s)."""
    return [(n >> i) & 1 for i in range(num_bits - 1, -1, -1)]


def bits_to_int(bits):
    """Convert a list of bits to an integer."""
    n = 0
    for bit in bits:
        n = (n << 1) | bit
    return n


def calculate_hash(image: Image.Image) -> str:
    """Calculates the SHA-256 hash of the image content."""
    img_byte_arr = io.BytesIO()
    # Use a lossless format like PNG for hashing consistency
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    hasher = hashlib.sha256()
    hasher.update(img_byte_arr)
    return hasher.hexdigest()


def generate_binary_qr_data(data: str) -> (np.ndarray, tuple):
    """
    Generates QR code for data and returns its binary representation
    (numpy array of 0s and 1s) and its dimensions (width, height).
    """
    qr = pyqrcode.create(data, error="H")  # High error correction
    # Get QR code as a string of '0' and '1'
    qr_text = qr.text(quiet_zone=1)  # Use quiet zone for better structure
    qr_lines = qr_text.strip().split("\n")
    qr_dim = len(qr_lines)  # QR codes are square

    # Convert to numpy array of ints (0 or 1)
    qr_matrix = np.zeros((qr_dim, qr_dim), dtype=int)
    for r, line in enumerate(qr_lines):
        for c, char in enumerate(line):
            qr_matrix[r, c] = 1 if char == "1" else 0

    # Flatten the matrix into a 1D array of bits
    qr_flat_bits = qr_matrix.flatten()

    print(f"Generated QR Code for data: '{data[:10]}...'")
    print(f"QR Code Dimensions: {qr_dim}x{qr_dim}")
    print(f"Total QR bits: {len(qr_flat_bits)}")

    return qr_flat_bits, (qr_dim, qr_dim)


# --- Embedding Function ---


def embed_qr_dct_wavelet(
    original_image: Image.Image, alpha: float = ALPHA
) -> Image.Image:
    """
    Embeds the hash QR code into the image using Wavelet-DCT.
    Args:
        original_image: The PIL Image object.
        alpha: Embedding strength.
    Returns:
        A new PIL Image object with the QR code embedded.
    """
    # 1. Calculate Hash and Generate Binary QR Data
    image_hash = calculate_hash(original_image)
    qr_bits, qr_dims = generate_binary_qr_data(image_hash)
    qr_width, qr_height = qr_dims

    # 2. Prepare data to embed: Dimensions (16 bits each) + QR bits
    dim_bits_w = int_to_bits(qr_width, 16)
    dim_bits_h = int_to_bits(qr_height, 16)
    data_bits = dim_bits_w + dim_bits_h + list(qr_bits)
    data_len = len(data_bits)
    print(f"Total bits to embed (dims + data): {data_len}")

    # 3. Convert Image to YCbCr
    img_ycbcr = rgb2ycbcr(original_image)
    y_channel = img_ycbcr[:, :, 0]
    cb_channel = img_ycbcr[:, :, 1]
    cr_channel = img_ycbcr[:, :, 2]

    # 4. Perform DWT on Y channel
    coeffs = pywt.dwt2(y_channel, WAVELET_TYPE)
    LL, (LH, HL, HH) = coeffs

    # Select subband for embedding
    if EMBED_SUBBAND == "LH":
        embed_coeffs = LH
    elif EMBED_SUBBAND == "HH":
        embed_coeffs = HH
    else:  # Default to HL
        embed_coeffs = HL

    # 5. Check Capacity
    sub_h, sub_w = embed_coeffs.shape
    max_blocks = (sub_h // DCT_BLOCK_SIZE) * (sub_w // DCT_BLOCK_SIZE)
    capacity = max_blocks * len(DCT_COEFF_INDICES)
    print(f"Embedding in subband: {EMBED_SUBBAND} (shape: {embed_coeffs.shape})")
    print(f"DCT Block Size: {DCT_BLOCK_SIZE}x{DCT_BLOCK_SIZE}")
    print(f"Coefficients per block: {len(DCT_COEFF_INDICES)}")
    print(f"Available embedding capacity: {capacity} bits")

    if data_len > capacity:
        raise ValueError(
            f"Data size ({data_len} bits) exceeds capacity ({capacity} bits). "
            "Try a smaller QR code (less data/lower error correction), "
            "larger image, different wavelet, more coeffs per block, or embed in more subbands."
        )

    # 6. Embed Data
    data_idx = 0
    embedded_coeffs = embed_coeffs.copy()  # Work on a copy

    for r in range(0, sub_h - DCT_BLOCK_SIZE + 1, DCT_BLOCK_SIZE):
        for c in range(0, sub_w - DCT_BLOCK_SIZE + 1, DCT_BLOCK_SIZE):
            if data_idx >= data_len:
                break  # Stop if all data is embedded

            block = embedded_coeffs[r : r + DCT_BLOCK_SIZE, c : c + DCT_BLOCK_SIZE]

            # Apply DCT
            dct_block = dct2(block)

            # Embed bits into selected coefficients
            for coeff_r, coeff_c in DCT_COEFF_INDICES:
                if data_idx < data_len:
                    bit = data_bits[data_idx]
                    # Simple modification: add/subtract alpha based on bit
                    # Map bit 0 to -1, bit 1 to +1
                    modification = alpha * (bit * 2 - 1)
                    dct_block[coeff_r, coeff_c] += modification
                    data_idx += 1
                else:
                    break  # No more data

            # Apply IDCT
            modified_block = idct2(dct_block)

            # Replace original block with modified one
            embedded_coeffs[r : r + DCT_BLOCK_SIZE, c : c + DCT_BLOCK_SIZE] = (
                modified_block
            )

        if data_idx >= data_len:
            break

    print(f"Finished embedding {data_idx} bits.")

    # 7. Reconstruct Y channel with IDWT
    if EMBED_SUBBAND == "LH":
        coeffs_modified = LL, (embedded_coeffs, HL, HH)
    elif EMBED_SUBBAND == "HH":
        coeffs_modified = LL, (LH, HL, embedded_coeffs)
    else:  # HL
        coeffs_modified = LL, (LH, embedded_coeffs, HH)

    y_channel_modified = pywt.idwt2(coeffs_modified, WAVELET_TYPE)

    # Ensure dimensions match original Y channel (IDWT can sometimes change size slightly)
    h_orig, w_orig = y_channel.shape
    y_channel_modified = y_channel_modified[:h_orig, :w_orig]

    # 8. Combine Channels and Convert Back to RGB
    stego_img_ycbcr = np.stack([y_channel_modified, cb_channel, cr_channel], axis=-1)
    stego_image_pil = ycbcr2rgb(stego_img_ycbcr)

    return stego_image_pil


# --- Extraction Function ---


def extract_qr_dct_wavelet(
    stego_image: Image.Image, alpha: float = ALPHA
) -> Image.Image | None:
    """
    Extracts the embedded QR code from the stego image.
    Args:
        stego_image: The PIL Image object containing the hidden QR code.
        alpha: The embedding strength used during embedding (important for thresholding).
    Returns:
        A PIL Image of the extracted QR code, or None if extraction fails.
    """
    try:
        # 1. Convert Image to YCbCr
        img_ycbcr = rgb2ycbcr(stego_image)
        y_channel = img_ycbcr[:, :, 0]

        # 2. Perform DWT on Y channel
        coeffs = pywt.dwt2(y_channel, WAVELET_TYPE)
        LL, (LH, HL, HH) = coeffs

        # Select the same subband used for embedding
        if EMBED_SUBBAND == "LH":
            embed_coeffs = LH
        elif EMBED_SUBBAND == "HH":
            embed_coeffs = HH
        else:  # Default to HL
            embed_coeffs = HL

        sub_h, sub_w = embed_coeffs.shape
        print(f"Extracting from subband: {EMBED_SUBBAND} (shape: {embed_coeffs.shape})")

        # 3. Extract Data Bits
        extracted_bits = []
        bits_to_extract = -1  # Unknown until dimensions are read
        dim_bits_extracted = 0
        qr_width, qr_height = -1, -1

        # First extract dimensions (32 bits)
        for r in range(0, sub_h - DCT_BLOCK_SIZE + 1, DCT_BLOCK_SIZE):
            for c in range(0, sub_w - DCT_BLOCK_SIZE + 1, DCT_BLOCK_SIZE):
                if dim_bits_extracted >= 32:
                    break

                block = embed_coeffs[r : r + DCT_BLOCK_SIZE, c : c + DCT_BLOCK_SIZE]
                dct_block = dct2(block)

                for coeff_r, coeff_c in DCT_COEFF_INDICES:
                    if dim_bits_extracted < 32:
                        coeff_val = dct_block[coeff_r, coeff_c]
                        # Simple thresholding: positive coeff -> bit 1, negative -> bit 0
                        # Assumes original coeff was near zero or the modification dominates.
                        # A zero threshold might be too simple; comparing to neighbors
                        # or using the original image could be more robust but complex.
                        # Let's assume the modification is strong enough relative to original value.
                        # We check against a small threshold around 0.
                        # If coeff_val > 0 -> likely 1 was embedded (added alpha)
                        # If coeff_val < 0 -> likely 0 was embedded (subtracted alpha)
                        bit = 1 if coeff_val > 0 else 0
                        extracted_bits.append(bit)
                        dim_bits_extracted += 1
                    else:
                        break
            if dim_bits_extracted >= 32:
                break

        if len(extracted_bits) < 32:
            print("Error: Could not extract enough bits for dimensions.")
            return None

        # Decode dimensions
        qr_width = bits_to_int(extracted_bits[0:16])
        qr_height = bits_to_int(extracted_bits[16:32])

        if (
            qr_width <= 0 or qr_height <= 0 or qr_width > 200 or qr_height > 200
        ):  # Sanity check
            print(
                f"Error: Extracted invalid dimensions ({qr_width}x{qr_height}). Extraction likely failed."
            )
            print(f"First 32 bits read: {extracted_bits[:32]}")
            return None

        print(f"Extracted QR Dimensions: {qr_width}x{qr_height}")
        total_qr_bits = qr_width * qr_height
        bits_to_extract = 32 + total_qr_bits  # Total bits = dim_bits + data_bits
        print(f"Expecting {total_qr_bits} QR data bits ({bits_to_extract} total bits).")

        # Continue extracting remaining QR data bits
        extracted_qr_data_bits = []
        bits_extracted_count = 32  # We already have the dimension bits

        # Continue iterating through blocks from where we left off (or restart is simpler)
        extracted_bits = []  # Restart bit collection for simplicity
        bits_extracted_count = 0

        for r in range(0, sub_h - DCT_BLOCK_SIZE + 1, DCT_BLOCK_SIZE):
            for c in range(0, sub_w - DCT_BLOCK_SIZE + 1, DCT_BLOCK_SIZE):
                if bits_extracted_count >= bits_to_extract:
                    break

                block = embed_coeffs[r : r + DCT_BLOCK_SIZE, c : c + DCT_BLOCK_SIZE]
                dct_block = dct2(block)

                for coeff_r, coeff_c in DCT_COEFF_INDICES:
                    if bits_extracted_count < bits_to_extract:
                        coeff_val = dct_block[coeff_r, coeff_c]
                        bit = 1 if coeff_val > 0 else 0
                        extracted_bits.append(bit)
                        bits_extracted_count += 1
                    else:
                        break
            if bits_extracted_count >= bits_to_extract:
                break

        print(f"Total bits extracted: {len(extracted_bits)}")

        if len(extracted_bits) < bits_to_extract:
            print(
                f"Error: Extracted only {len(extracted_bits)} bits, expected {bits_to_extract}."
            )
            return None

        # 4. Reconstruct QR Code
        # Extract the actual QR data bits (after the dimension bits)
        qr_data_bits = extracted_bits[32:bits_to_extract]

        if len(extracted_bits) < bits_to_extract:
            print(
                f"Error: Extracted only {len(extracted_bits)} bits, expected {bits_to_extract}."
            )
            return None

        # 4. Reconstruct QR Code
        qr_data_bits = extracted_bits[32:bits_to_extract]

        if len(qr_data_bits) != total_qr_bits:
            print(
                f"Error: Mismatch in expected QR data bits ({total_qr_bits}) and extracted ({len(qr_data_bits)})."
            )
            return None

        # Reshape into the QR matrix - FORCE uint8 HERE
        qr_matrix = np.array(qr_data_bits, dtype=np.uint8).reshape(
            (qr_height, qr_width)
        )

        # 5. Create QR Code Image (optional, for display/verification)
        scale_factor = 5  # Make the extracted QR image larger for viewing

        # Perform Kronecker product (uint8 * uint8 should result in uint8)
        qr_image_np = np.kron(
            qr_matrix, np.ones((scale_factor, scale_factor), dtype=np.uint8)
        )

        # Multiply by 255 to map 0->0 and 1->255. Ensure result is uint8.
        qr_image_np = (qr_image_np * 255).astype(np.uint8)

        # Add a white border (quiet zone)
        border_size = 4 * scale_factor
        # Padding uint8 with a uint8 value (255) should preserve uint8 type
        qr_image_bordered = np.pad(
            qr_image_np, pad_width=border_size, mode="constant", constant_values=255
        )

        # >>> FIX: Ensure the array is uint8 before passing to fromarray <<<
        # Although previous steps should ensure it, explicit cast is safest.
        if qr_image_bordered.dtype != np.uint8:
            print(
                f"Warning: Array dtype before fromarray was {qr_image_bordered.dtype}, converting to uint8."
            )
            qr_image_bordered = qr_image_bordered.astype(np.uint8)

        # Now create the PIL Image from the uint8 array
        qr_image_pil = Image.fromarray(qr_image_bordered)  # Mode 'L' should be inferred

        print("Extraction successful.")
        return qr_image_pil

    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        import traceback

        traceback.print_exc()
        return None


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Load Original Image
    # Make sure you have an image file named 'input_image.png' or 'input_image.jpg'
    # Or change the filename below
    try:
        original_image_path = "data/samples/jpeg/Wadapav.jpeg"  # CHANGE AS NEEDED
        original_image = Image.open(original_image_path).convert("RGB")
        print(
            f"Loaded original image: {original_image_path} ({original_image.width}x{original_image.height})"
        )

        # Ensure image dimensions are multiples of 2 for DWT (or handle padding)
        w, h = original_image.size
        # Simple check, might need more sophisticated handling for multi-level DWT
        if w % 2 != 0 or h % 2 != 0:
            print(
                "Warning: Image dimensions are not multiples of 2. Resizing slightly."
            )
            original_image = original_image.resize(((w // 2) * 2, (h // 2) * 2))
            print(f"Resized to: {original_image.width}x{original_image.height}")

        # 2. Embed the QR Code
        print("\n--- Embedding ---")
        stego_image = embed_qr_dct_wavelet(original_image, alpha=ALPHA)

        # 3. Save Stego Image
        stego_image_path = "stego_image.png"  # Save as PNG (lossless)
        stego_image.save(stego_image_path)
        print(f"\nStego image saved to: {stego_image_path}")

        # Display images (optional, requires matplotlib)
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(original_image)
            plt.subplot(1, 2, 2)
            plt.title("Stego Image (with hidden QR)")
            plt.imshow(stego_image)
            plt.show()
        except ImportError:
            print(
                "\nMatplotlib not found. Cannot display images. Please install it (`pip install matplotlib`)."
            )
            original_image.show(title="Original Image")
            stego_image.show(title="Stego Image")

        # 4. Load Stego Image and Extract
        print("\n--- Extraction ---")
        loaded_stego_image = Image.open(stego_image_path).convert("RGB")
        extracted_qr_image = extract_qr_dct_wavelet(loaded_stego_image, alpha=ALPHA)

        # 5. Display Extracted QR Code
        if extracted_qr_image:
            print("\nExtracted QR code successfully!")
            extracted_qr_path = "extracted_qr.png"
            extracted_qr_image.save(extracted_qr_path)
            print(f"Extracted QR code saved to: {extracted_qr_path}")
            extracted_qr_image.show(title="Extracted QR Code")

            # Optional: Decode the extracted QR code to verify the hash
            try:
                from pyzbar import pyzbar

                decoded_objects = pyzbar.decode(Image.open(extracted_qr_path))
                if decoded_objects:
                    extracted_hash = decoded_objects[0].data.decode("utf-8")
                    print(f"Decoded Hash from QR: {extracted_hash}")
                    # Verify against original hash
                    original_hash = calculate_hash(original_image)
                    print(f"Original Hash       : {original_hash}")
                    if extracted_hash == original_hash:
                        print("SUCCESS: Extracted hash matches original hash!")
                    else:
                        print("FAILURE: Extracted hash does NOT match original hash.")
                else:
                    print("Could not decode the extracted QR code using pyzbar.")
            except ImportError:
                print(
                    "\nPyzbar not found. Cannot automatically decode extracted QR. Install `pip install pyzbar`."
                )
                print(
                    "Please use a QR scanner app/tool on 'extracted_qr.png' to verify the hash."
                )
        else:
            print("\nExtraction failed.")

    except FileNotFoundError:
        print(
            f"Error: Input image file not found at '{original_image_path}'. Please provide a valid image."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
