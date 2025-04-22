# steganography.py
import io
import hashlib
import numpy as np
from PIL import Image
import pywt
from scipy.fftpack import dct, idct
import pyqrcode
import math
import struct  # For packing dimensions

# --- Configuration & Robustness Parameters ---
DCT_BLOCK_SIZE = 8
# Choose lower mid-frequency coefficients within the 8x8 DCT block.
# These are generally more robust to JPEG than higher frequencies.
# Avoid (0,0) DC component and very low frequencies like (0,1), (1,0), (1,1).
DCT_COEFF_INDICES = [
    (2, 1),
    (1, 2),  # Lower-mid frequencies
    (3, 2),
    (2, 3),  # Mid frequencies
    (4, 1),
    (1, 4),  # Slightly higher-mid
]
# Increase capacity by using more coefficients per block.

# --- Robustness Enhancement: Redundancy ---
# Embed each bit multiple times. Must be odd to avoid ties in majority vote.
REPETITIONS = 3


# --- Helper Functions (dct2, idct2, rgb2ycbcr, ycbcr2rgb, int_to_bits, bits_to_int, calculate_hash - remain the same) ---
def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct2(block):
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def rgb2ycbcr(im_rgb):
    im_np = np.array(im_rgb).astype(float)
    r, g, b = im_np[:, :, 0], im_np[:, :, 1], im_np[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def ycbcr2rgb(im_ycbcr):
    y, cb, cr = im_ycbcr[:, :, 0], im_ycbcr[:, :, 1], im_ycbcr[:, :, 2]
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    im_np = np.stack([r, g, b], axis=-1)
    im_np = np.clip(im_np, 0, 255).astype(np.uint8)
    return Image.fromarray(im_np)


def int_to_bits(n, num_bits):
    return [(n >> i) & 1 for i in range(num_bits - 1, -1, -1)]


def bits_to_int(bits):
    n = 0
    for bit in bits:
        n = (n << 1) | bit
    return n


def calculate_hash(image: Image.Image) -> str:
    if image is None:
        return ""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        hasher = hashlib.sha256()
        hasher.update(img_byte_arr)
        return hasher.hexdigest()
    except Exception:
        return ""


def generate_binary_qr_data(data: str) -> (np.ndarray | None, tuple | None, str):
    if not data:
        return None, None, "Error: No data provided for QR code generation."
    try:
        qr = pyqrcode.create(data, error="H")
        qr_text = qr.text(quiet_zone=1)
        qr_lines = qr_text.strip().split("\n")
        qr_dim = len(qr_lines)
        qr_matrix = np.zeros((qr_dim, qr_dim), dtype=int)
        for r, line in enumerate(qr_lines):
            for c, char in enumerate(line):
                qr_matrix[r, c] = 1 if char == "1" else 0
        qr_flat_bits = qr_matrix.flatten()
        status = (
            f"Generated QR Code for data: '{data[:10]}...'\n"
            f"QR Code Dimensions: {qr_dim}x{qr_dim}\n"
            f"Total QR bits: {len(qr_flat_bits)}"
        )
        return qr_flat_bits, (qr_dim, qr_dim), status
    except Exception as e:
        return None, None, f"Error generating QR code: {e}"


# --- Majority Vote Helper ---
def majority_vote(bits):
    """Performs majority vote on a list of bits. Assumes odd length."""
    if not bits:
        # This case should ideally not happen if logic is correct
        print("Warning: Majority vote called with empty list.")
        return 0  # Default or raise error
    zeros = bits.count(0)
    ones = len(bits) - zeros
    # If REPETITIONS is odd, zeros != ones.
    return 1 if ones > zeros else 0


# --- Embedding Function (Enhanced for Robustness) ---


def embed_qr_dct_wavelet(
    original_image: Image.Image, alpha: float, wavelet_type: str, embed_subband: str
) -> (Image.Image | None, str):
    if original_image is None:
        return None, "Error: No original image provided."
    try:
        original_image = original_image.convert("RGB")
    except Exception as e:
        return None, f"Error converting image to RGB: {e}"

    w_orig, h_orig = original_image.size
    min_dim = 2 * DCT_BLOCK_SIZE
    w_new = (w_orig // min_dim) * min_dim
    h_new = (h_orig // min_dim) * min_dim
    if w_new == 0 or h_new == 0:
        return None, f"Error: Image dimensions ({w_orig}x{h_orig}) too small."
    if w_new != w_orig or h_new != h_orig:
        print(f"Warning: Resizing image from {w_orig}x{h_orig} to {w_new}x{h_new}.")
        try:
            original_image = original_image.resize(
                (w_new, h_new), Image.Resampling.LANCZOS
            )
        except Exception as e:
            return None, f"Error resizing image: {e}"

    status_msgs = [f"Robustness Enhancement: Repetition factor = {REPETITIONS}"]
    try:
        image_hash = calculate_hash(original_image)
        if not image_hash:
            return None, "Error calculating image hash."
        status_msgs.append(f"Image Hash: {image_hash[:16]}...")

        qr_bits, qr_dims, qr_status = generate_binary_qr_data(image_hash)
        status_msgs.append(qr_status)
        if qr_bits is None:
            return None, "\n".join(status_msgs)
        qr_width, qr_height = qr_dims

        dim_bits_w = int_to_bits(qr_width, 16)
        dim_bits_h = int_to_bits(qr_height, 16)
        # Combine dimension bits and QR data bits
        all_data_bits = dim_bits_w + dim_bits_h + list(qr_bits)
        num_unique_bits = len(all_data_bits)

        # Calculate total raw bits needed including repetitions
        total_raw_bits_to_embed = num_unique_bits * REPETITIONS
        status_msgs.append(f"Unique bits (dims + data): {num_unique_bits}")
        status_msgs.append(
            f"Total raw bits to embed (with repetition): {total_raw_bits_to_embed}"
        )

        img_ycbcr = rgb2ycbcr(original_image)
        y_channel, cb_channel, cr_channel = (
            img_ycbcr[:, :, 0],
            img_ycbcr[:, :, 1],
            img_ycbcr[:, :, 2],
        )

        coeffs = pywt.dwt2(y_channel, wavelet_type)
        LL, (LH, HL, HH) = coeffs

        subband_map = {"LH": LH, "HL": HL, "HH": HH}
        if embed_subband not in subband_map:
            return None, f"Error: Invalid subband '{embed_subband}' specified."
        embed_coeffs = subband_map[embed_subband]

        sub_h, sub_w = embed_coeffs.shape
        max_blocks = (sub_h // DCT_BLOCK_SIZE) * (sub_w // DCT_BLOCK_SIZE)
        capacity = max_blocks * len(DCT_COEFF_INDICES)
        status_msgs.append(
            f"Embedding in subband: {embed_subband} (shape: {embed_coeffs.shape})"
        )
        status_msgs.append(f"DCT Block Size: {DCT_BLOCK_SIZE}x{DCT_BLOCK_SIZE}")
        status_msgs.append(f"Coefficients used per block: {len(DCT_COEFF_INDICES)}")
        status_msgs.append(f"Available embedding capacity (raw bits): {capacity}")

        if total_raw_bits_to_embed > capacity:
            error_msg = (
                f"Data size with redundancy ({total_raw_bits_to_embed} raw bits) exceeds capacity ({capacity} bits). "
                "Try a larger image, decrease repetitions (code), or use fewer QR bits (less data/lower ECC)."
            )
            status_msgs.append(f"Error: {error_msg}")
            return None, "\n".join(status_msgs)

        # --- Embedding Loop (with Repetition) ---
        raw_bit_idx = 0
        embedded_coeffs = embed_coeffs.copy()
        coeff_indices_iter = iter(
            DCT_COEFF_INDICES
        )  # Cycle through coeffs within a block

        # Iterate through blocks
        for r in range(0, sub_h - DCT_BLOCK_SIZE + 1, DCT_BLOCK_SIZE):
            for c in range(0, sub_w - DCT_BLOCK_SIZE + 1, DCT_BLOCK_SIZE):
                if raw_bit_idx >= total_raw_bits_to_embed:
                    break

                block = embedded_coeffs[r : r + DCT_BLOCK_SIZE, c : c + DCT_BLOCK_SIZE]
                dct_block = dct2(block)

                # Try to embed multiple bits (from repetitions) into coefficients of this block
                num_bits_embedded_in_block = 0
                while (
                    num_bits_embedded_in_block < len(DCT_COEFF_INDICES)
                    and raw_bit_idx < total_raw_bits_to_embed
                ):
                    try:
                        # Get the coefficient index for this bit
                        coeff_r, coeff_c = next(coeff_indices_iter)
                    except StopIteration:  # Reached end of coeff list for this block
                        coeff_indices_iter = iter(
                            DCT_COEFF_INDICES
                        )  # Reset for next block
                        break  # Move to next block

                    # Determine the original data bit value (handling repetition)
                    unique_bit_index = raw_bit_idx // REPETITIONS
                    bit = all_data_bits[unique_bit_index]

                    # Apply modification
                    modification = alpha * (bit * 2 - 1)
                    dct_block[coeff_r, coeff_c] += modification

                    raw_bit_idx += 1
                    num_bits_embedded_in_block += 1

                # Reset coefficient iterator if we finished embedding in this block
                # This ensures the next block starts with the first coefficient index again
                if (
                    num_bits_embedded_in_block == len(DCT_COEFF_INDICES)
                    or raw_bit_idx >= total_raw_bits_to_embed
                ):
                    coeff_indices_iter = iter(DCT_COEFF_INDICES)

                modified_block = idct2(dct_block)
                embedded_coeffs[r : r + DCT_BLOCK_SIZE, c : c + DCT_BLOCK_SIZE] = (
                    modified_block
                )

            if raw_bit_idx >= total_raw_bits_to_embed:
                break
        # --- End Embedding Loop ---

        status_msgs.append(f"Finished embedding {raw_bit_idx} raw bits.")
        if raw_bit_idx < total_raw_bits_to_embed:
            status_msgs.append(
                f"Warning: Potential issue embedding all data bits (expected {total_raw_bits_to_embed})."
            )

        coeffs_modified_map = {
            "LH": (embedded_coeffs, HL, HH),
            "HL": (LH, embedded_coeffs, HH),
            "HH": (LH, HL, embedded_coeffs),
        }
        coeffs_modified = LL, coeffs_modified_map[embed_subband]
        y_channel_modified = pywt.idwt2(coeffs_modified, wavelet_type)

        h_y_orig, w_y_orig = y_channel.shape
        y_channel_modified = y_channel_modified[:h_y_orig, :w_y_orig]

        stego_img_ycbcr = np.stack(
            [y_channel_modified, cb_channel, cr_channel], axis=-1
        )
        stego_image_pil = ycbcr2rgb(stego_img_ycbcr)

        status_msgs.append("Embedding process completed successfully.")
        return stego_image_pil, "\n".join(status_msgs)

    except pywt.InvalidWavelet:
        return None, f"Error: Invalid wavelet type '{wavelet_type}'."
    except Exception as e:
        import traceback

        status_msgs.append(f"An unexpected error occurred during embedding: {e}")
        # status_msgs.append(traceback.format_exc()) # Uncomment for detailed debug
        return None, "\n".join(status_msgs)


# --- Extraction Function (Enhanced for Robustness) ---


def extract_qr_dct_wavelet(
    stego_image: Image.Image,
    alpha: float,  # Less critical for sign-based extraction, but kept for signature
    wavelet_type: str,
    embed_subband: str,
) -> (Image.Image | None, str):
    if stego_image is None:
        return None, "Error: No stego image provided."
    try:
        stego_image = stego_image.convert("RGB")
    except Exception as e:
        return None, f"Error converting image to RGB: {e}"

    status_msgs = [f"Robustness Enhancement: Using Repetition factor = {REPETITIONS}"]
    try:
        img_ycbcr = rgb2ycbcr(stego_image)
        y_channel = img_ycbcr[:, :, 0]

        coeffs = pywt.dwt2(y_channel, wavelet_type)
        LL, (LH, HL, HH) = coeffs

        subband_map = {"LH": LH, "HL": HL, "HH": HH}
        if embed_subband not in subband_map:
            return (
                None,
                f"Error: Invalid subband '{embed_subband}' specified for extraction.",
            )
        embed_coeffs = subband_map[embed_subband]

        sub_h, sub_w = embed_coeffs.shape
        status_msgs.append(
            f"Extracting from subband: {embed_subband} (shape: {embed_coeffs.shape})"
        )
        status_msgs.append(
            f"Using Wavelet: {wavelet_type}, DCT Block Size: {DCT_BLOCK_SIZE}"
        )

        # --- Extract ALL potentially embedded raw bits ---
        raw_extracted_bits = []
        max_possible_bits = (
            (sub_h // DCT_BLOCK_SIZE)
            * (sub_w // DCT_BLOCK_SIZE)
            * len(DCT_COEFF_INDICES)
        )
        coeff_indices_iter = iter(DCT_COEFF_INDICES)  # Cycle through coeffs

        for r in range(0, sub_h - DCT_BLOCK_SIZE + 1, DCT_BLOCK_SIZE):
            for c in range(0, sub_w - DCT_BLOCK_SIZE + 1, DCT_BLOCK_SIZE):
                if len(raw_extracted_bits) >= max_possible_bits:
                    break  # Safety break

                block = embed_coeffs[r : r + DCT_BLOCK_SIZE, c : c + DCT_BLOCK_SIZE]
                dct_block = dct2(block)

                num_bits_extracted_in_block = 0
                while num_bits_extracted_in_block < len(DCT_COEFF_INDICES):
                    try:
                        coeff_r, coeff_c = next(coeff_indices_iter)
                    except StopIteration:
                        coeff_indices_iter = iter(DCT_COEFF_INDICES)  # Reset
                        break  # Move to next block

                    if len(raw_extracted_bits) < max_possible_bits:
                        coeff_val = dct_block[coeff_r, coeff_c]
                        # Simple sign-based detection (assumes alpha modification dominates)
                        bit = 1 if coeff_val > 0 else 0
                        raw_extracted_bits.append(bit)
                        num_bits_extracted_in_block += 1
                    else:
                        break  # Reached max capacity

                if (
                    num_bits_extracted_in_block == len(DCT_COEFF_INDICES)
                    or len(raw_extracted_bits) >= max_possible_bits
                ):
                    coeff_indices_iter = iter(DCT_COEFF_INDICES)  # Reset for next block

            if len(raw_extracted_bits) >= max_possible_bits:
                break
        # --- End Raw Bit Extraction ---

        status_msgs.append(f"Total raw bits extracted: {len(raw_extracted_bits)}")

        # --- Decode Dimensions using Majority Vote ---
        dim_bits_needed = 32
        raw_dim_bits_needed = dim_bits_needed * REPETITIONS
        if len(raw_extracted_bits) < raw_dim_bits_needed:
            status_msgs.append(
                f"Error: Insufficient raw bits ({len(raw_extracted_bits)}) extracted to recover dimensions (needed {raw_dim_bits_needed})."
            )
            return None, "\n".join(status_msgs)

        decoded_dim_bits = []
        for i in range(dim_bits_needed):
            bit_repetitions = raw_extracted_bits[
                i * REPETITIONS : (i + 1) * REPETITIONS
            ]
            decoded_dim_bits.append(majority_vote(bit_repetitions))

        try:
            qr_width = bits_to_int(decoded_dim_bits[0:16])
            qr_height = bits_to_int(decoded_dim_bits[16:32])
        except Exception as dim_e:
            status_msgs.append(
                f"Error converting voted dimension bits to integers: {dim_e}"
            )
            return None, "\n".join(status_msgs)

        # Sanity check dimensions
        if (
            qr_width <= 0 or qr_height <= 0 or qr_width > 500 or qr_height > 500
        ):  # Increased sanity limit
            status_msgs.append(
                f"Error: Extracted invalid dimensions ({qr_width}x{qr_height}) after majority vote. Possible data corruption."
            )
            return None, "\n".join(status_msgs)

        status_msgs.append(
            f"Recovered QR Dimensions (post-vote): {qr_width}x{qr_height}"
        )

        # --- Decode QR Data using Majority Vote ---
        num_qr_data_bits = qr_width * qr_height
        total_unique_bits = dim_bits_needed + num_qr_data_bits
        total_raw_bits_expected = total_unique_bits * REPETITIONS

        if len(raw_extracted_bits) < total_raw_bits_expected:
            status_msgs.append(
                f"Error: Insufficient raw bits ({len(raw_extracted_bits)}) for full QR data recovery (expected {total_raw_bits_expected})."
            )
            return None, "\n".join(status_msgs)

        final_qr_bits = []
        qr_data_start_index = raw_dim_bits_needed  # Start after dimension bits
        for i in range(num_qr_data_bits):
            bit_repetitions = raw_extracted_bits[
                qr_data_start_index
                + i * REPETITIONS : qr_data_start_index
                + (i + 1) * REPETITIONS
            ]
            # Check length just in case, though prior check should cover it
            if len(bit_repetitions) == REPETITIONS:
                final_qr_bits.append(majority_vote(bit_repetitions))
            else:
                status_msgs.append(
                    f"Error: Unexpected missing bits during QR data majority vote at index {i}."
                )
                return None, "\n".join(status_msgs)  # Abort if structure is wrong

        # --- Reconstruct QR Code ---
        if len(final_qr_bits) != num_qr_data_bits:
            status_msgs.append(
                f"Error: Final QR bits count ({len(final_qr_bits)}) mismatch expected ({num_qr_data_bits})."
            )
            return None, "\n".join(status_msgs)

        qr_matrix = np.array(final_qr_bits, dtype=np.uint8).reshape(
            (qr_height, qr_width)
        )

        scale_factor = 5
        qr_image_np = np.kron(
            qr_matrix, np.ones((scale_factor, scale_factor), dtype=np.uint8)
        )
        qr_image_np = (qr_image_np * 255).astype(np.uint8)
        border_size = 2 * scale_factor
        qr_image_bordered = np.pad(
            qr_image_np, pad_width=border_size, mode="constant", constant_values=255
        )
        if qr_image_bordered.dtype != np.uint8:
            qr_image_bordered = qr_image_bordered.astype(np.uint8)
        qr_image_pil = Image.fromarray(qr_image_bordered)

        status_msgs.append("Extraction successful using majority vote.")
        return qr_image_pil, "\n".join(status_msgs)

    except pywt.InvalidWavelet:
        return None, f"Error: Invalid wavelet type '{wavelet_type}'."
    except Exception as e:
        import traceback

        status_msgs.append(f"An unexpected error occurred during extraction: {e}")
        # status_msgs.append(traceback.format_exc()) # Uncomment for detailed debug
        return None, "\n".join(status_msgs)
