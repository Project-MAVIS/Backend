import os
import numpy as np
from pathlib import Path
from PIL import Image
import io
import tempfile
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from skimage.color import rgb2gray
from qreader import QReader
import cv2  # Keep cv2 for potential RS implementation if needed later

# Assuming your existing QR code logic is in 'qr_code.py'
# Make sure qr_code.py functions accept/return PIL Images
import qr_code

# Placeholder for Reed-Solomon implementation if you add it
# import reedsolo_steg # Example name

# --- Configuration ---
# Default parameters (can be overridden by Gradio inputs)
DEFAULT_ALPHA = 25.0
DEFAULT_WAVELET = "db4"
DEFAULT_SUBBAND = "HL"

# --- Image Loading ---


def load_image_np(image_input):
    """Loads an image from various inputs (path, PIL, np) and returns NumPy array."""
    if isinstance(image_input, np.ndarray):
        # Assume already RGB if numpy
        return image_input
    elif isinstance(image_input, Image.Image):
        # Convert PIL to NumPy
        img = image_input.convert("RGB")
        return np.array(img)
    elif isinstance(image_input, (str, Path)):
        # Load from path
        img = Image.open(image_input).convert("RGB")
        return np.array(img)
    else:
        raise TypeError(
            "Invalid image input type. Expected path, PIL Image, or NumPy array."
        )


def load_image_pil(image_input):
    """Loads an image from various inputs and returns PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        # Convert NumPy to PIL
        return Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, (str, Path)):
        return Image.open(image_input).convert("RGB")
    else:
        raise TypeError(
            "Invalid image input type. Expected path, PIL Image, or NumPy array."
        )


# --- Metric Calculations ---


def calculate_psnr(origin_np, modified_np):
    """Calculate PSNR between two NumPy image arrays."""
    if origin_np.shape != modified_np.shape:
        # Basic resize if shapes differ - might affect metrics slightly
        h, w = origin_np.shape[:2]
        modified_np = np.array(
            Image.fromarray(modified_np).resize((w, h), Image.Resampling.LANCZOS)
        )

    # Ensure data types are suitable, handle potential range issues
    max_val = 255.0  # Assuming uint8 images scaled to float or compared directly
    return psnr_metric(origin_np, modified_np, data_range=max_val)


def calculate_ssim(origin_np, modified_np):
    """Calculate SSIM between two NumPy image arrays."""
    if origin_np.shape != modified_np.shape:
        h, w = origin_np.shape[:2]
        modified_np = np.array(
            Image.fromarray(modified_np).resize((w, h), Image.Resampling.LANCZOS)
        )

    # Convert to grayscale for SSIM calculation
    origin_gray = rgb2gray(origin_np)
    modified_gray = rgb2gray(modified_np)
    # Data range for grayscale float is typically 0-1
    return ssim_metric(origin_gray, modified_gray, data_range=1.0)


def string_to_bits(s: str) -> list[int]:
    """Convert string to a list of bits (0s or 1s)."""
    if not isinstance(s, str):
        return []  # Return empty list if input is not a string
    try:
        return [int(bit) for char in s.encode("utf-8") for bit in format(char, "08b")]
    except Exception:
        return []  # Handle potential encoding errors


def bit_error_rate(original_bits: list[int], extracted_bits: list[int]) -> float:
    """Calculate Bit Error Rate between two bit lists."""
    len_orig = len(original_bits)
    len_extr = len(extracted_bits)

    if len_orig == 0:
        return (
            1.0 if len_extr > 0 else 0.0
        )  # BER is 1 if lengths mismatch and orig is empty, 0 if both empty

    # Handle unequal lengths - compare up to the shorter length and penalize difference
    min_len = min(len_orig, len_extr)
    errors = sum(original_bits[i] != extracted_bits[i] for i in range(min_len))
    # Add errors for the length difference
    errors += abs(len_orig - len_extr)

    # Calculate BER based on the original length
    return errors / len_orig


# --- Watermarking Method Implementations ---


# Wrapper for QR Code Embedding
def qr_code_watermark_wrapper(
    original_image_pil: Image.Image, payload: str, **kwargs
) -> (Image.Image | None, str):
    """
    Wrapper for qr_code.embed_qr_dct_wavelet.
    Accepts PIL Image, payload string, and parameters.
    Returns PIL Image and status string.
    """
    try:
        # Extract specific params or use defaults
        alpha = float(kwargs.get("alpha", DEFAULT_ALPHA))
        wavelet = kwargs.get("wavelet", DEFAULT_WAVELET)
        subband = kwargs.get("subband", DEFAULT_SUBBAND)

        # Call the actual embedding function (ensure it accepts payload)
        # NOTE: You might need to modify your qr_code.embed_qr_dct_wavelet
        # to accept a 'payload' argument directly instead of calculating hash inside.
        # For now, assuming it does:
        stego_image, status = qr_code.embed_qr_dct_wavelet(
            original_image=original_image_pil,
            alpha=alpha,
            wavelet_type=wavelet,
            embed_subband=subband,
            # --- IMPORTANT ---
            # Pass the payload directly if your function supports it
            # If not, you'll need to modify qr_code.embed_qr_dct_wavelet
            # to take payload instead of calculating hash internally.
            payload=payload,  # Assuming function accepts 'payload_str'
            # --- ---
        )
        return stego_image, status
    except Exception as e:
        return None, f"Error during QR Code embedding: {str(e)}"


# Wrapper for QR Code Extraction
def qr_code_extract_wrapper(
    watermarked_image_pil: Image.Image, **kwargs
) -> (str | None, str):
    """
    Wrapper for qr_code.extract_qr_dct_wavelet and QReader decoding.
    Accepts PIL Image and parameters.
    Returns extracted string payload and status string.
    """
    extracted_payload_str = None
    status_list = []
    try:
        # Extract specific params or use defaults
        alpha = float(kwargs.get("alpha", DEFAULT_ALPHA))
        wavelet = kwargs.get("wavelet", DEFAULT_WAVELET)
        subband = kwargs.get("subband", DEFAULT_SUBBAND)

        # Call the actual extraction function
        extracted_qr_image, extract_status = qr_code.extract_qr_dct_wavelet(
            stego_image=watermarked_image_pil,
            alpha=alpha,
            wavelet_type=wavelet,
            embed_subband=subband,
        )
        status_list.append(extract_status)

        if extracted_qr_image:
            # Decode the extracted QR image
            try:
                qr_np_array = np.array(extracted_qr_image.convert("L"))
                reader = QReader()
                # Use detect_and_decode which returns a tuple of decoded strings
                decoded_data_tuple = reader.detect_and_decode(image=qr_np_array)

                # Filter out None values and join potentially multiple decoded results
                # (usually only one QR code is expected)
                valid_decoded = [s for s in decoded_data_tuple if s is not None]
                if valid_decoded:
                    extracted_payload_str = "\n---\n".join(valid_decoded)
                    status_list.append("QR Code decoded successfully.")
                else:
                    status_list.append(
                        "QR Code detected, but content decoding failed or was empty."
                    )
            except Exception as decode_error:
                status_list.append(f"Error during QR decoding: {decode_error}")
        else:
            status_list.append("QR Code image extraction failed.")

    except Exception as e:
        status_list.append(f"Error during QR Code extraction process: {str(e)}")

    return extracted_payload_str, "\n".join(status_list)


# --- Placeholder for Reed-Solomon ---
# def rs_watermark_wrapper(original_image_pil: Image.Image, payload: str, **kwargs) -> (Image.Image | None, str):
#     try:
#         strength = int(kwargs.get('strength', 100)) # Example RS param
#         # Convert payload string to bytes
#         payload_bytes = payload.encode('utf-8')
#
#         # --- Call your reedsolo_steg.embed_payload ---
#         # IMPORTANT: Modify reedsolo_steg.embed_payload to:
#         # 1. Accept PIL image input.
#         # 2. Return the modified PIL image directly instead of saving to file.
#         # Example call (adjust based on your actual function):
#         # embedded_pil_image = reedsolo_steg.embed_payload(original_image_pil, payload_bytes, strength)
#         # ---
#
#         # Placeholder return
#         embedded_pil_image = None # Replace with actual call result
#         if embedded_pil_image:
#            return embedded_pil_image, "RS Embedding successful (placeholder)."
#         else:
#            return None, "RS Embedding failed (placeholder)."
#
#     except Exception as e:
#         return None, f"Error during RS embedding: {str(e)}"
#
# def rs_extract_wrapper(watermarked_image_pil: Image.Image, **kwargs) -> (str | None, str):
#     try:
#         strength = int(kwargs.get('strength', 100)) # Example RS param
#
#         # --- Call your reedsolo_steg.extract_payload ---
#         # IMPORTANT: Modify reedsolo_steg.extract_payload to:
#         # 1. Accept PIL image input.
#         # 2. Return the extracted payload as bytes.
#         # Example call (adjust based on your actual function):
#         # extracted_bytes = reedsolo_steg.extract_payload(watermarked_image_pil, strength)
#         # ---
#
#         # Placeholder return
#         extracted_bytes = None # Replace with actual call result
#         if extracted_bytes:
#             try:
#                 # Attempt to decode bytes back to string
#                 extracted_payload_str = extracted_bytes.decode('utf-8')
#                 return extracted_payload_str, "RS Extraction successful (placeholder)."
#             except UnicodeDecodeError:
#                 return None, "RS Extraction successful, but failed to decode bytes to UTF-8 string."
#         else:
#             return None, "RS Extraction failed (placeholder)."
#
#     except Exception as e:
#         return None, f"Error during RS extraction: {str(e)}"

# --- Benchmarking Function ---


def benchmark(
    orig_img_np: np.ndarray,
    water_img_np: np.ndarray,
    original_payload: str,
    extracted_payload: str | None,
) -> dict:
    """Calculates PSNR, SSIM, and BER metrics."""
    results = {"psnr": None, "ssim": None, "ber": None, "accuracy": None, "error": None}
    try:
        # Image quality metrics
        results["psnr"] = calculate_psnr(orig_img_np, water_img_np)
        results["ssim"] = calculate_ssim(orig_img_np, water_img_np)

        # Watermark accuracy
        if extracted_payload is None:
            # If extraction failed completely, BER is 100%
            results["ber"] = 1.0
            results["accuracy"] = 0.0
        else:
            orig_bits = string_to_bits(original_payload)
            ext_bits = string_to_bits(extracted_payload)  # Handle potential non-string

            # Calculate BER, handle potential errors
            if not orig_bits and not ext_bits:  # Both empty strings
                results["ber"] = 0.0
            elif not orig_bits:  # Original empty, extracted not
                results["ber"] = (
                    1.0  # Treat as 100% error if extracted something from nothing
                )
            else:
                results["ber"] = bit_error_rate(orig_bits, ext_bits)

            if results["ber"] is not None:
                results["accuracy"] = 1.0 - results["ber"]

    except Exception as e:
        print(f"Error during benchmarking: {e}")
        results["error"] = str(e)

    return results


# --- Approach Definitions ---
# This dictionary maps approach names to their functions.
# Add your other methods here following the same pattern.
APPROACHES = {
    "qr_code_wavelet_dct": {
        "watermark_fn": qr_code_watermark_wrapper,
        "extract_fn": qr_code_extract_wrapper,
        # Add default parameters specific to this method if needed
        "params": {
            "alpha": DEFAULT_ALPHA,
            "wavelet": DEFAULT_WAVELET,
            "subband": DEFAULT_SUBBAND,
        },
    },
    # "reed_solomon_dct": { # Uncomment and implement when ready
    #     "watermark_fn": rs_watermark_wrapper,
    #     "extract_fn": rs_extract_wrapper,
    #     "params": {
    #         "strength": 100 # Example parameter for RS method
    #     }
    # },
}
