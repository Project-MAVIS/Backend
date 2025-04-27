import os
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from qreader import QReader

# import pdb

# pdb.set_trace()


import qr_code

ALPHA = 25
WAVELET = "db4"
SUBBAND = "HL"

INPUT_FOLDER = "benchmarking/input/images"
PAYLOAD_FOLDER = "benchmarking/input/strings"

QR_OUTPUT_FOLDER = "benchmarking/output/qr_code"
RS_OUTPUT_FOLDER = "benchmarking/output/rs"


def load_image(path):
    """Loads an image and converts to numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_image_pil(path):
    img = Image.open(path).convert("RGB")
    name = os.path.basename(path)
    return img, name


def calculate_psnr(origin, modified):
    """Calculate PSNR between two images."""
    return psnr(origin, modified, data_range=origin.max() - origin.min())


def calculate_ssim(origin, modified):
    """Calculate SSIM between two images."""
    # convert to grayscale
    from skimage.color import rgb2gray

    origin_gray = rgb2gray(origin)
    modified_gray = rgb2gray(modified)
    return ssim(
        origin_gray, modified_gray, data_range=origin_gray.max() - origin_gray.min()
    )


def bit_error_rate(original_bits, extracted_bits):
    """Calculate Bit Error Rate between two bit strings."""
    assert len(original_bits) == len(extracted_bits), "Lengths must match"
    errors = sum(o != e for o, e in zip(original_bits, extracted_bits))
    return errors / len(original_bits)


def string_to_bits(s):
    """Convert string to a list of bits."""
    return [int(bit) for char in s.encode("utf-8") for bit in format(char, "08b")]


def qr_code_watermark(original_image_path, payload, payload_path):
    image, name = load_image_pil(original_image_path)
    stego_image, _ = qr_code.embed_qr_dct_wavelet(
        original_image=image,
        alpha=float(ALPHA),
        wavelet_type=WAVELET,
        embed_subband=SUBBAND,
        payload=payload,
    )

    w_orig, h_orig = image.size
    stego_image = stego_image.resize((w_orig, h_orig), Image.Resampling.LANCZOS)
    stego_image.save(f"{QR_OUTPUT_FOLDER}/{name}_{payload_path}", format="PNG")
    return f"{QR_OUTPUT_FOLDER}/{name}"


def qr_code_extract_watermark(watermarked_image_path) -> str:
    image, _ = load_image_pil(watermarked_image_path)
    extracted_qr, _ = qr_code.extract_qr_dct_wavelet(
        stego_image=image,
        alpha=float(ALPHA),
        wavelet_type=WAVELET,
        embed_subband=SUBBAND,
    )
    qr_np_array = np.array(extracted_qr.convert("L"))
    reader = QReader()
    decoded_data_tuple = reader.detect_and_decode(image=qr_np_array)
    return decoded_data_tuple


def rs_watermark(original_image_path, payload, payload_path):
    pass


def rs_extract_watermark(original_image_path, payload):
    pass


def benchmark(
    original_img_path, watermarked_img_path, original_watermark, extracted_watermark
):
    # load images
    orig = load_image(original_img_path)
    water = load_image(watermarked_img_path)

    # image quality metrics
    p = calculate_psnr(orig, water)
    s = calculate_ssim(orig, water)

    # watermark accuracy
    orig_bits = string_to_bits(original_watermark)
    ext_bits = string_to_bits(extracted_watermark)

    ber = -1
    accuracy = -1
    try:
        ber = bit_error_rate(orig_bits, ext_bits)
        accuracy = 1 - ber
    except:
        pass

    # summary
    print(f"PSNR: {p:.2f} dB")
    print(f"SSIM: {s:.4f}")
    print(f"Bit Error Rate (BER): {ber:.4%}")
    print(f"Extraction Accuracy: {accuracy:.4%}")

    return {"psnr": p, "ssim": s, "ber": ber, "accuracy": accuracy}


APPROACHES = {
    "qr_code": {
        "watermark_fn": qr_code_watermark,
        "extract_fn": qr_code_extract_watermark,
        "output_dir": QR_OUTPUT_FOLDER,
    },
    # "reed_solomon": {
    #     "watermark_fn": rs_watermark,
    #     "extract_fn": rs_extract_watermark,
    #     "output_dir": RS_OUTPUT_FOLDER,
    # },
}


def get_outputs():
    for approach in APPROACHES.keys():
        print("Approach: ", approach)
        for image_path in Path(INPUT_FOLDER).iterdir():
            print("Image Path: ", image_path)
            for payload_path in Path(PAYLOAD_FOLDER).iterdir():
                print("Payload Path: ", payload_path)
                with open(payload_path, "r", encoding="utf-8") as f:
                    payload = f.read()

                output_img_path = APPROACHES[approach]["watermark_fn"](
                    image_path, payload, os.path.basename(payload_path)
                )
                extracted_data = APPROACHES[approach]["extract_fn"](output_img_path)
                benchmark(
                    image_path,
                    output_img_path,
                    payload,
                    "".join(filter(None, extracted_data)),
                )


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="Benchmark steganographic watermarking"
    # )
    # parser.add_argument("--orig", required=True, help="Path to original image")
    # parser.add_argument("--water", required=True, help="Path to watermarked image")
    # parser.add_argument("--orig_wm", required=True, help="Original watermark text")
    # parser.add_argument("--ext_wm", required=True, help="Extracted watermark text")
    # args = parser.parse_args()
    # benchmark(args.orig, args.water, args.orig_wm, args.ext_wm)

    get_outputs()
