import gradio as gr
import pywt
from PIL import Image
import numpy as np
import os
import io  # Needed for in-memory compression

# Import the backend functions
import steganography as steg

# Try importing QReader and handle potential import error
try:
    from qreader import QReader

    qreader_available = True
except ImportError:
    print("Warning: QReader library not found. QR decoding will be disabled.")
    print("Install it using: pip install qreader opencv-python")
    QReader = None  # Define QReader as None if import fails
    qreader_available = False


# --- Configuration for Gradio ---
DEFAULT_ALPHA = 10.0
DEFAULT_WAVELET = "db4"
DEFAULT_SUBBAND = "HL"
DEFAULT_JPEG_QUALITY = 75  # Default quality for JPEG test

# Get a list of available discrete wavelets from PyWavelets
available_wavelets = pywt.wavelist(kind="discrete")
common_wavelets = [
    w
    for w in available_wavelets
    if len(w) < 6 and not w.startswith("sym") or w in ["sym2", "sym3", "sym4"]
]
if DEFAULT_WAVELET not in common_wavelets:
    common_wavelets.insert(0, DEFAULT_WAVELET)

available_subbands = ["HL", "LH", "HH"]

# --- Gradio Interface Functions ---


def embed_interface(image, alpha, wavelet, subband):
    """Gradio wrapper for embedding."""
    if image is None:
        return None, "Please upload an image first."
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(image)
        except Exception as e:
            return None, f"Error processing input image: {e}"

    stego_image, status = steg.embed_qr_dct_wavelet(
        original_image=image,
        alpha=float(alpha),
        wavelet_type=wavelet,
        embed_subband=subband,
    )
    # Add note about saving as PNG
    if stego_image is not None:
        status += "\n\nRecommendation: Download this image and save as PNG for lossless storage before compression testing or extraction."
    return stego_image, status


def extract_interface(stego_image, alpha, wavelet, subband):
    """Gradio wrapper for extraction AND decoding."""
    decoded_text_output = "QR Decoding disabled: QReader library not found."
    if stego_image is None:
        return None, "Please upload the stego image first.", decoded_text_output

    if not isinstance(stego_image, Image.Image):
        try:
            stego_image = Image.fromarray(stego_image)
        except Exception as e:
            return None, f"Error processing input stego image: {e}", decoded_text_output

    # Call the backend extraction function
    extracted_qr_image, status = steg.extract_qr_dct_wavelet(
        stego_image=stego_image,
        alpha=float(alpha),
        wavelet_type=wavelet,
        embed_subband=subband,
    )

    # --- Add QR Decoding Step ---
    if extracted_qr_image is not None and qreader_available:
        try:
            qr_np_array = np.array(extracted_qr_image.convert("L"))
            reader = QReader()
            decoded_data_tuple = reader.detect_and_decode(image=qr_np_array)
            if decoded_data_tuple and decoded_data_tuple[0] is not None:
                decoded_text_output = "\n---\n".join(filter(None, decoded_data_tuple))
                status += "\nQR Code decoded successfully."
            elif decoded_data_tuple:
                decoded_text_output = "QR Code detected, but content decoding failed."
                status += "\nQR Code detected, but content decoding failed."
            else:
                decoded_text_output = (
                    "No QR Code detected or decoded in the extracted image."
                )
                status += "\nNo QR Code found in the extracted image."
        except Exception as decode_error:
            error_msg = f"Error during QR decoding: {decode_error}"
            print(error_msg)
            status += f"\n{error_msg}"
            decoded_text_output = error_msg
    elif not qreader_available and extracted_qr_image is not None:
        status += "\nQR decoding skipped (QReader not installed)."
    elif extracted_qr_image is None:
        decoded_text_output = "Extraction failed, cannot decode QR."

    return extracted_qr_image, status, decoded_text_output


# --- NEW: JPEG Compression Function ---
def compress_image_jpeg(input_image, quality):
    """Applies JPEG compression to an image and returns it + info."""
    if input_image is None:
        return None, "Please upload an image first."

    # Ensure input is a PIL Image
    if not isinstance(input_image, Image.Image):
        try:
            input_image = Image.fromarray(input_image)
        except Exception as e:
            return None, f"Error processing input image: {e}"

    # Ensure image is in RGB mode for JPEG saving
    try:
        input_image = input_image.convert("RGB")
    except Exception as e:
        return None, f"Error converting image to RGB for JPEG: {e}"

    info_str = ""
    compressed_image = None

    try:
        # 1. Get approximate original size (by saving as PNG in memory)
        png_buffer = io.BytesIO()
        input_image.save(png_buffer, format="PNG")
        original_size = len(png_buffer.getvalue())
        info_str += f"Approx. Original (PNG) Size: {original_size / 1024:.2f} KB\n"

        # 2. Compress to JPEG in memory
        jpeg_buffer = io.BytesIO()
        input_image.save(
            jpeg_buffer, format="JPEG", quality=int(quality), optimize=True
        )  # Use optimize
        compressed_size = len(jpeg_buffer.getvalue())
        info_str += f"Compressed (JPEG Q={int(quality)}) Size: {compressed_size / 1024:.2f} KB\n"

        # 3. Calculate ratio
        ratio = (compressed_size / original_size) * 100 if original_size > 0 else 0
        info_str += f"Compression Ratio (JPEG Size / PNG Size): {ratio:.2f}%\n"

        # 4. Reload the compressed image from buffer to display in Gradio
        jpeg_buffer.seek(0)  # Reset buffer pointer to the beginning
        compressed_image = Image.open(jpeg_buffer)
        info_str += "Compression successful."

    except Exception as e:
        info_str = f"Error during compression: {e}"
        compressed_image = None  # Ensure no image is returned on error

    return compressed_image, info_str


# --- Build Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Image Steganography: Wavelet-DCT QR Code Embedding & Testing
        Embed, compress, and extract QR codes hidden in images.
        **Important:** Extraction requires the *exact same* parameters (Alpha, Wavelet, Subband) used during embedding. Compression (especially low quality JPEG) may destroy the hidden data.
        """
    )

    with gr.Tabs():
        # --- Embed Tab ---
        with gr.TabItem("1. Embed QR Code"):
            with gr.Row():
                with gr.Column(scale=1):
                    embed_input_image = gr.Image(
                        type="pil", label="Upload Original Image"
                    )
                    embed_alpha = gr.Slider(
                        minimum=1.0,
                        maximum=50.0,
                        step=0.5,
                        value=DEFAULT_ALPHA,
                        label="Embedding Strength (Alpha)",
                    )
                    embed_wavelet = gr.Dropdown(
                        choices=common_wavelets,
                        value=DEFAULT_WAVELET,
                        label="Wavelet Type",
                    )
                    embed_subband = gr.Radio(
                        choices=available_subbands,
                        value=DEFAULT_SUBBAND,
                        label="Subband for Embedding",
                    )
                    embed_button = gr.Button("Embed QR Code", variant="primary")
                with gr.Column(scale=1):
                    embed_output_image = gr.Image(
                        type="pil", label="Stego Image (with embedded QR)"
                    )
                    embed_status = gr.Textbox(
                        label="Status / Log", lines=10, interactive=False
                    )

        # --- Compress Tab (NEW) ---
        with gr.TabItem("2. JPEG Compression Test"):
            with gr.Row():
                with gr.Column(scale=1):
                    compress_input_image = gr.Image(
                        type="pil", label="Upload Stego Image (PNG Recommended)"
                    )
                    compress_quality = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=DEFAULT_JPEG_QUALITY,
                        label="JPEG Quality (1=Max Compression, 100=Max Quality)",
                    )
                    compress_button = gr.Button("Compress Image", variant="primary")
                with gr.Column(scale=1):
                    compress_output_image = gr.Image(
                        type="pil", label="Compressed Image (JPEG)"
                    )
                    compress_status = gr.Textbox(
                        label="Compression Info", lines=6, interactive=False
                    )
            gr.Markdown(
                "*(Use this compressed image in the 'Extract' tab to test robustness)*"
            )

        # --- Extract Tab ---
        with gr.TabItem("3. Extract QR Code"):
            with gr.Row():
                with gr.Column(scale=1):
                    extract_input_image = gr.Image(
                        type="pil",
                        label="Upload Stego Image (Original PNG or Compressed JPEG)",
                    )
                    gr.Markdown("👇 **Use the SAME parameters as embedding!** 👇")
                    extract_alpha = gr.Slider(
                        minimum=1.0,
                        maximum=50.0,
                        step=0.5,
                        value=DEFAULT_ALPHA,
                        label="Embedding Strength (Alpha)",
                    )
                    extract_wavelet = gr.Dropdown(
                        choices=common_wavelets,
                        value=DEFAULT_WAVELET,
                        label="Wavelet Type",
                    )
                    extract_subband = gr.Radio(
                        choices=available_subbands,
                        value=DEFAULT_SUBBAND,
                        label="Subband Used for Embedding",
                    )
                    extract_button = gr.Button("Extract QR Code", variant="primary")
                with gr.Column(scale=1):
                    extract_output_image = gr.Image(
                        type="pil", label="Extracted QR Code Image"
                    )
                    extract_decoded_text = gr.Textbox(
                        label="Decoded QR Content", lines=3, interactive=False
                    )
                    extract_status = gr.Textbox(
                        label="Status / Log", lines=7, interactive=False
                    )

    # --- Connect Components ---
    embed_button.click(
        fn=embed_interface,
        inputs=[embed_input_image, embed_alpha, embed_wavelet, embed_subband],
        outputs=[embed_output_image, embed_status],
    )

    # NEW: Connect compression button
    compress_button.click(
        fn=compress_image_jpeg,
        inputs=[compress_input_image, compress_quality],
        outputs=[compress_output_image, compress_status],
    )

    extract_button.click(
        fn=extract_interface,
        inputs=[extract_input_image, extract_alpha, extract_wavelet, extract_subband],
        outputs=[extract_output_image, extract_status, extract_decoded_text],
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    demo.launch()
