import gradio as gr
from PIL import Image
import numpy as np
import tempfile
import os
import json  # For displaying metrics dict

# Import the utility functions and definitions
import watermarking_utils as utils


# --- Helper to save PIL image temporarily ---
def save_temp_pil(image_pil, suffix=".png"):
    """Saves PIL image to a temp file, returns path."""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            fmt = "PNG" if suffix.lower() == ".png" else "JPEG"
            image_pil.save(temp_file.name, format=fmt)
            return temp_file.name
    except Exception as e:
        print(f"Error saving temp file: {e}")
        return None


# --- Gradio Backend Functions ---


def gradio_embed(original_image_pil, payload_str, method_name):
    """Handles the embedding process for Gradio."""
    if original_image_pil is None or not payload_str or not method_name:
        return None, None, "Missing input: Please provide image, payload, and method."

    if method_name not in utils.APPROACHES:
        return None, None, f"Error: Unknown method '{method_name}'."

    watermark_func = utils.APPROACHES[method_name]["watermark_fn"]
    # Get default params for the method (can be extended to take params from UI)
    params = utils.APPROACHES[method_name].get("params", {})

    # Call the embedding function
    stego_image_pil, status = watermark_func(original_image_pil, payload_str, **params)

    if stego_image_pil:
        # Save result to temp file for download link
        temp_file_path = save_temp_pil(
            stego_image_pil, suffix=".png"
        )  # Save lossless PNG
        if temp_file_path:
            return stego_image_pil, temp_file_path, f"Embedding Status:\n{status}"
        else:
            return (
                stego_image_pil,
                None,
                f"Embedding Status:\n{status}\n\nError: Failed to create download file.",
            )
    else:
        return None, None, f"Embedding Failed:\n{status}"


def gradio_benchmark(
    original_image_pil, watermarked_image_pil, original_payload_str, method_name
):
    """Handles the extraction and benchmarking process for Gradio."""
    if (
        original_image_pil is None
        or watermarked_image_pil is None
        or not original_payload_str
        or not method_name
    ):
        return (
            None,
            None,
            "Missing input: Please provide both images, original payload, and method.",
        )

    if method_name not in utils.APPROACHES:
        return None, None, f"Error: Unknown method '{method_name}'."

    extract_func = utils.APPROACHES[method_name]["extract_fn"]
    # Get default params for the method (can be extended to take params from UI)
    params = utils.APPROACHES[method_name].get("params", {})

    # --- Extraction ---
    extracted_payload, extract_status = extract_func(watermarked_image_pil, **params)

    # --- Benchmarking ---
    # Load images as NumPy arrays for metric calculation
    try:
        orig_np = utils.load_image_np(original_image_pil)
        water_np = utils.load_image_np(watermarked_image_pil)

        # Ensure dimensions match for metrics (optional but safer)
        if orig_np.shape != water_np.shape:
            print(
                f"Warning: Image shapes differ ({orig_np.shape} vs {water_np.shape}). Resizing watermarked for metrics."
            )
            h, w = orig_np.shape[:2]
            # Use PIL for consistent resizing
            water_pil_resized = Image.fromarray(water_np).resize(
                (w, h), Image.Resampling.LANCZOS
            )
            water_np = np.array(water_pil_resized)

    except Exception as e:
        return (
            extracted_payload,
            None,
            f"Extraction Status:\n{extract_status}\n\nError loading images for benchmarking: {e}",
        )

    # Calculate metrics
    metrics = utils.benchmark(
        orig_np, water_np, original_payload_str, extracted_payload
    )

    # Format metrics for display
    metrics_str = json.dumps(metrics, indent=2)  # Pretty print JSON
    # Or format manually:
    # metrics_display = (
    #     f"PSNR: {metrics.get('psnr', 'N/A'):.2f} dB\n"
    #     f"SSIM: {metrics.get('ssim', 'N/A'):.4f}\n"
    #     f"BER: {metrics.get('ber', 'N/A'):.4%}\n"
    #     f"Accuracy: {metrics.get('accuracy', 'N/A'):.4%}"
    # )
    status = f"Extraction Status:\n{extract_status}\n\nBenchmarking Status:\n{metrics.get('error', 'Completed')}"

    return extracted_payload, metrics_str, status


# --- Build Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Image Watermarking Embedding & Benchmarking")

    method_choices = list(utils.APPROACHES.keys())

    with gr.Tabs():
        # --- Embed Tab ---
        with gr.TabItem("Embed Watermark"):
            gr.Markdown("Upload an image, enter a payload, select a method, and embed.")
            with gr.Row():
                with gr.Column(scale=1):
                    embed_input_image = gr.Image(type="pil", label="1. Original Image")
                    embed_payload_text = gr.Textbox(
                        label="2. Payload (Text to Embed)", lines=3
                    )
                    embed_method_dd = gr.Dropdown(
                        choices=method_choices,
                        label="3. Embedding Method",
                        value=method_choices[0] if method_choices else None,
                    )
                    embed_button = gr.Button("Embed Watermark", variant="primary")
                with gr.Column(scale=1):
                    embed_output_image = gr.Image(
                        type="pil", label="Watermarked Image (Result)"
                    )
                    embed_download_file = gr.File(
                        label="Download Watermarked Image (PNG)"
                    )
                    embed_status_text = gr.Textbox(
                        label="Status", lines=5, interactive=False
                    )

        # --- Benchmark Tab ---
        with gr.TabItem("Benchmark"):
            gr.Markdown(
                "Upload the original and watermarked images, provide the original payload, select the method used for embedding, and calculate metrics."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    bench_orig_image = gr.Image(type="pil", label="1. Original Image")
                    bench_water_image = gr.Image(
                        type="pil", label="2. Watermarked Image"
                    )
                    bench_orig_payload = gr.Textbox(
                        label="3. Original Payload (Exact text embedded)", lines=3
                    )
                    bench_method_dd = gr.Dropdown(
                        choices=method_choices,
                        label="4. Method Used for Embedding",
                        value=method_choices[0] if method_choices else None,
                    )
                    bench_button = gr.Button(
                        "Extract & Calculate Metrics", variant="primary"
                    )
                with gr.Column(scale=1):
                    bench_extracted_payload = gr.Textbox(
                        label="Extracted Payload", lines=3, interactive=False
                    )
                    bench_metrics_display = gr.JSON(
                        label="Calculated Metrics"
                    )  # Use JSON component
                    # bench_metrics_display = gr.Label(label="Calculated Metrics") # Alternative: Label
                    bench_status_text = gr.Textbox(
                        label="Status", lines=5, interactive=False
                    )

    # --- Connect Components ---
    embed_button.click(
        fn=gradio_embed,
        inputs=[embed_input_image, embed_payload_text, embed_method_dd],
        outputs=[embed_output_image, embed_download_file, embed_status_text],
    )

    bench_button.click(
        fn=gradio_benchmark,
        inputs=[
            bench_orig_image,
            bench_water_image,
            bench_orig_payload,
            bench_method_dd,
        ],
        outputs=[bench_extracted_payload, bench_metrics_display, bench_status_text],
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    # Clean up old temp files (optional)
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            try:
                # Add more specific prefix checks if needed
                # os.remove(os.path.join(temp_dir, filename))
                pass  # Be cautious with auto-deletion
            except OSError:
                pass
    # Launch
    demo.launch()
