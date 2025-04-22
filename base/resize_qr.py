import pyqrcode
import io
import base64
from PIL import Image


def file_generate_qr_code(data, original_image_path, output_path, scale_factor=0.7):
    """
    Generate a QR code with the given data and paste it onto the original image.
    """

    # Generate QR code
    qr = pyqrcode.create(data)
    qr_image = qr.png_as_base64_str(scale=1)

    # Save QR code as an image
    qr_image = Image.open(io.BytesIO(base64.b64decode(qr_image)))
    qr_width, qr_height = qr_image.size

    # Load the original image to get dimensions
    original_image = Image.open(original_image_path)
    original_width, original_height = original_image.size

    # Print dimensions of the original image
    print(f"Original Image Dimensions: {original_width}x{original_height}")

    new_qr_width = int(original_width * scale_factor)
    new_qr_height = int(original_height * scale_factor)
    qr_image = qr_image.resize((new_qr_width, new_qr_height), Image.Resampling.LANCZOS)

    # Create a new image with white background and the same size as the original image
    new_image = Image.new("RGB", (original_width, original_height), "white")

    # Calculate position to paste the QR code (centered)
    paste_position = (
        (original_width - new_qr_width) // 2,
        (original_height - new_qr_height) // 2,
    )

    # Paste the QR code onto the new image
    new_image.paste(qr_image, paste_position)

    # Save the new image
    new_image.save(output_path)

    # Print dimensions of the output image
    print(f"Output Image Dimensions: {new_image.size[0]}x{new_image.size[1]}")


def generate_qr_code(
    data, original_image: Image.Image, scale_factor=0.85
) -> Image.Image:
    # Generate QR code
    qr = pyqrcode.create(data)
    qr_image = qr.png_as_base64_str(scale=1)

    # Save QR code as an image
    qr_image = Image.open(io.BytesIO(base64.b64decode(qr_image)))
    qr_width, qr_height = qr_image.size

    original_width, original_height = original_image.size

    # Print dimensions of the original image
    print(f"Original Image Dimensions: {original_width}x{original_height}")

    # Scale the QR code
    new_qr_width = int(original_width * scale_factor)
    new_qr_height = int(original_height * scale_factor)
    qr_image = qr_image.resize((new_qr_width, new_qr_height), Image.Resampling.LANCZOS)

    # Create a new image with white background and the same size as the original image
    new_image = Image.new("RGB", (original_width, original_height), "white")

    # Calculate position to paste the QR code (centered)
    paste_position = (
        (original_width - new_qr_width) // 2,
        (original_height - new_qr_height) // 2,
    )

    # Paste the QR code onto the new image
    new_image.paste(qr_image, paste_position)

    # Print dimensions of the output image
    print(f"Output Image Dimensions: {new_image.size[0]}x{new_image.size[1]}")

    return new_image


# # Example usage
# data = "https://example.com"
# original_image_path = "media2/images/chili_20250215172945_Wadapav.jpeg"
# output_path = "media2/result/image_with_qr.png"
# generate_qr_code(data, original_image_path, output_path)
