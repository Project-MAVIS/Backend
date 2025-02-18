import datetime
import hashlib
from PIL import Image as PILImage
import json
import numpy as np
import piexif
from io import BytesIO
from exif import Image as ExifImage
from backend.logging_utils import get_verbose_logger

logger = get_verbose_logger("server_log")


def add_exif_to_array_image(
    array: np.ndarray, exif_dict: dict
) -> tuple[PILImage.Image, BytesIO]:
    # Convert array to PIL Image
    img = PILImage.fromarray(array)

    # First save it as JPEG to establish the format
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # Now create the EXIF bytes
    exif_bytes = piexif.dump(exif_dict)

    # Create a new buffer and save with EXIF
    final_buffer = BytesIO()
    img.save(final_buffer, format="PNG", exif=exif_bytes)
    final_buffer.seek(0)

    logger.V(2).info("Added EXIF to array image")
    # Return the final image
    return (PILImage.open(final_buffer), final_buffer)


def add_complex_metadata(file_path, metadata_dict):
    # Open the image
    img = PILImage.open(file_path)

    # Convert metadata dictionary to JSON string
    metadata_json = json.dumps(metadata_dict)

    # Add metadata to the image using piexif
    exif_dict = {"Exif": {}}
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = metadata_json.encode("utf-8")
    exif_bytes = piexif.dump(exif_dict)

    # Save the image with the new metadata
    output_path = file_path.replace(".png", "_with_metadata.png")
    img.save(output_path, exif=exif_bytes)
    print(f"Image with metadata saved at: {output_path}")


def fadd_complex_metadata(image_obj: PILImage.Image, metadata_dict):
    # Convert metadata dictionary to JSON string
    metadata_json = json.dumps(metadata_dict)

    # Add metadata to the image using piexif
    exif_dict = {"Exif": {}}
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = metadata_json.encode("utf-8")
    exif_bytes = piexif.dump(exif_dict)

    # Add exif data to the image
    image_obj.info["exif"] = exif_bytes

    return image_obj


def extract_metadata(file_path):
    # Open the image
    img = PILImage.open(file_path)

    # Extract Exif data from the image
    exif_data = img._getexif()

    # Extract custom metadata (UserComment)
    if exif_data is not None and piexif.ExifIFD.UserComment in exif_data:
        user_comment = exif_data[piexif.ExifIFD.UserComment]
        try:
            metadata = json.loads(user_comment.decode("utf-8"))
            return metadata
        except json.JSONDecodeError:
            print("Error decoding JSON metadata.")
            return None
    else:
        print("No custom metadata found.")
        return None


def fextract_metadata(img: PILImage.Image):
    # Extract Exif data from the image
    exif_data = img.getexif()

    # Extract custom metadata (UserComment)
    if exif_data is not None and piexif.ExifIFD.UserComment in exif_data:
        user_comment = exif_data[piexif.ExifIFD.UserComment]
        try:
            metadata = json.loads(user_comment.decode("utf-8"))
            return metadata
        except json.JSONDecodeError:
            print("Error decoding JSON metadata.")
            return None
    else:
        print("No custom metadata found.")
        return None


def extract_exif_data_PIEXIF(img: PILImage.Image):
    """
    Extracts EXIF data from an image using the PILImage.Image object and the piexif library.

    Args:
        img (PILImage.Image): The image object to extract EXIF data from.

    Returns:
        dict: A dictionary containing the EXIF data.
    """
    try:
        exif_data = piexif.load(img.info.get("exif", b""))

        # Convert bytes to string for JSON serialization
        for ifd in exif_data:
            if ifd in ("0th", "1st", "Exif", "GPS", "Interop"):
                for key in exif_data[ifd]:
                    if isinstance(exif_data[ifd][key], bytes):
                        exif_data[ifd][key] = exif_data[ifd][key].decode(
                            "utf-8", errors="ignore"
                        )

        return exif_data
    except KeyError:
        logger.warning("No EXIF data found in image")
        return None
    except Exception as e:
        logger.error(f"Error extracting EXIF data: {str(e)}")
        return None


def extract_exif_data_EXIF(img_file: bytes):
    """
    Extracts EXIF data from an image using the exif library.

    Args:
        img_file (bytes): The image file to extract EXIF data from.

    Returns:
    """
    img = ExifImage(img_file)
    if not img.has_exif:
        print(f"[+] Skipping file because it does not have EXIF metadata")
    else:
        dict_i = {}

        attr_list = img.list_all()
        for attr in attr_list:
            value = img.get(attr)
            dict_i[attr] = value

        # Convert bytes to string for JSON serialization
        for key, value in dict_i.items():
            if isinstance(value, bytes):
                dict_i[key] = value.decode("utf-8")
            elif isinstance(value, (datetime.datetime, datetime.date)):
                dict_i[key] = value.isoformat()
            elif not isinstance(value, (str, int, float, bool, type(None))):
                dict_i[key] = str(value)

        return dict_i
