import hashlib
from PIL import Image as PILImage
import json
import numpy as np
import piexif


def verify_hash(file_path):
    digest = hashlib.sha256()

    img = PILImage.open(file_path)
    img_arr = np.array(img)

    first_row = img_arr[0, :, :]
    rest = img_arr[1:, :, :]

    rest_bytes = rest.tobytes()
    digest = hashlib.sha256(rest_bytes).hexdigest()

    binary_hash = "".join(f"{int(char, 16):04b}" for char in digest)
    binary_hash = binary_hash[: first_row.size * 4]

    embedded_hash = ""
    for i, (r, g, b, a) in enumerate(first_row):
        embedded_hash += str(b & 1)
        if i == 255:
            break

    for i in range(len(binary_hash)):
        if embedded_hash[i] != binary_hash[i]:
            return False

    return True


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


# metadata = {
#     "Author": "Omkar",
#     "Description": "This is an example image with complex metadata.",
#     "Project": {
#         "Name": "Provenance_Addition",
#         "Version": "1.0",
#         "HashAlgorithm": "SHA-256",
#     },
#     "Timestamp": "2024-12-06T12:00:00Z",
# }

# file_path = "./media/testing/omkar_gate_mod.png"
# file_path_mod = "./media/testing/omkar_test.jpeg"

# # add_complex_metadata(file_path, metadata)
# print(extract_metadata(file_path_mod))
