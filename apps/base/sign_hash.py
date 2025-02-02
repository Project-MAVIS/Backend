import hashlib
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.types import (
    PublicKeyTypes,
    PrivateKeyTypes,
)
import base64
import os
import numpy as np
from cv2.typing import MatLike
from PIL import Image

import json
import piexif
import datetime

from cryptography.hazmat.primitives import serialization


def encrypt_string(plain_text: str, public_key: PublicKeyTypes) -> str:
    """Encrypts a given string using the RSA public key."""
    plain_text_bytes = plain_text.encode("utf-8")

    # Encrypt the bytes
    encrypted_bytes = public_key.encrypt(
        plain_text_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    # Encode the encrypted bytes in base64 to make it string-friendly
    encrypted_base64 = base64.b64encode(encrypted_bytes)
    return encrypted_base64.decode("utf-8")


def decrypt_string(encrypted_text: str, private_key: PrivateKeyTypes) -> str:
    """Decrypts a given encrypted string using the RSA private key."""
    encrypted_bytes = base64.b64decode(encrypted_text)
    decrypted_bytes = private_key.decrypt(
        encrypted_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return decrypted_bytes.decode("utf-8")


# Function to get the SHA-256 hash of an image file
def get_hash(file_path):
    # Initialize SHA-256 hash object
    digest = hashlib.sha256()

    # Read the image file in binary mode and update the hash
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            digest.update(byte_block)

    # Return the hash as a hexadecimal string
    return digest.hexdigest()


def get_hash_from_mat(mat: MatLike):
    """
    :param mat: MatLike object (as returned by cv2.imdecode)
    """
    # Convert the Mat object to bytes
    mat_bytes = mat.tobytes()

    # Initialize SHA-256 hash object
    digest = hashlib.sha256()

    # Update the hash with the bytes from the Mat object
    digest.update(mat_bytes)

    # Return the hash as a hexadecimal string
    return digest.hexdigest()


def get_partial_hash(file_path):
    """
    :param file_path: Path of PNG file (as PNG contains RGBA channels)
    """
    # Open the Image
    img = Image.open(file_path)

    # Convert the image to a Numpy 4D array
    img_arr = np.array(img)

    # Separate the first row of pixels and the rest of the image
    first_row = img_arr[0, :, :]
    rest = img_arr[1:, :, :]

    # Calculate the hash of the remaining image
    rest_bytes = rest.tobytes()
    digest = hashlib.sha256(rest_bytes).hexdigest()

    # Convert the hash to binary and truncate it to the size of the first row
    binary_hash = "".join(f"{int(char, 16):04b}" for char in digest)
    binary_hash = binary_hash[: first_row.size * 4]

    for i, (r, g, b, a) in enumerate(first_row):
        print(i, r, g, b, a)
        b = b & 254
        b = b | int(binary_hash[i])

        first_row[i][2] = b

        if i == 254:
            break

    new_img_arr = np.vstack([first_row[np.newaxis, :, :], rest])
    new_img = Image.fromarray(new_img_arr.astype("uint8"), "RGBA")

    s = file_path.split("/")
    s[-2] = "wm_uploads"
    wm_uploads_path = "/".join(s)
    new_img.save(wm_uploads_path)

    print(binary_hash)


def verify_hash(file_path):
    digest = hashlib.sha256()

    img = Image.open(file_path)
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
    img = Image.open(file_path)

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


def extract_metadata(file_path):
    # Open the image
    img = Image.open(file_path)

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
