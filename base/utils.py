from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.types import (
    PublicKeyTypes,
    PrivateKeyTypes,
)
import base64
import hashlib
from cv2.typing import MatLike
from PIL import Image
from base64 import b64encode, b64decode
import os
import json
import piexif
from dotenv import load_dotenv
load_dotenv()

from cryptography.hazmat.primitives import serialization


def generate_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return private_pem.decode(), public_pem.decode()


def verify_signature(public_key_pem, signature, data):
    try:
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode(), backend=default_backend()
        )

        public_key.verify(
            signature,
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
        return True
    except:
        return False


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


def initialize_server_keys() -> tuple[str, str]:
    """
    Initialize server keys from environment variables or generate new ones.
    Returns tuple of (private_key, public_key) in PEM format.
    """
    # Try to get private key from environment
    private_pem = os.environ.get("SERVER_PRIVATE_KEY")

    if private_pem:
        # Decode base64 encoded private key
        private_pem = b64decode(private_pem.encode())

        # Load private key to generate public key
        private_key = serialization.load_pem_private_key(private_pem, password=None)

        # Generate public key from private key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return private_pem, public_pem

    # If no private key in environment, raise an error
    raise ValueError(
        "No private key found in environment variables.\n Generate key pair with the following command:\n\nsh scripts/keys.sh"
    )

def calculate_image_hash(image: Image):
    """
    Calculate SHA-256 hash of a PIL Image object.
    
    Args:
        image: PIL Image object
        
    Returns:
        str: SHA-256 hash string
    """
    import hashlib
    import io
    
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format or 'PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Calculate and return SHA-256 hash
    return hashlib.sha256(img_byte_arr).hexdigest()

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
