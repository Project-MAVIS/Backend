from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.types import (
    PublicKeyTypes,
    PrivateKeyTypes,
)
import base64
from PIL import Image
from base64 import b64decode
import os
import json
import piexif
from dotenv import load_dotenv
from backend.logging_utils import logger

load_dotenv()


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

    logger.V(2).info(f"Private key: {private_pem.decode()}")
    logger.V(2).info(f"Public key: {public_pem.decode()}")

    return private_pem.decode(), public_pem.decode()


def verify_signature(public_key_pem: PublicKeyTypes, signature, data):
    logger.V(2).info(f"Public key PEM: {public_key_pem}")
    try:
        public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), base64.b64decode(public_key_pem))

        logger.V(4).info(f"Public key: {public_key}")

        public_key.verify(
            signature,
            data,
            # padding.PSS(
            #     mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            # ),
            ec.ECDSA(hashes.SHA256()),
        )
        return True 
    except Exception as e:
        logger.V(2).error(f"Signature verification failed: {e}")
        return False


def encrypt_string(plain_text: str, public_key: rsa.RSAPublicKey) -> str:
    print(type(public_key), public_key.key_size)
    print(plain_text)
    print(len(plain_text.encode("utf-8")))  # Should be ≤190 bytes for RSA-2048

    """Encrypts a given string using the RSA public key."""
    plain_text_bytes = plain_text.encode("utf-8")

    # Encrypt the bytes using the public key
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
    logger.V(2).info(f"Encrypted base64: {encrypted_base64.decode('utf-8')}")

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
    logger.V(2).info(f"Decrypted bytes: {decrypted_bytes.decode('utf-8')}")
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


def initialize_server_keys() -> tuple[PrivateKeyTypes, PublicKeyTypes]:
    """
    Initialize server keys from environment variables or generate new ones.
    Returns tuple of (private_key, public_key) in PEM format.
    """
    # Try to get private key from environment
    private_pem = os.environ.get("SERVER_PRIVATE_KEY")
    public_pem = os.environ.get("SERVER_PUBLIC_KEY")

    if private_pem:
        # Decode base64 encoded private key
        private_pem = b64decode(private_pem.encode())
        public_pem = b64decode(public_pem.encode())

        # Load private key to generate public key
        private_key = serialization.load_pem_private_key(
            private_pem, password=None, backend=default_backend()
        )
        public_key = serialization.load_pem_public_key(
            public_pem, backend=default_backend()
        )

        return private_key, public_key

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
    image.save(img_byte_arr, format=image.format or "PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Calculate and return SHA-256 hash
    hash = hashlib.sha256(img_byte_arr).hexdigest()
    logger.V(2).info(f"Image hash: {hash}")
    return hash


def calculate_string_hash(input_string: str) -> str:
    """
    Calculate SHA-256 hash of a string.

    Args:
        input_string: String to hash

    Returns:
        str: SHA-256 hash string
    """
    import hashlib

    # Encode string to bytes and calculate hash
    hash = hashlib.sha256(input_string.encode()).hexdigest()
    logger.V(2).info(f"String hash: {hash}")
    return hash

def verify_apple_signature(data, base64_public_key, base64_signature):
    """
    Verify a signature created by Apple's SecKeyCreateSignature function
    
    Args:
        data (bytes): The original data that was signed
        base64_public_key (str): Base64-encoded public key from SecKeyCopyExternalRepresentation
        base64_signature (str): Base64-encoded signature from SecKeyCreateSignature
        
    Returns:
        bool: True if signature is valid, False otherwise
    """
    import base64
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    from cryptography.exceptions import InvalidSignature
    
    try:
        # Clean up and fix base64 strings
        def fix_base64(b64str):
            # Remove whitespace, newlines, and PEM headers if present
            b64str = ''.join([line for line in b64str.splitlines() 
                              if not line.startswith('-----') and not line.endswith('-----')])
            b64str = b64str.replace(' ', '').replace('\n', '')
            
            # Add padding if needed
            padding = len(b64str) % 4
            if padding > 0:
                b64str += '=' * (4 - padding)
                
            return b64str
        
        # Fix the base64 strings
        base64_public_key = fix_base64(base64_public_key)
        base64_signature = fix_base64(base64_signature)
        
        # Convert the key and decode the signature
        pem_key = convert_apple_ec_key_to_pem(base64_public_key)
        signature = base64.b64decode(base64_signature)
        
        # Load the public key
        public_key = load_pem_public_key(pem_key.encode())
        
        # The Swift code uses X9.62 (ASN.1 DER) format for the signature
        from asn1crypto.core import Sequence, Integer
        
        # Parse the ASN.1 DER signature
        class ECDSASignature(Sequence):
            _fields = [
                ('r', Integer),
                ('s', Integer)
            ]
        
        parsed_sig = ECDSASignature.load(signature)
        r = int(parsed_sig['r'].native)
        s = int(parsed_sig['s'].native)
        
        # Recreate the signature in the format needed for verification
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
        reconstructed_sig = encode_dss_signature(r, s)
        
        # Verify the signature
        public_key.verify(
            reconstructed_sig,
            data,
            ec.ECDSA(hashes.SHA256())
        )
        return True
    
    except InvalidSignature:
        return False
    except Exception as e:
        print(f"Verification error: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_apple_ec_key_to_pem(base64_key):
    """
    Convert an Apple EC public key (from SecKeyCopyExternalRepresentation) to PEM format
    
    Args:
        base64_key (str): Base64-encoded key string from Apple's APIs
        
    Returns:
        str: The key in PEM format that can be used with Python's cryptography
    """
    import base64
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    
    # Decode the base64 data
    raw_key_data = base64.b64decode(base64_key)
    
    # Apple's SecKeyCopyExternalRepresentation returns EC keys in X9.62 format
    # The format for uncompressed keys is: 0x04 + x-coordinate + y-coordinate
    
    # Check if this looks like an uncompressed EC key
    if len(raw_key_data) >= 65 and raw_key_data[0] == 0x04:
        # Assuming a P-256 key (most common)
        if len(raw_key_data) == 65:  # 1 byte header + 32 bytes x + 32 bytes y
            curve = ec.SECP256R1()
            key_size = 32
        elif len(raw_key_data) == 97:  # 1 byte header + 48 bytes x + 48 bytes y
            curve = ec.SECP384R1()
            key_size = 48
        elif len(raw_key_data) == 133:  # 1 byte header + 66 bytes x + 66 bytes y
            curve = ec.SECP521R1()
            key_size = 66
        else:
            raise ValueError(f"Unsupported key size: {len(raw_key_data)}")
        
        # Extract x and y coordinates (skip the first byte which is the format byte)
        x = int.from_bytes(raw_key_data[1:1+key_size], byteorder='big')
        y = int.from_bytes(raw_key_data[1+key_size:], byteorder='big')
        
        # Create the EC public key object
        public_key = ec.EllipticCurvePublicNumbers(x, y, curve).public_key()
        
        # Serialize to PEM
        pem_key = public_key.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo
        )
        
        return pem_key.decode('ascii')
    else:
        # If it's not in the expected format, try to see if it's already in DER/SPKI format
        try:
            from cryptography.hazmat.primitives.serialization import load_der_public_key
            public_key = load_der_public_key(raw_key_data)
            pem_key = public_key.public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            )
            return pem_key.decode('ascii')
        except Exception:
            # As a last resort, try to construct a SubjectPublicKeyInfo structure
            from cryptography.hazmat.primitives.serialization import load_der_public_key
            from asn1crypto.keys import ECPointBitString, ECDomainParameters, PublicKeyInfo, PublicKeyAlgorithm
            from asn1crypto.algos import AlgorithmIdentifier
            
            # Default to P-256 parameters
            params = {'named': 'secp256r1'}
            alg_id = {'algorithm': 'ec', 'parameters': params}
            
            # Create the public key info
            key_info = {'algorithm': alg_id, 'public_key': raw_key_data}
            der_key = PublicKeyInfo(key_info).dump()
            
            try:
                public_key = load_der_public_key(der_key)
                pem_key = public_key.public_bytes(
                    encoding=Encoding.PEM,
                    format=PublicFormat.SubjectPublicKeyInfo
                )
                return pem_key.decode('ascii')
            except Exception as e:
                raise ValueError(f"Could not parse key data: {e}")



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
