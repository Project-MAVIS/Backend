import logging
import struct
import time
from dataclasses import dataclass
from typing import Tuple
from .models import Image, DeviceKeys
from django.contrib.auth.models import User

logger = logging.getLogger("server_log")


@dataclass
class ImageCertificate:
    """Data class representing the image certificate structure."""

    cert_len: int
    timestamp: int
    image_id: int
    user_id: int
    device_id: int
    username: str
    device_name: struct

    def __str__(self):
        return f"""
cert_len: {self.cert_len}
timestamp: {self.timestamp}
image_id: {self.image_id}
user_id: {self.user_id}
device_id: {self.device_id}
username: {self.username}
device_name: {self.device_name}
"""


def calculate_cert_length(username: str, device_name: str) -> int:
    """
    Calculate the total certificate length based on the fixed fields and variable length strings.

    Fixed fields:
    - cert_len: 1 byte
    - timestamp: 8 bytes
    - image_id: 8 bytes
    - user_id: 4 bytes
    - device_id: 4 bytes
    - username_length: 1 byte
    - device_name_length: 1 byte

    Variable fields:
    - username: len(username) bytes
    - device_name: len(device_name) bytes
    """
    username_bytes = len(username.encode("utf-8"))
    device_name_bytes = len(device_name.encode("utf-8"))

    # Fixed fields total: 1 + 8 + 8 + 4 + 4 + 1 + 1 = 27 bytes + 40 in worst case of variable length fields
    fixed_fields_length = struct.calcsize(">BQQII") + 2  # +2 for the length fields

    # Total length is fixed fields plus variable length strings
    total_length = fixed_fields_length + username_bytes + device_name_bytes

    return total_length


def create_certificate(
    image: Image, device_key: DeviceKeys, timestamp: int = int(time.time())
) -> str:
    user: User = image.device_key.user
    time_stamp = timestamp
    image_id = image.id
    user_id = user.id
    device_key_id = device_key.id
    user_name = user.username
    device_name = device_key.name

    cert = ImageCertificate(
        cert_len=calculate_cert_length(username=user_name, device_name=device_name),
        timestamp=time_stamp,
        image_id=image_id,
        user_id=user_id,
        device_id=device_key_id,
        username=user_name,
        device_name=device_name,
    )

    serialized_data = serialize_certificate(cert)
    # print(serialized_data)
    print(f"certificate: {serialized_data.hex()}")

    return serialized_data.hex()


def serialize_certificate(cert: ImageCertificate) -> bytes:
    """
    Serialize an ImageCertificate object into bytes according to the specified format.

    Format:
    - cert_len: 8 bits (unsigned char)
    - timestamp: 64 bits (unsigned long long)
    - image_id: 64 bits (unsigned long long)
    - user_id: 32 bits (unsigned int)
    - device_id: 32 bits (unsigned int)
    - username_length: 8 bits (unsigned char)
    - username: variable length string
    - device_name_length: 8 bits (unsigned char)
    - device_name: variable length string
    """
    # Convert strings to bytes
    username_bytes = cert.username.encode("utf-8")
    device_name_bytes = cert.device_name.encode("utf-8")

    # Get lengths of variable fields
    username_length = len(username_bytes)
    device_name_length = len(device_name_bytes)

    # Pack fixed-length fields
    header = struct.pack(
        ">BQQII",  # > for big-endian, B=uint8, Q=uint64, I=uint32
        int(cert.cert_len & 0xFF),  # 8 bits
        int(cert.timestamp & 0xFFFFFFFFFFFFFFFF),  # 64 bits
        int(cert.image_id & 0xFFFFFFFFFFFFFFFF),  # 64 bits
        int(cert.user_id & 0xFFFFFFFF),  # 32 bits
        int(cert.device_id & 0xFFFFFFFF),  # 32 bits
    )

    # Pack variable-length fields
    variable_fields = struct.pack(
        f">B{username_length}sB{device_name_length}s",
        username_length,
        username_bytes,
        device_name_length,
        device_name_bytes,
    )

    return header + variable_fields


def deserialize_certificate(data: bytes) -> Tuple[ImageCertificate, int]:
    """
    Deserialize bytes into an ImageCertificate object.
    Returns a tuple of (certificate, bytes_consumed).
    """
    # Unpack fixed-length portion
    fixed_format = ">BQQII"
    fixed_size = struct.calcsize(fixed_format)

    cert_len, timestamp, image_id, user_id, device_id = struct.unpack(
        fixed_format, data[:fixed_size]
    )

    logger.V(3).info(
        "Deserializer info: ",
        cert_len,
        timestamp,
        image_id,
        user_id,
        device_id,
    )
    # Get username length and username
    username_length = data[fixed_size]
    username_start = fixed_size + 1
    username_end = username_start + username_length
    username = data[username_start:username_end].decode("utf-8")

    # Get device name length and device name
    device_name_length = data[username_end]
    device_name_start = username_end + 1
    device_name_end = device_name_start + device_name_length
    device_name = data[device_name_start:device_name_end].decode("utf-8")

    certificate = ImageCertificate(
        cert_len=cert_len,
        timestamp=timestamp,
        image_id=image_id,
        user_id=user_id,
        device_id=device_id,
        username=username,
        device_name=device_name,
    )

    return certificate, device_name_end


@dataclass
class SignedCertificateData:
    public_key_length: int  # Length of the public key in bytes
    public_key: bytes  # The actual public key
    signed_certificate: bytes  # The previously created certificate


def serialize_signed_certificate(data: SignedCertificateData) -> str:
    """Serialize the signed certificate data into bytes."""
    # Validate public key length matches actual public key
    if len(data.public_key) != data.public_key_length:
        raise ValueError("Public key length doesn't match actual public key size")

    # Pack everything except the total length first
    certificate_content = struct.pack(
        "=I", data.public_key_length  # public_key_length as 32-bit unsigned int
    )
    # Add the variable length fields
    certificate_content += data.public_key + data.signed_certificate

    # Calculate total length
    total_length = len(certificate_content)

    # Pack the length followed by the content
    full_certificate = struct.pack("=B", total_length) + certificate_content

    return full_certificate.hex()


def deserialize_signed_certificate(data: str) -> Tuple[int, SignedCertificateData]:
    """Deserialize bytes into signed certificate data."""
    data = str.encode(data)
    # First byte is the total length
    total_length = struct.unpack("=B", data[0:1])[0]

    # Next 4 bytes are the public key length
    public_key_length = struct.unpack("=I", data[1:5])[0]

    # Extract public key based on its length
    public_key = data[5 : 5 + public_key_length]

    # Rest is the signed certificate
    signed_certificate = data[5 + public_key_length :]

    cert_data = SignedCertificateData(
        public_key_length=public_key_length,
        public_key=public_key,
        signed_certificate=signed_certificate,
    )

    return total_length, cert_data


# Example usage
def example_certificate_usage():
    timestamp = 0x123456789ABCDEF0
    image_id = 0xFEDCBA9876543210
    user_id = 0x12345678
    user_name = "bbbbbbbbbbbbbbbbbbbb"
    device_key_id = 0x87654321
    device_name = "aaaaaaaaaaaaaaaaaaaa"

    # Create a sample certificate
    cert = ImageCertificate(
        cert_len=255,  # This should be calculated based on actual content
        timestamp=timestamp,
        image_id=image_id,
        user_id=user_id,
        device_id=device_key_id,
        username=user_name,
        device_name=device_name,
    )

    # Serialize
    serialized_data = serialize_certificate(cert)
    # print(serialized_data)
    print(serialized_data.hex())

    # Deserialize
    deserialized_cert, bytes_consumed = deserialize_certificate(serialized_data)
    print(f"deserialized_cert: {deserialized_cert}")
    return cert, deserialized_cert, serialized_data


# original, deserialized, binary_data = example_certificate_usage()
# print(f"Original: {original}")
# print(f"Deserialized: {deserialized}")
# print(f"Binary length: {len(binary_data)} bytes")
# print(f"Data matches: {original == deserialized}")


def example_signed_certificate_usage():
    # Create sample data
    sample_public_key = b"SamplePublicKey123"  # Just an example
    sample_signed_cert = b"PreviousCertificate"  # Just an example

    sample_data = SignedCertificateData(
        public_key_length=len(sample_public_key),
        public_key=sample_public_key,
        signed_certificate=sample_signed_cert,
    )

    # Serialize
    binary_data = serialize_signed_certificate(sample_data)

    # print as hex
    print("Serialized data (hex):")
    print(binary_data.hex())
    print(f"Total length: {len(binary_data)} bytes")

    # Deserialize
    total_length, recovered_data = deserialize_signed_certificate(binary_data)
    print("\nDeserialized data:")
    print(f"Total Length: {total_length} bytes")
    print(f"Public Key Length: {recovered_data.public_key_length}")
    print(f"Public Key: {recovered_data.public_key}")
    print(f"Signed Certificate: {recovered_data.signed_certificate}")


if __name__ == "__main__":
    # example_signed_certificate_usage()
    example_certificate_usage()
