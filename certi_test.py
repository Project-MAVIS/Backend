import struct
import time
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ImageCertificate:
    """Data class representing the image certificate structure."""
    cert_len: int
    timestamp: int
    image_id: int
    user_id: int
    device_id: int
    username: str
    device_name: str

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
    username_bytes = cert.username.encode('utf-8')
    device_name_bytes = cert.device_name.encode('utf-8')
    
    # Get lengths of variable fields
    username_length = len(username_bytes)
    device_name_length = len(device_name_bytes)
    
    # Pack fixed-length fields
    header = struct.pack(
        '>BQQII',  # > for big-endian, B=uint8, Q=uint64, I=uint32
        cert.cert_len,
        cert.timestamp,
        cert.image_id,
        cert.user_id,
        cert.device_id
    )
    
    # Pack variable-length fields
    variable_fields = struct.pack(
        f'>B{username_length}sB{device_name_length}s',
        username_length,
        username_bytes,
        device_name_length,
        device_name_bytes
    )
    
    return header + variable_fields

def deserialize_certificate(data: bytes) -> Tuple[ImageCertificate, int]:
    """
    Deserialize bytes into an ImageCertificate object.
    Returns a tuple of (certificate, bytes_consumed).
    """
    # Unpack fixed-length portion
    fixed_format = '>BQQII'
    fixed_size = struct.calcsize(fixed_format)
    
    cert_len, timestamp, image_id, user_id, device_id = struct.unpack(
        fixed_format,
        data[:fixed_size]
    )
    
    # Get username length and username
    username_length = data[fixed_size]
    username_start = fixed_size + 1
    username_end = username_start + username_length
    username = data[username_start:username_end].decode('utf-8')
    
    # Get device name length and device name
    device_name_length = data[username_end]
    device_name_start = username_end + 1
    device_name_end = device_name_start + device_name_length
    device_name = data[device_name_start:device_name_end].decode('utf-8')
    
    certificate = ImageCertificate(
        cert_len=cert_len,
        timestamp=timestamp,
        image_id=image_id,
        user_id=user_id,
        device_id=device_id,
        username=username,
        device_name=device_name
    )
    
    return certificate, device_name_end

# Example usage
def example_usage():
    timestamp=0x123456789ABCDEF0
    image_id=0xFEDCBA9876543210
    user_id=0x12345678
    user_name="bbbbbbbbbbbbbbbbbbbb"
    device_key_id=0x87654321
    device_name="aaaaaaaaaaaaaaaaaaaa"
    
    # Create a sample certificate
    cert = ImageCertificate(
        cert_len=255,  # This should be calculated based on actual content
        timestamp=timestamp,
        image_id=image_id,
        user_id=user_id,
        device_id=device_key_id,
        username=user_name,
        device_name=device_name
    )
    
    # Serialize
    serialized_data = serialize_certificate(cert)
    # print(serialized_data)
    print(serialized_data.hex())

    # Deserialize
    deserialized_cert, bytes_consumed = deserialize_certificate(serialized_data)
    
    return cert, deserialized_cert, serialized_data


original, deserialized, binary_data = example_usage()
print(f"Original: {original}")
print(f"Deserialized: {deserialized}")
print(f"Binary length: {len(binary_data)} bytes")
print(f"Data matches: {original == deserialized}")