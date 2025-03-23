from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import utils
import base64
from cryptography.hazmat.primitives.serialization import load_pem_public_key


import base64
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
import binascii

proper_key_base64 = "BM7Ix/Lgdc04OXJHliwfxHwSoMmhB6dmgpO3UFyeURGPA+NNrn3yJNeDCUdn/+b3HF/zQZG8CcomcXKdSyDcp1Y="
public_key_pem = "-----BEGIN PUBLIC KEY-----\n"
for i in range(0, len(proper_key_base64), 64):
    public_key_pem += proper_key_base64[i:i+64] + "\n"
public_key_pem += "-----END PUBLIC KEY-----"
raw_key_bytes = base64.b64decode(proper_key_base64)

public_key = ec.EllipticCurvePublicKey.from_encoded_point(
    ec.SECP256R1(),  # NIST P-256 curve
    raw_key_bytes
)

data = "9fa0a51a9c44d55b12f6e80cc20639742de9235777e2c9466845159bfd7c8210"
data_bytes = bytearray.fromhex(data)

signature = "MEQCIGFDVIDLclNbjiXDIFdXSsTsNRXu/70yN+k6h1Yd6IEyAiBCPxpMIb4QI9QRTBE0p3xNL0U4vdjyqZYNmq/8QVsBug=="
signature_bytes = base64.b64decode(signature)


# public_key.verify(
#     signature_bytes,
#     data_bytes,
#     ec.ECDSA(hashes.SHA256())
# )

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature
from cryptography.exceptions import InvalidSignature
import base64

def verify_signature(data, public_key_pem, signature):
    """
    Verify a signature created by the Swift SecKeyCreateSignature function
    
    Args:
        data (bytes): The original data that was signed
        public_key_pem (str): The PEM-encoded public key
        signature (bytes): The signature in X9.62 format created by the Swift code
        
    Returns:
        bool: True if signature is valid, False otherwise
    """
    try:
        # Load the public key from PEM format
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        public_key = load_pem_public_key(public_key_pem.encode())
        
        if not isinstance(public_key, ec.EllipticCurvePublicKey):
            raise ValueError("The provided key is not an EC public key")
        
        # The Swift code uses X9.62 (ASN.1 DER) format for the signature
        # We need to extract r and s from this format
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
        return 

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
            print("THIS IS B64STR: ", b64str)
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


print(verify_apple_signature(b'OmkarMadarchod', public_key_pem, signature))