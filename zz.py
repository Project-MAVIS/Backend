from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.serialization import load_pem_public_key
import base64
import binascii


def verify_signature_alt(data, public_key_pem, signature):
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
        print(f"public key PEM: {public_key_pem}")
        print(f"data: {data}")
        print(f"signature: {signature}")
        public_key = load_pem_public_key(public_key_pem.encode())
        print(f"public key: {public_key}")
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
        return False


# using ECC key, please dont change public key
# Code used to generate the signature:

    # func sign(data: Data) throws -> Data {
    #     let query: [String: Any] = [
    #         kSecClass as String: kSecClassKey,
    #         kSecAttrApplicationTag as String: tag.data(using: .utf8)!,
    #         kSecAttrKeyType as String: kSecAttrKeyTypeECSECPrimeRandom,
    #         kSecReturnRef as String: true
    #     ]
        
    #     var item: CFTypeRef?
    #     let status = SecItemCopyMatching(query as CFDictionary, &item)
        
    #     guard status == errSecSuccess, let item = item else {
    #         throw CryptographerError.keyNotFound
    #     }
        
    #     let privateKey = item as! SecKey
    #     var error: Unmanaged<CFError>?
    #     guard let signature = SecKeyCreateSignature(
    #         privateKey,
    #         .ecdsaSignatureMessageX962SHA256,
    #         data as CFData,
    #         &error
    #     ) as Data? else {
    #         throw CryptographerError.signingFailed(
    #             error?.takeRetainedValue().localizedDescription ?? "Unknown error"
    #         )
    #     }
        
    #     return signature
    # }

    # func exportPublicKey() throws -> String {
    #     let query: [String: Any] = [
    #         kSecClass as String: kSecClassKey,
    #         kSecAttrApplicationTag as String: tag.data(using: .utf8)!,
    #         kSecAttrKeyType as String: kSecAttrKeyTypeECSECPrimeRandom,
    #         kSecReturnRef as String: true
    #     ]
        
    #     var item: CFTypeRef?
    #     let status = SecItemCopyMatching(query as CFDictionary, &item)
        
    #     guard status == errSecSuccess, let item = item else {
    #         throw CryptographerError.keyNotFound
    #     }
        
    #     let privateKey = item as! SecKey
        
    #     guard let publicKey = SecKeyCopyPublicKey(privateKey) else {
    #         throw CryptographerError.publicKeyExtractionFailed
    #     }
        
    #     var error: Unmanaged<CFError>?
    #     guard let keyData = SecKeyCopyExternalRepresentation(publicKey, &error) as Data? else {
    #         throw CryptographerError.exportFailed(error?.takeRetainedValue().localizedDescription ?? "Unknown error")
    #     }
        
    #     let pemHeader = "-----BEGIN PUBLIC KEY-----\n"
    #     let pemFooter = "\n-----END PUBLIC KEY-----"
    #     let base64Key = keyData.base64EncodedString(options: .lineLength64Characters)
    #     let pemKey = pemHeader + base64Key + pemFooter
        
    #     return pemKey
    # }


from cryptography.hazmat.primitives.serialization import load_pem_public_key
# proper_key_pem = "BM7Ix/Lgdc04OXJHliwfxHwSoMmhB6dmgpO3UFyeURGPA+NNrn3yJNeDCUdn/+b3HF/zQZG8CcomcXKdSyDcp1Y="
proper_key_pem = "BM7Ix/Lgdc04OXJHliwfxHwSoMmhB6dmgpO3UFyeURGPA+NNrn3yJNeDCUdn/+b3HF/zQZG8CcomcXKdSyDcp1Y="
raw_key_bytes = base64.b64decode(proper_key_pem)
public_key = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(),
        raw_key_bytes
    )

print(public_key)
# public_key = load_pem_public_key(proper_key_pem.encode())

data = "9fa0a51a9c44d55b12f6e80cc20639742de9235777e2c9466845159bfd7c8210"
signature_hex = "MEQCIGFDVIDLclNbjiXDIFdXSsTsNRXu/70yN+k6h1Yd6IEyAiBCPxpMIb4QI9QRTBE0p3xNL0U4vdjyqZYNmq/8QVsBug=="

signature_bytes = base64.b64decode(signature_hex)
data = base64.b64decode(data)
# signature_bytes = bytes.fromhex(signature_hex)
# data = bytes.fromhex(data)
# data =  bytes.fromhex(signature_hex)

# For ECDSA, signature is often in raw r||s format (32 bytes for r + 32 bytes for s with P-256)
r = int.from_bytes(signature_bytes[:32], byteorder='big')
s = int.from_bytes(signature_bytes[32:], byteorder='big')

# Convert to DER format which is what cryptography library expects
from cryptography.hazmat.primitives.asymmetric import utils

der_signature = utils.encode_dss_signature(r, s)

public_key.verify(
    der_signature,
    data,
    ec.ECDSA(hashes.SHA256())
)

# proper_key_pem = """-----BEGIN PUBLIC KEY-----
# BM7Ix/Lgdc04OXJHliwfxHwSoMmhB6dmgpO3UFyeURGPA+NNrn3yJNeDCUdn/+b3
# HF/zQZG8CcomcXKdSyDcp1Y=
# -----END PUBLIC KEY-----"""