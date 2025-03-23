import numpy as np
from scipy.fftpack import dct, idct
import cv2
import reedsolo as rs


def resize_image_max_dimension(img, max_dimension=1024):
    height, width = img.shape[:2]
    scale_factor = max_dimension / max(height, width)
    if scale_factor < 1:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_img
    return img

def binary_string_to_byte_array(binary_string):
    if len(binary_string) % 8 != 0:
        raise ValueError("invalid")
    integer_value = int(binary_string, 2)
    num_bytes = len(binary_string) // 8
    byte_array = integer_value.to_bytes(num_bytes, byteorder='big')
    return byte_array


def embed_payload(image_path: str, payload: bytes, strength: int):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to read Image")
    if len(payload) > 128:
        raise ValueError("Payload too large. Maximum payload size is 1024 bits")
    
    img = resize_image_max_dimension(img)

    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb_img[:, :, 0].astype(float)

    length = len(payload)
    length_bytes = length.to_bytes(4, byteorder='big')
    length_ecc = rs.RSCodec(1).encode(length_bytes)
    payload_ecc = rs.RSCodec(length // 2).encode(payload)
    combined_payload = length_ecc + payload_ecc
    all_bits = ''.join(format(byte, '08b') for byte in combined_payload)
    height, width = y_channel.shape
    block_size = 8
    height_pad = height + (block_size - height % block_size) if height % block_size != 0 else height
    width_pad = width + (block_size - width % block_size) if width % block_size != 0 else width

    if height_pad != height or width_pad != width:
        padded_img = np.zeros((height_pad, width_pad))
        padded_img[:height, :width] = y_channel
        y_channel = padded_img

    bit_idx = 0
    
    for i in range(0, height_pad, block_size):
        for j in range(0, width_pad, block_size):
            if bit_idx >= len(all_bits):
                break
                
            block = y_channel[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            if bit_idx < len(all_bits):
                if all_bits[bit_idx] == '1':
                    dct_block[4, 3] = np.floor(dct_block[4, 3] / strength) * strength + strength * 0.75
                else:
                    dct_block[4, 3] = np.floor(dct_block[4, 3] / strength) * strength + strength * 0.25
                bit_idx += 1
            
            block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            y_channel[i:i+block_size, j:j+block_size] = block
        
    y_channel = np.clip(y_channel, 0, 255)
    ycrcb_img[:, :, 0] = y_channel[:height, :width].astype(np.uint8)
    embedded_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite('embedded.jpg', embedded_img)


def extract_payload(image_path: str, strength: int) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to read Image")
        
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb_img[:, :, 0].astype(float)
    height, width = y_channel.shape
    block_size = 8
    height_pad = height + (block_size - height % block_size) if height % block_size != 0 else height
    width_pad = width + (block_size - width % block_size) if width % block_size != 0 else width
    
    if height_pad != height or width_pad != width:
        padded_img = np.zeros((height_pad, width_pad))
        padded_img[:height, :width] = y_channel
        y_channel = padded_img
    extracted_bits = []

    for i in range(0, height_pad, block_size):
        for j in range(0, width_pad, block_size):
            block = y_channel[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            coef = dct_block[4, 3]
            relative_val = (coef % strength) / strength
            extracted_bits.append('1' if relative_val > 0.5 else '0')
    try:
        length_ecc = binary_string_to_byte_array(''.join(extracted_bits[:40]))
        length_bytes = rs.RSCodec(1).decode(length_ecc)[0]
        length = int.from_bytes(length_bytes, byteorder='big')
        payload_ecc_length = min(length // 2, 255)
        payload_bits = ''.join(extracted_bits[40:40+(length + payload_ecc_length) * 8])
        payload_ecc_bytes = binary_string_to_byte_array(payload_bits)
        payload = rs.RSCodec(payload_ecc_length).decode(payload_ecc_bytes)[0]
        return payload
    except (ValueError, IndexError):
        return "Error: Could not extract valid payload"