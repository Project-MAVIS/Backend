from django.conf import settings
import numpy as np
import pywt
from PIL import Image
from scipy.fftpack import dct, idct
from pathlib import Path

class WaveletDCTWatermark:
    def __init__(self, base_path=None):
        """Initialize the watermarking system with base path"""
        media_root = Path(settings.MEDIA_ROOT)
        base_path = media_root
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.dataset_path = self.base_path / "dataset"
        self.result_path = self.base_path / "result"

        # Improved watermark parameters
        self.alpha = 0.08  # Reduced for better imperceptibility
        self.block_size = 8
        self.dct_positions = [(4,4), (5,5), (6,6), (3,3)]  # Added more positions for redundancy
        
        # Create necessary directories
        self.dataset_path.mkdir(exist_ok=True)
        self.result_path.mkdir(exist_ok=True)

    def convert_image(self, image_path, size, to_grayscale=False):
        """Convert and resize image, with option to convert to grayscale"""
        try:
            img = Image.open(image_path).resize((size, size), Image.Resampling.LANCZOS)
            if to_grayscale:
                img = img.convert("L")
                img = self.enhance_qr_contrast(img)
                image_array = np.array(img.getdata(), dtype=np.float64).reshape((size, size))
            else:
                image_array = np.array(img, dtype=np.float64)

            # Save processed image
            processed_path = self.dataset_path / Path(image_path).name
            img.save(processed_path)

            return image_array
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            raise
    
    def enhance_qr_contrast(self, img):
        """Enhance contrast for QR code"""
        img_array = np.array(img)
        threshold = self.otsu_threshold(img_array)
        binary_img = Image.fromarray(np.uint8(img_array > threshold) * 255)
        return binary_img

    def otsu_threshold(self, image):
        """Calculate Otsu's threshold"""
        histogram = np.histogram(image, bins=256, range=(0, 256))[0]
        total = histogram.sum()
        current_max = 0
        threshold = 0
        sumT = 0
        weightB = 0
        weightF = 0

        for i in range(256):
            sumT += i * histogram[i]

        for i in range(256):
            weightB += histogram[i]
            if weightB == 0:
                continue
            weightF = total - weightB
            if weightF == 0:
                break

            sumB = 0
            for j in range(i + 1):
                sumB += j * histogram[j]

            meanB = sumB / weightB
            meanF = (sumT - sumB) / weightF
            varBetween = weightB * weightF * (meanB - meanF) ** 2

            if varBetween > current_max:
                current_max = varBetween
                threshold = i

        return threshold

    def process_coefficients(self, image_array, model, level):
        """Process wavelet coefficients for each color channel if RGB"""
        try:
            if len(image_array.shape) == 3:  # RGB image
                coeffs_by_channel = []
                for channel in range(3):
                    coeffs = pywt.wavedec2(
                        data=image_array[:, :, channel], wavelet=model, level=level
                    )
                    coeffs_by_channel.append(list(coeffs))
                return coeffs_by_channel
            else:  # Grayscale image
                coeffs = pywt.wavedec2(data=image_array, wavelet=model, level=level)
                return list(coeffs)
        except Exception as e:
            print(f"Error processing coefficients: {str(e)}")
            raise

    def enhance_recovered_watermark(self, watermark):
        """Enhance recovered watermark for better QR code visibility"""
        watermark = (
            (watermark - watermark.min()) / (watermark.max() - watermark.min()) * 255
        )
        threshold = self.otsu_threshold(watermark)
        watermark = np.where(watermark > threshold, 255, 0)
        return watermark
    
    def apply_dct(self, image_array):
        """Apply DCT transform to image"""
        try:
            size = image_array.shape[0]
            all_subdct = np.empty((size, size), dtype=np.float64)
            for i in range(0, size, 8):
                for j in range(0, size, 8):
                    subpixels = image_array[i : i + 8, j : j + 8]
                    subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
                    all_subdct[i : i + 8, j : j + 8] = subdct
            return all_subdct
        except Exception as e:
            print(f"Error applying DCT: {str(e)}")
            raise

    def inverse_dct(self, all_subdct):
        """Apply inverse DCT transform"""
        try:
            size = all_subdct.shape[0]
            all_subidct = np.empty((size, size), dtype=np.float64)
            for i in range(0, size, 8):
                for j in range(0, size, 8):
                    subidct = idct(
                        idct(all_subdct[i : i + 8, j : j + 8].T, norm="ortho").T,
                        norm="ortho",
                    )
                    all_subidct[i : i + 8, j : j + 8] = subidct
            return all_subidct
        except Exception as e:
            print(f"Error applying inverse DCT: {str(e)}")
            raise

    def save_image(self, image_array, name, format="PNG"):
        """Save image array as image file (now with PNG support by default)"""
        try:
            image_array_copy = image_array.clip(0, 255).astype("uint8")
            img = Image.fromarray(image_array_copy)
            img.save(self.result_path / name, format=format)
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            raise

    def embed_watermark(self, watermark_array, orig_image):
        """Enhanced watermark embedding with multiple coefficient positions"""
        try:
            watermark_flat = watermark_array.ravel()
            height, width = orig_image.shape
            embedded = orig_image.copy()
            ind = 0

            for x in range(0, height - self.block_size + 1, self.block_size):
                for y in range(0, width - self.block_size + 1, self.block_size):
                    if ind < len(watermark_flat):
                        block = embedded[x:x + self.block_size, y:y + self.block_size]
                        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                        
                        # Embed watermark in multiple positions with varying strengths
                        for idx, (i, j) in enumerate(self.dct_positions):
                            strength = self.alpha * (1.0 - idx * 0.15)  # Decreasing strength
                            dct_block[i, j] = watermark_flat[ind] * strength
                        
                        # Apply inverse DCT
                        idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                        embedded[x:x + self.block_size, y:y + self.block_size] = idct_block
                        ind += 1

            return embedded
        except Exception as e:
            print(f"Error embedding watermark: {str(e)}")
            raise

    def get_watermark(self, dct_watermarked_coeff, watermark_size):
        """Enhanced watermark extraction with weighted averaging"""
        try:
            subwatermarks = []
            height, width = dct_watermarked_coeff.shape
            weights = [1.0, 0.85, 0.7, 0.5]  # Weights for different positions

            for x in range(0, height - self.block_size + 1, self.block_size):
                for y in range(0, width - self.block_size + 1, self.block_size):
                    block = dct_watermarked_coeff[x:x + self.block_size, y:y + self.block_size]
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    
                    # Extract and combine values from multiple positions
                    weighted_sum = 0
                    total_weight = 0
                    
                    for (i, j), weight in zip(self.dct_positions, weights):
                        weighted_sum += dct_block[i, j] * weight
                        total_weight += weight
                    
                    value = weighted_sum / (total_weight * self.alpha)
                    subwatermarks.append(value)

            watermark = np.array(subwatermarks[:(watermark_size * watermark_size)])
            watermark = watermark.reshape(watermark_size, watermark_size)
            return self.enhance_recovered_watermark(watermark)
        except Exception as e:
            print(f"Error extracting watermark: {str(e)}")
            raise

    def watermark_image(self, image_path, watermark_path):
        """Enhanced main watermarking process with improved channel handling"""
        try:
            model = "haar"
            level = 1

            print("Converting images...")
            image_array = self.convert_image(image_path, 2048, to_grayscale=False)
            watermark_array = self.convert_image(watermark_path, 128, to_grayscale=True)

            print("Processing and embedding watermark...")
            coeffs_image = self.process_coefficients(image_array, model, level)
            watermarked_image = np.empty_like(image_array)

            # Enhanced channel handling
            for channel in range(3):
                channel_coeffs = coeffs_image[channel]
                
                if channel in [1, 2]:  # Green and Blue channels
                    # Apply DCT and embed watermark
                    dct_array = self.apply_dct(channel_coeffs[0])
                    dct_array = self.embed_watermark(watermark_array, dct_array)
                    channel_coeffs[0] = self.inverse_dct(dct_array)
                
                # Reconstruct the channel
                watermarked_image[:, :, channel] = pywt.waverec2(channel_coeffs, model)

            # Apply post-processing to maintain image quality
            watermarked_image = np.clip(watermarked_image, 0, 255)
            
            print("Saving watermarked image...")
            self.save_image(watermarked_image, "watermarked_image.png")

            return watermarked_image

        except Exception as e:
            print(f"Error in watermarking process: {str(e)}")
            raise

    def recover_watermark(self, image_path, model="haar", level=1):
        """Enhanced watermark recovery with multi-channel fusion"""
        try:
            print("Loading watermarked image...")
            image_array = self.convert_image(image_path, 2048, to_grayscale=False)
            coeffs = self.process_coefficients(image_array, model, level)

            # Extract watermarks from both channels with weights
            green_coeffs = self.apply_dct(coeffs[1][0])
            blue_coeffs = self.apply_dct(coeffs[2][0])

            green_watermark = self.get_watermark(green_coeffs, 128)
            blue_watermark = self.get_watermark(blue_coeffs, 128)

            # Weighted combination of watermarks
            combined_watermark = (0.6 * green_watermark + 0.4 * blue_watermark)
            combined_watermark = self.enhance_recovered_watermark(combined_watermark)

            print("Saving recovered watermark...")
            self.save_image(combined_watermark, "recovered_watermark.png")
            
            return combined_watermark

        except Exception as e:
            print(f"Error recovering watermark: {str(e)}")
            raise