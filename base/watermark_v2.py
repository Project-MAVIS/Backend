import numpy as np
import pywt
from PIL import Image
from scipy.fftpack import dct, idct
from pathlib import Path
from backend.logging_utils import logger


class WaveletDCTWatermark:
    def __init__(self, base_path=None):
        """Initialize the watermarking system with base path"""
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.dataset_path = self.base_path / "media" / "dataset"
        self.result_path = self.base_path / "media" / "result"

        # Watermark strength factor
        self.alpha = 0.2

        # Create necessary directories
        self.dataset_path.mkdir(exist_ok=True)
        self.result_path.mkdir(exist_ok=True)

    def convert_image(
        self, image: Image.Image, to_grayscale=False, resize=True, size=2048
    ):
        """Convert and resize an image from a PIL Image object."""

        try:
            # Ensure image dimensions are multiples of 8
            width, height = image.size
            new_width = (width + 7) // 8 * 8
            logger.info(f"[convert_image] new_width: {new_width}")
            new_height = (height + 7) // 8 * 8
            logger.info(f"[convert_image] new_height: {new_height}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to RGBA first
            img = image.convert("RGBA")

            # Handle alpha channel for PNG
            if img.mode == "RGBA":
                # Create white background
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                # Composite the image onto the background
                img = Image.alpha_composite(background, img)

            # Convert to RGB for processing
            img = img.convert("RGB")
            if resize:
                img = img.resize((size, size), Image.Resampling.LANCZOS)

            if to_grayscale:
                img = img.convert("L")
                img = self.enhance_qr_contrast(img)
                w, h = img.size
                logger.info(f"[convert_image] img size: {w,h}")
                image_array = np.array(img.getdata(), dtype=np.float64)
                if resize:
                    image_array = image_array.reshape(size, size)
            else:
                w, h = img.size
                logger.info(f"[convert_image] img size: {w,h}")
                image_array = np.array(img, dtype=np.float64)

            return image_array
        except Exception as e:
            logger.info(f"Error processing image: {str(e)}")
            raise

    def enhance_qr_contrast(self, img):
        """Enhance contrast for QR code"""
        # Convert to numpy array
        img_array = np.array(img)

        # Calculate threshold using Otsu's method
        threshold = self.otsu_threshold(img_array)

        # Binarize the image
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
            logger.info(f"image_array shape: {image_array.shape}")
            if len(image_array.shape) == 3:  # RGB image
                coeffs_by_channel = []
                for channel in range(3):
                    coeffs = pywt.wavedec2(
                        data=image_array[:, :, channel], wavelet=model, level=level
                    )
                    coeffs_by_channel.append(list(coeffs))
                logger.info(f"coeffs_by_channel: {coeffs_by_channel}")
                return coeffs_by_channel
            else:  # Grayscale image
                coeffs = pywt.wavedec2(data=image_array, wavelet=model, level=level)
                return list(coeffs)
        except Exception as e:
            logger.error(f"Error processing coefficients: {str(e)}")
            raise

    def embed_watermark(self, watermark_array, orig_image):
        """Embed watermark in DCT coefficients with enhanced strength"""
        try:
            watermark_flat = watermark_array.ravel()
            ind = 0
            x, y = 0, 0

            logger.info(f"orig_image shape: {orig_image.shape}")
            for x in range(0, orig_image.shape[0], 8):
                for y in range(0, orig_image.shape[1], 8):
                    if ind < len(watermark_flat):
                        subdct = orig_image[x : x + 8, y : y + 8].copy()
                        logger.info(f"subdct shape: {subdct.shape}")
                        # Embed in multiple coefficients for redundancy
                        subdct[4][4] = watermark_flat[ind] * self.alpha
                        subdct[5][5] = watermark_flat[ind] * self.alpha
                        subdct[6][6] = watermark_flat[ind] * self.alpha
                        orig_image[x : x + 8, y : y + 8] = subdct
                        ind += 1

            return orig_image
        except Exception as e:
            logger.error(f"Error embedding watermark: {str(e)}")
            raise

    def get_watermark(self, dct_watermarked_coeff, watermark_size):
        """Extract watermark from DCT coefficients with averaging"""
        try:
            subwatermarks = []

            for x in range(0, dct_watermarked_coeff.shape[0], 8):
                for y in range(0, dct_watermarked_coeff.shape[1], 8):
                    coeff_slice = dct_watermarked_coeff[x : x + 8, y : y + 8]
                    # Average multiple coefficients for better recovery
                    value = (
                        coeff_slice[4][4] + coeff_slice[5][5] + coeff_slice[6][6]
                    ) / (3 * self.alpha)
                    subwatermarks.append(value)

            watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)

            # Enhance recovered watermark
            watermark = self.enhance_recovered_watermark(watermark)
            return watermark
        except Exception as e:
            logger.error(f"Error extracting watermark: {str(e)}")
            raise

    def enhance_recovered_watermark(self, watermark):
        """Enhance recovered watermark for better QR code visibility"""
        # Normalize to 0-255 range
        watermark = (
            (watermark - watermark.min()) / (watermark.max() - watermark.min()) * 255
        )

        # Apply threshold to make QR code more distinct
        threshold = self.otsu_threshold(watermark)
        watermark = np.where(watermark > threshold, 255, 0)

        return watermark

    @staticmethod
    def apply_dct(image_array):
        """Apply DCT transform to image"""
        try:
            height, width = image_array.shape[0], image_array.shape[1]
            all_subdct = np.empty((height, width), dtype=np.float64)
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    subpixels = image_array[i : i + 8, j : j + 8]
                    subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
                    all_subdct[i : i + 8, j : j + 8] = subdct
            return all_subdct
        except Exception as e:
            logger.info(f"Error applying DCT: {str(e)}")
            raise

    def inverse_dct(self, all_subdct):
        """Apply inverse DCT transform"""
        try:
            height, width = all_subdct.shape[0], all_subdct.shape[1]
            all_subidct = np.empty((height, width), dtype=np.float64)
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    subidct = idct(
                        idct(all_subdct[i : i + 8, j : j + 8].T, norm="ortho").T,
                        norm="ortho",
                    )
                    all_subidct[i : i + 8, j : j + 8] = subidct

            return all_subidct
        except Exception as e:
            logger.info(f"Error applying inverse DCT: {str(e)}")
            raise

    def save_image(self, image_array, name):
        """Save image array as image file"""
        try:
            image_array_copy = image_array.clip(0, 255)
            image_array_copy = image_array_copy.astype("uint8")
            img = Image.fromarray(image_array_copy)

            # Determine output format based on filename
            output_format = "PNG" if name.lower().endswith(".png") else "JPEG"

            if output_format == "PNG":
                img.save(self.result_path / name, format=output_format, optimize=True)
            else:
                img.save(
                    self.result_path / name,
                    format=output_format,
                    quality=95,
                    optimize=True,
                )

        except Exception as e:
            logger.info(f"Error saving image: {str(e)}")
            raise

    def watermark_image(self, original_image: Image.Image, watermark: Image.Image):
        """Watermark image received directly from view, returning the watermarked image array.
        Returns the watermarked image array without saving to disk - useful for API endpoints.

        Args:
            original_image (PIL.Image.Image): PIL Image object of the original image
            watermark (PIL.Image.Image): PIL Image object of the watermark

        Returns:
            numpy.ndarray: The watermarked image array as uint8 type, ready for conversion to PIL Image

        Raises:
            Exception: If there is an error during the watermarking process.
            The specific error message is printed before re-raising.
        """
        try:
            model = "haar"
            level = 0

            image_array = self.convert_image(
                original_image, to_grayscale=False, resize=False
            )
            logger.info(f"image_array shape: {image_array.shape}")
            watermark_array = self.convert_image(
                watermark, to_grayscale=True, resize=False
            )

            coeffs_image = self.process_coefficients(image_array, model, level)

            # Handle each color channel separately
            watermarked_image = np.empty_like(image_array)
            for channel in range(3):
                dct_array = self.apply_dct(coeffs_image[channel][0])
                # Embed watermark in both green and blue channels for redundancy
                if channel in [1, 2]:  # Green and Blue channels
                    dct_array = self.embed_watermark(watermark_array, dct_array)
                coeffs_image[channel][0] = self.inverse_dct(dct_array)
                watermarked_image[:, :, channel] = pywt.waverec2(
                    coeffs_image[channel], model
                )

            image_array_copy = watermarked_image.clip(0, 255)
            image_array_copy = image_array_copy.astype("uint8")

            return image_array_copy
        except Exception as e:
            print(f"Error in watermarking process: {str(e)}")
            raise

    def recover_watermark(self, image: Image.Image, model="haar", level=1):
        """Recover watermark from a PIL Image object.

        Similar to recover_watermark() but takes a PIL Image directly instead of a file path.
        This method extracts the embedded watermark from both the green and blue channels
        of a watermarked image and averages them for better clarity.

        Args:
            image (PIL.Image.Image): PIL Image object containing the watermarked image
            model (str, optional): Wavelet transform model to use. Defaults to "haar".
            level (int, optional): Level of wavelet decomposition. Defaults to 1.

        Returns:
            numpy.ndarray: Recovered watermark as a numpy array with uint8 dtype.

        Raises:
            Exception: If there is an error during the watermark recovery process.
            The specific error message is printed before re-raising.
        """
        try:
            image_array = self.convert_image(image, 2048, to_grayscale=False)
            coeffs_watermarked_image = self.process_coefficients(
                image_array, model, level
            )

            # Average watermarks from both green and blue channels
            dct_green = self.apply_dct(coeffs_watermarked_image[1][0])
            dct_blue = self.apply_dct(coeffs_watermarked_image[2][0])

            watermark_green = self.get_watermark(dct_green, 128)
            watermark_blue = self.get_watermark(dct_blue, 128)

            # Average the watermarks from both channels
            watermark_array = (watermark_green + watermark_blue) / 2
            watermark_array = np.uint8(watermark_array)

            return watermark_array
        except Exception as e:
            print(f"Error recovering watermark: {str(e)}")
            raise
