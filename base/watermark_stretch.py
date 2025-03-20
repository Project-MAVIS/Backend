# Basic whatsapp image resilience works, image manipulation at least on whats app is not working yet
import numpy as np
import pywt
from PIL import Image
from scipy.fftpack import dct, idct
from pathlib import Path
from backend.logging_utils import logger

import cv2

# Install zbar using sudo apt install libzbar0
# Works only on linux
# Comment out this code if working on mac or windows
# from pyzbar.pyzbar import decode


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

    def convert_image(self, image_path, size, to_grayscale=False):
        """Convert and resize an image from a file path.

        Args:
            image_path (str): Path to the image file to process
            size (int): Target size to resize image to (both width and height)
            to_grayscale (bool, optional): Whether to convert image to grayscale. Defaults to False.

        Returns:
            numpy.ndarray: Processed image as a numpy array with float64 dtype.
                If grayscale=True, returns 2D array of shape (size, size).
                If grayscale=False, returns 3D array of shape (size, size, channels).

        Raises:
            Exception: If there is an error processing the image.
        """
        try:
            img = Image.open(image_path).convert("RGBA")  # Convert to RGBA first

            # Handle alpha channel for PNG
            if img.mode == "RGBA":
                # Create white background
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                # Composite the image onto the background
                img = Image.alpha_composite(background, img)

            # Convert to RGB for processing
            img = img.convert("RGB")
            img = img.resize((size, size), Image.Resampling.LANCZOS)

            if to_grayscale:
                img = img.convert("L")
                img = self.enhance_qr_contrast(img)
                image_array = np.array(img.getdata(), dtype=np.float64).reshape(
                    (size, size)
                )
            else:
                image_array = np.array(img, dtype=np.float64)

            # Save processed image
            processed_path = self.dataset_path / Path(image_path).name
            img.save(processed_path)

            return image_array
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def fconvert_image(self, image: Image.Image, size, to_grayscale=False):
        """Convert and resize an image from a PIL Image object.

        Similar to convert_image() but takes a PIL Image directly instead of a file path.
        Used when processing images received from views/forms rather than files.

        Args:
            image (PIL.Image.Image): PIL Image object to process
            size (int): Target size to resize image to (both width and height)
            to_grayscale (bool, optional): Whether to convert image to grayscale. Defaults to False.

        Returns:
            numpy.ndarray: Processed image as a numpy array with float64 dtype.
                If grayscale=True, returns 2D array of shape (size, size).
                If grayscale=False, returns 3D array of shape (size, size, channels).

        Raises:
            Exception: If there is an error processing the image.
        """
        try:
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
            img = img.resize((size, size), Image.Resampling.LANCZOS)

            if to_grayscale:
                img = img.convert("L")
                img = self.enhance_qr_contrast(img)
                image_array = np.array(img.getdata(), dtype=np.float64).reshape(
                    (size, size)
                )
            else:
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
            logger.error(f"Error processing coefficients: {str(e)}")
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

    def watermark_image(self, image_path, watermark_path):
        """Main watermarking process that saves the watermarked image to disk.

        This method embeds a watermark into an image by applying wavelet transform and DCT,
        then embedding the watermark in both green and blue channels for redundancy.
        The result is saved to disk - useful for local testing.

        Args:
            image_path (str): Path to the original image file
            watermark_path (str): Path to the watermark image file

        Returns:
            numpy.ndarray: The watermarked image array

        Raises:
            Exception: If there is an error during the watermarking process.
            The specific error message is logger.infoed before re-raising.
        """
        try:
            model = "haar"
            level = 1

            # Get input image format
            input_format = Path(image_path).suffix.lower()
            output_filename = f"image_with_watermark{input_format}"

            logger.info("Converting images...")
            image_array = self.convert_image(image_path, 2048, to_grayscale=False)
            watermark_array = self.convert_image(watermark_path, 128, to_grayscale=True)

            logger.info("Processing and embedding watermark...")
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

            logger.info("Saving watermarked image...")
            self.save_image(watermarked_image, output_filename)
            logger.info("watermarked image saved")

            return watermarked_image
        except Exception as e:
            logger.info(f"Error in watermarking process: {str(e)}")
            raise

    def fwatermark_image(self, original_image: Image, watermark: Image):
        """Watermark image received directly from view, returning the watermarked image array.

        Similar to watermark_image() but takes PIL Image objects directly instead of file paths
        and returns the watermarked image array without saving to disk - useful for API endpoints.

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
            level = 1

            image_array = self.fconvert_image(original_image, 2048, to_grayscale=False)
            watermark_array = self.fconvert_image(watermark, 128, to_grayscale=True)

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

    def recover_watermark(self, image_path, model="haar", level=1):
        """Recover watermark from a watermarked image file.

        This method extracts the embedded watermark from both the green and blue channels
        of a watermarked image file, averages them for better clarity, and saves the
        recovered watermark as a JPEG file.

        Args:
            image_path (str): Path to the watermarked image file
            model (str, optional): Wavelet transform model to use. Defaults to "haar".
            level (int, optional): Level of wavelet decomposition. Defaults to 1.

        Raises:
            Exception: If there is an error during the watermark recovery process.
            The specific error message is printed before re-raising.
        """
        try:
            image_array = self.convert_image(image_path, 2048, to_grayscale=False)
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

            # Save recovered watermark
            img = Image.fromarray(watermark_array)
            img.save(self.result_path / "recovered_watermark.jpg")
        except Exception as e:
            print(f"Error recovering watermark: {str(e)}")
            raise

    def frecover_watermark(self, image: Image.Image, model="haar", level=1):
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
            image_array = self.fconvert_image(image, 2048, to_grayscale=False)
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

    def pad_watermark_to_aspect_ratio(self, watermark_image, original_image):
        """
        Pad watermark (QR code) to match the aspect ratio of the original image.

        Args:
            watermark_image (PIL.Image.Image): The watermark image (typically QR code)
            original_image (PIL.Image.Image): The original image to match aspect ratio

        Returns:
            PIL.Image.Image: Padded watermark image with same aspect ratio as original
        """
        # Get original image dimensions
        orig_width, orig_height = original_image.size
        target_ratio = orig_width / orig_height

        # Get watermark dimensions
        wm_width, wm_height = watermark_image.size
        wm_ratio = wm_width / wm_height

        # Convert to grayscale if not already
        if watermark_image.mode != "L":
            watermark_image = watermark_image.convert("L")

        # Determine new dimensions for padded watermark
        if target_ratio > wm_ratio:  # Original is wider
            new_wm_width = int(wm_height * target_ratio)
            new_wm_height = wm_height

            # Create new white background image with proper ratio
            padded_watermark = Image.new("L", (new_wm_width, new_wm_height), 255)

            # Paste watermark in center
            paste_position = ((new_wm_width - wm_width) // 2, 0)
            padded_watermark.paste(watermark_image, paste_position)
        else:  # Original is taller
            new_wm_width = wm_width
            new_wm_height = int(wm_width / target_ratio)

            # Create new white background image with proper ratio
            padded_watermark = Image.new("L", (new_wm_width, new_wm_height), 255)

            # Paste watermark in center
            paste_position = (0, (new_wm_height - wm_height) // 2)
            padded_watermark.paste(watermark_image, paste_position)

        return padded_watermark

    # Modified version that stretches QR code to match original image aspect ratio
    def stretch_watermark_to_aspect_ratio(self, watermark_image, original_image):
        """
        Stretch watermark (QR code) to match the aspect ratio of the original image.

        Args:
            watermark_image (PIL.Image.Image): The watermark image (typically QR code)
            original_image (PIL.Image.Image): The original image to match aspect ratio

        Returns:
            PIL.Image.Image: Stretched watermark image with same aspect ratio as original
        """
        # Get original image dimensions and aspect ratio
        orig_width, orig_height = original_image.size
        target_ratio = orig_width / orig_height

        # Get watermark dimensions
        wm_width, wm_height = watermark_image.size

        # Convert to grayscale if not already
        if watermark_image.mode != "L":
            watermark_image = watermark_image.convert("L")

        # Calculate new dimensions for stretched watermark with same aspect ratio as original
        new_wm_width = int(np.round(wm_height * target_ratio))
        new_wm_height = wm_height

        # Resize the watermark to match the target aspect ratio
        stretched_watermark = watermark_image.resize(
            (new_wm_width, new_wm_height), Image.Resampling.LANCZOS
        )

        return stretched_watermark

    def fconvert_image_preserve_ratio(
        self, image: Image.Image, target_size, to_grayscale=False
    ):
        """
        Convert and resize an image from a PIL Image object, preserving aspect ratio.

        Args:
            image (PIL.Image.Image): PIL Image object to process
            target_size (int): Target size for the longest dimension
            to_grayscale (bool, optional): Whether to convert image to grayscale. Defaults to False.

        Returns:
            numpy.ndarray: Processed image as a numpy array with float64 dtype.
        """
        try:
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

            # Get original dimensions
            width, height = img.size
            aspect_ratio = width / height

            # Calculate new dimensions preserving aspect ratio
            if width > height:
                new_width = target_size
                new_height = int(target_size / aspect_ratio)
            else:
                new_height = target_size
                new_width = int(target_size * aspect_ratio)

            # Resize the image preserving aspect ratio
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if to_grayscale:
                img = img.convert("L")
                img = self.enhance_qr_contrast(img)
                image_array = np.array(img.getdata(), dtype=np.float64).reshape(
                    (new_height, new_width)
                )
            else:
                image_array = np.array(img, dtype=np.float64)

            return image_array
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise

    def fwatermark_image_v2(self, original_image: Image, watermark: Image):
        """
        Watermark image received directly from view, returning the watermarked image array.
        This version stretches the QR code to match the aspect ratio of the original image.

        Args:
            original_image (PIL.Image.Image): PIL Image object of the original image
            watermark (PIL.Image.Image): PIL Image object of the watermark (typically QR code)

        Returns:
            numpy.ndarray: The watermarked image array as uint8 type, ready for conversion to PIL Image
        """
        try:
            model = "haar"
            level = 1

            # Get original image dimensions and aspect ratio
            orig_width, orig_height = original_image.size
            orig_ratio = orig_width / orig_height

            # Stretch the watermark (QR code) to match the original image's aspect ratio
            stretched_watermark = self.stretch_watermark_to_aspect_ratio(
                watermark, original_image
            )

            # For the original image: preserve aspect ratio and use a size of 2048 for the longest dimension
            if orig_width >= orig_height:
                new_width = 2048
                new_height = int(2048 / orig_ratio)
            else:
                new_height = 2048
                new_width = int(2048 * orig_ratio)

            # Resize original image preserving aspect ratio
            resized_original = original_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
            image_array = np.array(resized_original, dtype=np.float64)

            # For the watermark: preserve the new aspect ratio and use 128 for the longest dimension
            stretched_width, stretched_height = stretched_watermark.size
            stretched_ratio = stretched_width / stretched_height

            if stretched_width >= stretched_height:
                wm_new_width = 128
                wm_new_height = int(128 / stretched_ratio)
            else:
                wm_new_height = 128
                wm_new_width = int(128 * stretched_ratio)

            # Resize stretched watermark
            resized_watermark = stretched_watermark.resize(
                (wm_new_width, wm_new_height), Image.Resampling.LANCZOS
            )

            # Enhance QR code contrast
            resized_watermark = self.enhance_qr_contrast(resized_watermark)
            watermark_array = np.array(resized_watermark, dtype=np.float64)

            # Process wavelet coefficients
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

            # Clip and convert to uint8
            image_array_copy = watermarked_image.clip(0, 255)
            image_array_copy = image_array_copy.astype("uint8")

            return image_array_copy
        except Exception as e:
            print(f"Error in watermarking process: {str(e)}")
            raise

    # The apply_dct and other existing methods need modification to handle non-square images
    def apply_dct(self, image_array):
        """
        Apply DCT transform to image of any shape
        """
        try:
            height, width = image_array.shape
            all_subdct = np.empty((height, width), dtype=np.float64)

            # Process in 8x8 blocks
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    # Handle edge cases where block might be smaller than 8x8
                    h = min(8, height - i)
                    w = min(8, width - j)

                    if h == 8 and w == 8:
                        # Full 8x8 block
                        subpixels = image_array[i : i + 8, j : j + 8]
                        subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
                        all_subdct[i : i + 8, j : j + 8] = subdct
                    else:
                        # Partial block at edge - pad to 8x8
                        subpixels = np.zeros((8, 8), dtype=np.float64)
                        subpixels[:h, :w] = image_array[i : i + h, j : j + w]
                        subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
                        all_subdct[i : i + h, j : j + w] = subdct[:h, :w]

            return all_subdct
        except Exception as e:
            print(f"Error applying DCT: {str(e)}")
            raise

    def inverse_dct(self, all_subdct):
        """
        Apply inverse DCT transform to an array of any shape
        """
        try:
            height, width = all_subdct.shape
            all_subidct = np.empty((height, width), dtype=np.float64)

            # Process in 8x8 blocks
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    # Handle edge cases where block might be smaller than 8x8
                    h = min(8, height - i)
                    w = min(8, width - j)

                    if h == 8 and w == 8:
                        # Full 8x8 block
                        subidct = idct(
                            idct(all_subdct[i : i + 8, j : j + 8].T, norm="ortho").T,
                            norm="ortho",
                        )
                        all_subidct[i : i + 8, j : j + 8] = subidct
                    else:
                        # Partial block at edge - pad to 8x8
                        temp_subdct = np.zeros((8, 8), dtype=np.float64)
                        temp_subdct[:h, :w] = all_subdct[i : i + h, j : j + w]
                        subidct = idct(
                            idct(temp_subdct.T, norm="ortho").T, norm="ortho"
                        )
                        all_subidct[i : i + h, j : j + w] = subidct[:h, :w]

            return all_subidct
        except Exception as e:
            print(f"Error applying inverse DCT: {str(e)}")
            raise

    def embed_watermark(self, watermark_array, orig_image):
        """
        Embed watermark in DCT coefficients with enhanced strength.
        Modified to handle non-square arrays.
        """
        try:
            # Flatten the watermark to 1D array
            watermark_flat = watermark_array.ravel()
            ind = 0

            height, width = orig_image.shape

            for x in range(0, height, 8):
                for y in range(0, width, 8):
                    if ind < len(watermark_flat):
                        # Handle edge cases where block might be smaller than 8x8
                        h = min(8, height - x)
                        w = min(8, width - y)

                        if (
                            h >= 7 and w >= 7
                        ):  # Only embed if we have enough coefficients
                            subdct = orig_image[x : x + h, y : y + w].copy()

                            # For smaller blocks at edges, adjust the coefficients we modify
                            # to stay within bounds
                            if h >= 5 and w >= 5:
                                subdct[4][4] = watermark_flat[ind] * self.alpha
                            if h >= 6 and w >= 6:
                                subdct[5][5] = watermark_flat[ind] * self.alpha
                            if h >= 7 and w >= 7:
                                subdct[6][6] = watermark_flat[ind] * self.alpha

                            orig_image[x : x + h, y : y + w] = subdct
                            ind += 1

            return orig_image
        except Exception as e:
            print(f"Error embedding watermark: {str(e)}")
            raise

    def get_watermark(self, dct_watermarked_coeff, watermark_size):
        """
        Extract watermark from DCT coefficients with averaging.
        Modified to handle non-square watermark extraction.
        """
        try:
            subwatermarks = []
            height, width = dct_watermarked_coeff.shape

            for x in range(0, height, 8):
                for y in range(0, width, 8):
                    # Handle edge cases where block might be smaller than 8x8
                    h = min(8, height - x)
                    w = min(8, width - y)

                    if h >= 7 and w >= 7:  # Only extract if we have enough coefficients
                        coeff_slice = dct_watermarked_coeff[x : x + h, y : y + w]

                        # Average available coefficients
                        count = 0
                        value_sum = 0

                        if h >= 5 and w >= 5:
                            value_sum += coeff_slice[4][4]
                            count += 1
                        if h >= 6 and w >= 6:
                            value_sum += coeff_slice[5][5]
                            count += 1
                        if h >= 7 and w >= 7:
                            value_sum += coeff_slice[6][6]
                            count += 1

                        if count > 0:
                            value = value_sum / (count * self.alpha)
                            subwatermarks.append(value)

            # Calculate the dimensions for the recovered watermark (based on aspect ratio of original watermark)
            # Since we need to reconstruct the watermark, we'll need to infer the aspect ratio

            # Get the total number of values we have
            total_values = len(subwatermarks)

            # If there are enough values, reshape to a rectangle with approximately the right aspect ratio
            if total_values > 0:
                # We'll use a square for simplicity in this implementation
                # A more complex implementation could try to remember the original watermark aspect ratio
                side_length = int(np.sqrt(total_values))

                # Reshape to square
                watermark = np.array(
                    subwatermarks[: side_length * side_length]
                ).reshape(side_length, side_length)

                # Enhance recovered watermark
                watermark = self.enhance_recovered_watermark(watermark)
                return watermark
            else:
                # Return empty array if not enough values
                return np.array([])
        except Exception as e:
            print(f"Error extracting watermark: {str(e)}")
            raise


# def main():
#     """Example usage"""
#     try:
#         # Initialize watermarking system
#         watermarker = WaveletDCTWatermark()

#         # Get input paths
#         image_path = Path("./media/me.jpeg")
#         watermark_path = Path("./media/abc.png")

#         # Validate paths
#         if not image_path.exists():
#             raise FileNotFoundError(f"Original image not found: {image_path}")
#         if not watermark_path.exists():
#             raise FileNotFoundError(f"Watermark image not found: {watermark_path}")

#         # Process watermarking
#         logger.info("\nProcessing watermark...")
#         watermarker.watermark_image(image_path, watermark_path)

#         logger.info("Extracting watermark...")
#         watermarker.recover_watermark(image_path="/home/omkar/Desktop/Backend/base/media/result/image_with_watermark.jpg")
#         # watermarker.recover_watermark(image_name="./man_water_2.jpeg")

#         logger.info("\nResults saved:")
#         logger.info("- Watermarked image: ./result/image_with_watermark.jpg")
#         logger.info("- Recovered watermark: ./result/recovered_watermark.jpg")

#     except Exception as e:
#         logger.info(f"\nError: {str(e)}")
#         logger.info("Watermarking process failed.")

# if __name__ == "__main__":
#     main()
