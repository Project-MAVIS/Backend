import numpy as np
import pywt
from PIL import Image
from scipy.fftpack import dct, idct
from pathlib import Path
from backend.logging_utils import logger
import cv2


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

    def convert_image(self, image_path, max_size, to_grayscale=False):
        """Convert and resize an image from a file path while preserving aspect ratio.

        Args:
            image_path (str): Path to the image file to process
            max_size (int): Maximum size for either dimension (width or height)
            to_grayscale (bool, optional): Whether to convert image to grayscale. Defaults to False.

        Returns:
            tuple: (image_array, (width, height)) - Processed image as numpy array and its dimensions
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

            # Calculate new dimensions while preserving aspect ratio
            width, height = img.size
            aspect_ratio = width / height

            if width > height:
                new_width = max_size
                new_height = int(max_size / aspect_ratio)
            else:
                new_height = max_size
                new_width = int(max_size * aspect_ratio)

            # Resize image while preserving aspect ratio
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if to_grayscale:
                img = img.convert("L")
                img = self.enhance_qr_contrast(img)
                image_array = np.array(img.getdata(), dtype=np.float64).reshape(
                    (new_height, new_width)
                )
            else:
                image_array = np.array(img, dtype=np.float64)

            # Save processed image
            processed_path = self.dataset_path / Path(image_path).name
            img.save(processed_path)

            return image_array, (new_width, new_height)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def fconvert_image(self, image: Image.Image, max_size, to_grayscale=False):
        """Convert and resize an image from a PIL Image object while preserving aspect ratio.

        Args:
            image (PIL.Image.Image): PIL Image object to process
            max_size (int): Maximum size for either dimension (width or height)
            to_grayscale (bool, optional): Whether to convert image to grayscale. Defaults to False.

        Returns:
            tuple: (image_array, (width, height)) - Processed image as numpy array and its dimensions
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

            # Calculate new dimensions while preserving aspect ratio
            width, height = img.size
            aspect_ratio = width / height

            if width > height:
                new_width = max_size
                new_height = int(max_size / aspect_ratio)
            else:
                new_height = max_size
                new_width = int(max_size * aspect_ratio)

            # Resize image while preserving aspect ratio
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if to_grayscale:
                img = img.convert("L")
                img = self.enhance_qr_contrast(img)
                image_array = np.array(img.getdata(), dtype=np.float64).reshape(
                    (new_height, new_width)
                )
            else:
                image_array = np.array(img, dtype=np.float64)

            return image_array, (new_width, new_height)
        except Exception as e:
            logger.info(f"Error processing image: {str(e)}")
            raise

    def pad_watermark_to_match_aspect(
        self, watermark_array, target_width, target_height
    ):
        """Pad a watermark (QR code) with white pixels to match target dimensions.

        The QR code is placed in the top-left corner of the resulting image.

        Args:
            watermark_array (numpy.ndarray): Grayscale watermark/QR code array
            target_width (int): Target width to match
            target_height (int): Target height to match

        Returns:
            numpy.ndarray: Padded watermark array matching target dimensions
        """
        # Get current watermark dimensions
        wm_height, wm_width = watermark_array.shape

        # Create a new white (255) array with target dimensions
        padded_watermark = (
            np.ones((target_height, target_width), dtype=watermark_array.dtype) * 255
        )

        # Calculate the maximum size to fit the watermark while preserving aspect ratio
        # We'll make sure it's not larger than 1/4 of the target dimensions
        max_wm_width = min(wm_width, target_width // 4)
        max_wm_height = min(wm_height, target_height // 4)

        # Resize watermark if necessary
        if wm_width > max_wm_width or wm_height > max_wm_height:
            scale = min(max_wm_width / wm_width, max_wm_height / wm_height)
            new_wm_width = int(wm_width * scale)
            new_wm_height = int(wm_height * scale)

            # Resize the watermark
            wm_img = Image.fromarray(watermark_array.astype(np.uint8))
            wm_img = wm_img.resize(
                (new_wm_width, new_wm_height), Image.Resampling.LANCZOS
            )
            watermark_array = np.array(wm_img, dtype=watermark_array.dtype)
            wm_height, wm_width = new_wm_height, new_wm_width

        # Place the watermark in the top-left corner
        padded_watermark[:wm_height, :wm_width] = watermark_array

        return padded_watermark

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

    def embed_watermark(self, watermark_array, orig_image):
        """Embed watermark in DCT coefficients with enhanced strength"""
        try:
            watermark_flat = watermark_array.ravel()
            ind = 0

            for x in range(0, orig_image.shape[0], 8):
                for y in range(0, orig_image.shape[1], 8):
                    if ind < len(watermark_flat):
                        subdct = orig_image[x : x + 8, y : y + 8].copy()
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
        """Extract watermark from DCT coefficients with proper aspect ratio support."""
        try:
            # Extract dimensions from watermark_size tuple
            if isinstance(watermark_size, tuple):
                watermark_height, watermark_width = watermark_size
            else:
                # For backward compatibility
                watermark_height = watermark_width = watermark_size

            # We'll extract values in row-major order (left-to-right, top-to-bottom)
            values = []

            # Only process as many 8x8 blocks as needed for the watermark
            max_blocks_y = (watermark_height + 7) // 8
            max_blocks_x = (watermark_width + 7) // 8

            for x in range(0, min(dct_watermarked_coeff.shape[0], max_blocks_y * 8), 8):
                for y in range(
                    0, min(dct_watermarked_coeff.shape[1], max_blocks_x * 8), 8
                ):
                    coeff_slice = dct_watermarked_coeff[x : x + 8, y : y + 8]
                    # Check if we have a complete 8x8 block
                    if coeff_slice.shape == (8, 8):
                        # Average multiple coefficients for better recovery
                        value = (
                            coeff_slice[4][4] + coeff_slice[5][5] + coeff_slice[6][6]
                        ) / (3 * self.alpha)
                        values.append(value)

            # Calculate how many values we should have for the full watermark
            total_values = watermark_height * watermark_width

            # Ensure we have enough values - pad if necessary
            if len(values) < total_values:
                values.extend([0] * (total_values - len(values)))
            elif len(values) > total_values:
                values = values[:total_values]

            # Reshape to the expected watermark dimensions
            watermark = np.array(values).reshape(watermark_height, watermark_width)

            # Enhance recovered watermark
            watermark = self.enhance_recovered_watermark(watermark)
            return watermark

        except Exception as e:
            logger.error(f"Error extracting watermark: {str(e)}")
            raise

    def enhance_recovered_watermark(self, watermark):
        """Enhanced version of the watermark recovery process."""
        # Normalize to 0-255 range
        if watermark.max() > watermark.min():  # Avoid division by zero
            watermark = (
                (watermark - watermark.min())
                / (watermark.max() - watermark.min())
                * 255
            )
        else:
            watermark = np.zeros_like(watermark)

        # Apply threshold to make QR code more distinct
        threshold = self.otsu_threshold(watermark)
        watermark = np.where(watermark > threshold, 255, 0)

        # Apply morphological operations to enhance QR code
        # Convert to uint8 for OpenCV operations
        watermark_uint8 = watermark.astype(np.uint8)

        # Create a kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)

        # Apply opening to remove noise (erosion followed by dilation)
        watermark_uint8 = cv2.morphologyEx(watermark_uint8, cv2.MORPH_OPEN, kernel)

        # Apply closing to fill small holes (dilation followed by erosion)
        watermark_uint8 = cv2.morphologyEx(watermark_uint8, cv2.MORPH_CLOSE, kernel)

        return watermark_uint8

    @staticmethod
    def apply_dct(image_array):
        """Apply DCT transform to image"""
        try:
            height, width = image_array.shape
            all_subdct = np.empty((height, width), dtype=np.float64)

            # Process 8x8 blocks
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    # Handle edge cases where block might be smaller than 8x8
                    block_height = min(8, height - i)
                    block_width = min(8, width - j)

                    if block_height == 8 and block_width == 8:
                        # Full 8x8 block
                        subpixels = image_array[i : i + 8, j : j + 8]
                        subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
                        all_subdct[i : i + 8, j : j + 8] = subdct
                    else:
                        # Partial block at edges - pad to 8x8
                        subpixels = np.zeros((8, 8), dtype=np.float64)
                        subpixels[:block_height, :block_width] = image_array[
                            i : i + block_height, j : j + block_width
                        ]
                        subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
                        all_subdct[i : i + block_height, j : j + block_width] = subdct[
                            :block_height, :block_width
                        ]

            return all_subdct
        except Exception as e:
            logger.info(f"Error applying DCT: {str(e)}")
            raise

    def inverse_dct(self, all_subdct):
        """Apply inverse DCT transform"""
        try:
            height, width = all_subdct.shape
            all_subidct = np.empty((height, width), dtype=np.float64)

            # Process 8x8 blocks
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    # Handle edge cases where block might be smaller than 8x8
                    block_height = min(8, height - i)
                    block_width = min(8, width - j)

                    if block_height == 8 and block_width == 8:
                        # Full 8x8 block
                        subidct = idct(
                            idct(all_subdct[i : i + 8, j : j + 8].T, norm="ortho").T,
                            norm="ortho",
                        )
                        all_subidct[i : i + 8, j : j + 8] = subidct
                    else:
                        # Partial block at edges - pad to 8x8
                        subblock = np.zeros((8, 8), dtype=np.float64)
                        subblock[:block_height, :block_width] = all_subdct[
                            i : i + block_height, j : j + block_width
                        ]
                        subidct = idct(idct(subblock.T, norm="ortho").T, norm="ortho")
                        all_subidct[i : i + block_height, j : j + block_width] = (
                            subidct[:block_height, :block_width]
                        )

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
        """Main watermarking process that saves the watermarked image to disk while preserving aspect ratio."""
        try:
            model = "haar"
            level = 1

            # Get input image format
            input_format = Path(image_path).suffix.lower()
            output_filename = f"image_with_watermark{input_format}"

            logger.info("Converting images...")
            image_array, (img_width, img_height) = self.convert_image(
                image_path, 2048, to_grayscale=False
            )
            watermark_array, _ = self.convert_image(
                watermark_path, 128, to_grayscale=True
            )

            # Pad watermark to match aspect ratio of original image
            padded_watermark = self.pad_watermark_to_match_aspect(
                watermark_array, img_width, img_height
            )

            logger.info("Processing and embedding watermark...")
            coeffs_image = self.process_coefficients(image_array, model, level)

            # Handle each color channel separately
            watermarked_image = np.empty_like(image_array)
            for channel in range(3):
                dct_array = self.apply_dct(coeffs_image[channel][0])
                # Embed watermark in both green and blue channels for redundancy
                if channel in [1, 2]:  # Green and Blue channels
                    dct_array = self.embed_watermark(padded_watermark, dct_array)
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
        """Watermark image received directly from view, preserving aspect ratio."""
        try:
            model = "haar"
            level = 1

            # Process original image while preserving aspect ratio
            image_array, (img_width, img_height) = self.fconvert_image(
                original_image, 2048, to_grayscale=False
            )
            watermark_array, _ = self.fconvert_image(watermark, 128, to_grayscale=True)

            # Pad watermark to match aspect ratio of original image
            padded_watermark = self.pad_watermark_to_match_aspect(
                watermark_array, img_width, img_height
            )

            coeffs_image = self.process_coefficients(image_array, model, level)

            # Handle each color channel separately
            watermarked_image = np.empty_like(image_array)
            for channel in range(3):
                dct_array = self.apply_dct(coeffs_image[channel][0])
                # Embed watermark in both green and blue channels for redundancy
                if channel in [1, 2]:  # Green and Blue channels
                    dct_array = self.embed_watermark(padded_watermark, dct_array)
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
        """Recover watermark from a watermarked image file with correct aspect ratio."""
        try:
            # Get the original image dimensions
            image_array, (img_width, img_height) = self.convert_image(
                image_path, 2048, to_grayscale=False
            )
            coeffs_watermarked_image = self.process_coefficients(
                image_array, model, level
            )

            # Average watermarks from both green and blue channels
            dct_green = self.apply_dct(coeffs_watermarked_image[1][0])
            dct_blue = self.apply_dct(coeffs_watermarked_image[2][0])

            # Calculate watermark dimensions maintaining aspect ratio
            # The watermark is in the top-left quarter of the image
            wm_height = img_height // 4
            wm_width = img_width // 4

            # Get watermark from both channels
            watermark_green = self.get_watermark(dct_green, (wm_height, wm_width))
            watermark_blue = self.get_watermark(dct_blue, (wm_height, wm_width))

            # Average the watermarks from both channels
            watermark_array = (watermark_green + watermark_blue) / 2

            # Clean up the watermark using binary thresholding
            watermark_array = self.enhance_recovered_watermark(watermark_array)

            # Crop out empty areas (all black/white regions) to focus on the actual QR code
            watermark_array = self.crop_to_content(watermark_array)

            # Save recovered watermark
            img = Image.fromarray(np.uint8(watermark_array))
            img.save(self.result_path / "recovered_watermark.jpg")

            return watermark_array
        except Exception as e:
            print(f"Error recovering watermark: {str(e)}")
            raise

    def frecover_watermark(self, image: Image.Image, model="haar", level=1):
        """Recover watermark from a PIL Image object with correct aspect ratio."""
        try:
            # Get the original image dimensions
            image_array, (img_width, img_height) = self.fconvert_image(
                image, 2048, to_grayscale=False
            )
            coeffs_watermarked_image = self.process_coefficients(
                image_array, model, level
            )

            # Average watermarks from both green and blue channels
            dct_green = self.apply_dct(coeffs_watermarked_image[1][0])
            dct_blue = self.apply_dct(coeffs_watermarked_image[2][0])

            # Calculate watermark dimensions maintaining aspect ratio
            # The watermark is in the top-left quarter of the image
            wm_height = img_height // 4
            wm_width = img_width // 4

            # Get watermark from both channels
            watermark_green = self.get_watermark(dct_green, (wm_height, wm_width))
            watermark_blue = self.get_watermark(dct_blue, (wm_height, wm_width))

            # Average the watermarks from both channels
            watermark_array = (watermark_green + watermark_blue) / 2

            # Clean up the watermark using binary thresholding
            watermark_array = self.enhance_recovered_watermark(watermark_array)

            # Crop out empty areas to focus on the actual QR code
            watermark_array = self.crop_to_content(watermark_array)

            return np.uint8(watermark_array)
        except Exception as e:
            print(f"Error recovering watermark: {str(e)}")
            raise

    def crop_to_content(self, image_array, padding=10):
        """Crop image to content area, removing empty borders."""
        # Ensure image is binary (0 or 255)
        binary = np.where(image_array > 127, 255, 0)

        # Find rows and columns with content
        rows = np.where(np.any(binary < 255, axis=1))[0]
        cols = np.where(np.any(binary < 255, axis=0))[0]

        # Handle empty image case
        if len(rows) == 0 or len(cols) == 0:
            return image_array

        # Get bounds with padding
        top = max(0, rows.min() - padding)
        bottom = min(binary.shape[0], rows.max() + 1 + padding)
        left = max(0, cols.min() - padding)
        right = min(binary.shape[1], cols.max() + 1 + padding)

        # Crop to content area
        return image_array[top:bottom, left:right]
        # Works only on linux
        # Comment out this code if working on mac or windows
        # def read_qr_code_ZBAR(self, image_path):
        #     try:
        #         # Read the image
        #         image = cv2.imread(image_path)

        #         if image is None:
        #             raise ValueError(f"Could not read image at {image_path}")

        #         # Convert to grayscale
        #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #         # Decode QR codes
        #         qr_codes = decode(gray)

        #         if not qr_codes:
        #             logger.info("No QR codes found in the image")
        #             return []

        #         results = []
        #         for qr in qr_codes:
        #             # Convert bytes to string
        #             data = qr.data.decode("utf-8")
        #             results.append({"data": data, "type": qr.type, "position": qr.rect})

        #         return results

        #     except Exception as e:
        #         logger.info(f"Error reading QR code: {str(e)}")
        #         raise e

        # def fread_qr_code_ZBAR(pil_image: Image.Image):
        #     try:
        #         # Convert PIL image to OpenCV format (numpy array)
        #         image = np.array(pil_image)

        #         # Convert to grayscale
        #         gray = cv2.cvtColor(
        #             image, cv2.COLOR_RGB2GRAY
        #         )  # Use RGB since PIL loads in RGB format

        #         # Decode QR codes
        #         qr_codes = decode(gray)

        #         if not qr_codes:
        #             logger.info("No QR codes found in the image")
        #             return []

        #         results = []
        #         for qr in qr_codes:
        #             # Convert bytes to string
        #             data = qr.data.decode("utf-8")
        #             results.append({"data": data, "type": qr.type, "position": qr.rect})

        #         return results

        #     except Exception as e:
        #         logger.info(f"Error reading QR code: {str(e)}")
        #         raise e

        def read_qr_code_cv2(self, image_path):
            """Alternative QR code reader using OpenCV's QR detector.

            This method provides a backup QR code reading implementation using
            pure OpenCV, which may work better on some systems where ZBar fails.

            Args:
                image_path (str): Path to the image containing QR code

            Returns:
                list: List of dictionaries containing detected QR code data
            """
            try:
                # Read the image
                image = cv2.imread(str(image_path))

                if image is None:
                    raise ValueError(f"Could not read image at {image_path}")

                # Initialize QR Code detector
                qr_detector = cv2.QRCodeDetector()

                # Detect and decode QR code
                data, bbox, straight_qrcode = qr_detector.detectAndDecode(image)

                if data:
                    return [
                        {
                            "data": data,
                            "type": "QR",
                            "position": bbox[0] if bbox is not None else None,
                        }
                    ]
                else:
                    logger.info("No QR codes found in the image")
                    return []

            except Exception as e:
                logger.error(f"Error reading QR code with OpenCV: {str(e)}")
                raise

        def fread_qr_code_cv2(self, pil_image: Image.Image):
            """Alternative QR code reader using OpenCV's QR detector for PIL images.

            Similar to read_qr_code_cv2() but takes a PIL Image directly instead of a file path.

            Args:
                pil_image (PIL.Image.Image): PIL Image object containing QR code

            Returns:
                list: List of dictionaries containing detected QR code data
            """
            try:
                # Convert PIL image to OpenCV format
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                # Initialize QR Code detector
                qr_detector = cv2.QRCodeDetector()

                # Detect and decode QR code
                data, bbox, straight_qrcode = qr_detector.detectAndDecode(image)

                if data:
                    return [
                        {
                            "data": data,
                            "type": "QR",
                            "position": bbox[0] if bbox is not None else None,
                        }
                    ]
                else:
                    logger.info("No QR codes found in the image")
                    return []

            except Exception as e:
                logger.error(f"Error reading QR code with OpenCV: {str(e)}")
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
