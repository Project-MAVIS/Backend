# Python standard library
import io
import os
import base64
import hashlib
from pathlib import Path
from datetime import datetime
import random

# Django imports
from django.shortcuts import render
from django.urls import reverse
from django.conf import settings
from django.http import FileResponse, HttpResponse
from django.contrib.auth.models import User
from django.core.files.base import ContentFile

# Django REST Framework
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.permissions import AllowAny
from rest_framework.validators import UniqueValidator
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

# Cryptography
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet

# QR Code related
import pyqrcode

# Image processing
import numpy as np
import piexif
from PIL import Image as PILImage
from PIL.ExifTags import TAGS

# Local imports
from .models import DeviceKeys, Image
from .serializers import UserSerializer, ImageSerializer
from .utils import *
from .watermark import WaveletDCTWatermark
from .certificate import *
from .metadata import *
from . import models
from backend.logging_utils import logger


@api_view(["GET"])
def ping(_) -> Response:
    logger.V(1).info("PING PONG LOG")
    return Response("pong")


@api_view(["GET"])
def truncate(_):
    try:
        Image.objects.all().delete()
    except Exception as e:
        return Response({"Status": "Exception", "message": str(e)})
    return Response({"Status": "Complete"})


class UserRegistrationView(generics.CreateAPIView):
    """Used to mock user creation from mobile device"""

    serializer_class = UserSerializer
    permission_classes = permissions.AllowAny

    def create(self, request: Request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)

        logger.V(3).info(f"serializer: {serializer}")
        logger.V(3).info(f"request data: {request.data}")

        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        return Response(
            {
                "message": "Registration successful",
                "user_id": user.id,
                "username": user.username,
            },
            status=status.HTTP_201_CREATED,
        )


class DeviceRegistrationView(APIView):
    """
    View for registering a new device with associated public key and device name.
    Requires username, device_name, and public_key in the request.
    """

    permission_classes = (AllowAny,)

    def post(self, request: Request):
        try:
            # Get required fields from request data
            username = request.data.get("username")
            device_name = request.data.get("device_name")
            public_key = request.data.get("public_key")

            # Validate required fields
            if not all([username, device_name, public_key]):
                return Response(
                    {"error": "Username, device name and public key are required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Check if user exists
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                return Response(
                    {"error": "User not found"}, status=status.HTTP_404_NOT_FOUND
                )

            # Check if device name already exists for this user
            if DeviceKeys.objects.filter(name=device_name).exists():
                return Response(
                    {"error": "Device name already exists"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Create new device key entry
            device_key = DeviceKeys.objects.create(
                user=user, name=device_name, public_key=public_key
            )

            return Response(
                {
                    "message": "Device registered successfully",
                    "device_id": device_key.id,
                    "device_name": device_key.name,
                    "username": user.username,
                },
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            logger.error(f"Error in device registration: {str(e)}")
            return Response(
                {"error": f"Error registering device: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ImageUploadView(generics.CreateAPIView):
    """Device create image hash, device sign image hash, server verifies signature and saves image"""

    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request: Request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        FORMAT = request.data.get("format")

        try:
            # Get the device-signed image hash
            dev_signed_img_hash = base64.b64decode(request.data.get("image_hash"))
            logger.V(3).info(f"dev_signed_img_hash: {dev_signed_img_hash}")

            # Get the image object
            image_obj = serializer.validated_data["image"]
            image_hash = hashlib.sha256(image_obj.read()).hexdigest()
            logger.V(3).info(f"calculated image hash: {image_hash}")

            device_name = request.data.get("device_name")
            logger.V(3).info(f"device_name: {device_name}")

            if not device_name:
                return Response(
                    {"error": "device name is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                device_key = DeviceKeys.objects.get(name=device_name)
            except DeviceKeys.DoesNotExist:
                return Response(
                    {"error": "DeviceKeys not found"}, status=status.HTTP_404_NOT_FOUND
                )

            if verify_signature(
                device_key.public_key, dev_signed_img_hash, image_hash.encode()
            ):
                # Save the original image first
                image_object = serializer.save(
                    device_key=device_key, verified=False, image_hash=image_hash
                )

                # Make the certificate, sign the certificate
                img_cert = create_certificate(image_object, device_key)
                logger.V(3).info(f"img_cert: {img_cert}")

                enc_img_cert = encrypt_string(img_cert, settings.SERVER_PUBLIC_KEY)
                logger.V(3).info(f"enc_img_cert: {enc_img_cert}")

                # Calculate the hash of the certificate
                cert_hash = calculate_string_hash(img_cert)
                logger.V(3).info(f"cert_hash: {cert_hash}")

                # Create the QR code
                final_cert_qr_code = pyqrcode.create(cert_hash)
                buffer = io.BytesIO()
                final_cert_qr_code.png(buffer, scale=8)
                buffer.seek(0)

                # Create watermarked version with embedded certificate
                wm = WaveletDCTWatermark()
                watermarked_img_array = wm.fwatermark_image(
                    original_image=PILImage.open(image_obj),
                    watermark=PILImage.open(buffer),
                )

                # Add certificate metadata to watermarked image
                metadata = {
                    "0th": {
                        piexif.ImageIFD.Make: enc_img_cert.encode(),
                    }
                }

                watermarked_img_w_metadata, image_buffer = add_exif_to_array_image(
                    array=watermarked_img_array, exif_dict=metadata
                )

                new_hash = calculate_image_hash(watermarked_img_w_metadata)

                # Save watermarked version to db
                watermarked_image_obj = models.Image(
                    device_key=device_key,
                    image=ContentFile(
                        image_buffer.getvalue(),
                        name=f"{device_key.name}_watermarked_image_{random.randint(1,1000000)}.{FORMAT}",
                    ),
                    image_hash=new_hash,
                    original_image_hash=image_hash,
                    verified=True,
                )
                watermarked_image_obj.save()

                # Save the watermarked image as a PNG to local disk
                # TODO: Migrate this to Azure Blob
                watermarked_img_w_metadata.save(
                    Path.cwd() / "media" / "temp" / watermarked_image_obj.image.name,
                    format=FORMAT,
                    optimize=True,
                    exif=piexif.dump(metadata),
                )

                logger.V(3).info(
                    f"Watermarked Image Saved Path: {watermarked_image_obj.image.path}"
                )

                # Generate download link
                download_url = request.build_absolute_uri(
                    reverse(
                        "download-image", kwargs={"image_id": watermarked_image_obj.id}
                    )
                )

                return Response(
                    {
                        "message": "Image verified successfully",
                        "download_url": download_url,
                    },
                    status=status.HTTP_201_CREATED,
                )
            return Response(
                {"error": "Invalid signature"}, status=status.HTTP_400_BAD_REQUEST
            )

        except Exception as e:
            logger.error(f"Error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ImageVerifierView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request: Request):
        try:
            if "image" not in request.FILES:
                return Response(
                    {"error": "Image file is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            image_file = request.FILES["image"]
            try:
                pil_image = PILImage.open(image_file)
            except Exception as e:
                return Response(
                    {"error": f"Invalid image file: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                metadata = extract_exif_data_PIEXIF(pil_image)
                if "0th" not in metadata or 271 not in metadata["0th"]:
                    return Response(
                        {"error": "Image does not contain required metadata"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                signed_certificate = metadata["0th"][271]
            except Exception as e:
                return Response(
                    {"error": f"Error extracting metadata: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            logger.V(3).info(f"signed_certificate: {signed_certificate}")

            try:
                certificate = decrypt_string(
                    signed_certificate, settings.SERVER_PRIVATE_KEY
                )
                logger.V(3).info(f"certificate: {certificate}")

                certificate_hash = calculate_string_hash(certificate)
                logger.V(3).info(f"certificate_hash: {certificate_hash}")

                deserialized_certificate, length = deserialize_certificate(
                    # Read the certificate as bytes from hex-encoded string as the certificate is a hex string
                    bytes.fromhex(certificate)
                )
                logger.V(3).info(
                    f"deserialized_certificate: {deserialized_certificate}"
                )
            except ValueError as e:
                return Response(
                    {"error": f"Invalid certificate format: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            except Exception as e:
                return Response(
                    {"error": f"Error processing certificate: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                watermarker = WaveletDCTWatermark()
                watermark_arr = watermarker.frecover_watermark(pil_image)
                watermark_img = PILImage.fromarray(watermark_arr)
                data = watermarker.fread_qr_code_cv2(watermark_img)
            except Exception as e:
                return Response(
                    {"error": f"Error recovering watermark: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            logger.V(3).info(f"data: {data}")

            if not data:
                return Response(
                    {"error": "No QR code data found in watermark"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            hash_from_qr = data[0]["data"]
            logger.V(2).info(f"hash_from_qr: {hash_from_qr}")

            if certificate_hash != hash_from_qr:
                return Response(
                    {"error": "Certificate hash does not match QR code hash"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            else:
                return Response(
                    {
                        "success": True,
                        "message": "Verification successful. Image is authentic.",
                        "image_id": deserialized_certificate.image_id,
                        "user_id": deserialized_certificate.user_id,
                        "device_id": deserialized_certificate.device_id,
                        "timestamp": datetime.datetime.fromtimestamp(
                            deserialized_certificate.timestamp
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "username": deserialized_certificate.username,
                        "device_name": deserialized_certificate.device_name,
                    },
                    status=status.HTTP_200_OK,
                )

        except Exception as e:
            logger.error(f"Unexpected error during verification: {str(e)}")
            return Response(
                {"error": "Internal server error occurred"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class Demo_RegisterUserView(generics.CreateAPIView):
    """Used to mock user creation from mobile device"""

    serializer_class = UserSerializer
    permission_classes = (permissions.AllowAny,)

    def create(self, request: Request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)

        logger.V(3).info(f"serializer: {serializer}")
        logger.V(3).info(f"request data: {request.data}")

        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # Generate and save key pair
        private_key, public_key = generate_key_pair()

        logger.V(3).info(f"private key: {private_key}")

        DeviceKeys.objects.create(
            user=user,
            private_key=private_key,
            public_key=public_key,
            name=user.username,
        )

        return Response(
            {
                "message": "Registration successful",
                "user_id": user.id,
                "username": user.username,
                "public_key": public_key,
                "private_key": private_key,
            },
            status=status.HTTP_201_CREATED,
        )


# TODO: FIX THIS
# ! DOES NOT WORK
class ImageLinkProvider(APIView):
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request: Request):
        try:
            # # Here the server will decrypt the signed hash then get the relevant image if it exists **and is unverified** if verified these is malicious behaviour
            # client_and_server_signed_certificate = request.data.get("signature")
            # server_signed_hash = request.data.get("signed_hash")
            # image_hash = decrypt_string(server_signed_hash, server_private_key)
            # image_object = Image.objects.get(image_hash=image_hash)
            # image_object.verified = True

            # # The server generates a new certificate with the photographer new certificate length, public key and the signed certificate
            # # =====================================================================
            # # | length | public key length | user public key | signed certificate |
            # # =====================================================================
            # device_name = request.data.get("device_name")
            # device_key = DeviceKeys.objects.get(name=device_name)
            # final_certificate = create_signed_certificate(
            #     device_key.public_key,
            #     signed_certificate=cliend_and_server_signed_certificate
            # )

            # # The server then adds/ embeds this new certificate to the image
            # # This signed image is saved separately, the original and watermarked image is now marked verified
            # # The link is send back to the photographer to download the image or share the link

            # Get signed data from request
            hybrid_signed_cert_and_key = request.data.get("signed_certificate_and_key")
            server_signed_hash = request.data.get("signed_hash")

            # Decrypt the server signed hash to get original image hash
            image_hash = decrypt_string(server_signed_hash, settings.SERVER_PRIVATE_KEY)

            # Get the image and verify it's not already verified
            image_object = Image.objects.get(image_hash=image_hash)
            if image_object.verified:
                return Response(
                    {"error": "Image already verified"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Mark image as verified
            image_object.verified = True
            image_object.save()

            # Get device key to create final certificate
            device_name = request.data.get("device_name")
            device_key = DeviceKeys.objects.get(name=device_name)

            # Decrypt the hybrid-encrypted certificate
            combined = base64.b64decode(hybrid_signed_cert_and_key)
            # The first 256 bytes will be the RSA-encrypted AES key
            encrypted_aes_key = combined[:256]  # For 2048-bit RSA key
            encrypted_data = combined[256:]

            # Decrypt the AES key using RSA private key
            aes_key = settings.SERVER_PRIVATE_KEY.decrypt(
                encrypted_aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            # Create Fernet instance with decrypted key
            f = Fernet(aes_key)

            # Decrypt the actual certificate data
            dec_signed_cert = f.decrypt(encrypted_data).decode("utf-8")

            cert_data = SignedCertificateData(
                public_key_length=len(device_key.public_key),
                public_key=device_key.public_key,
                signed_certificate=dec_signed_cert,
            )

            final_certificate = serialize_signed_certificate(cert_data)

            # Rest of your existing code...
            final_cert_qr_code = pyqrcode.create(final_certificate)
            buffer = io.BytesIO()
            final_cert_qr_code.png(buffer, scale=8)
            final_cert_qr_code.png(
                "/home/omkar/Desktop/Backend/media/qr_codes", scale=8
            )
            buffer.seek(0)

            # Create watermarked version with embedded certificate
            watermarker = WaveletDCTWatermark()
            # watermarked_image = PILImage.open(
            #     watermarker.fwatermark_image(
            #         original_image=PILImage.open(open(image_object.image.path, "rb")),
            #         watermark=PILImage.open(buffer),
            #     )
            # )

            watermarked_image = PILImage.open(
                watermarker.watermark_image(
                    image_path=image_object.image.path,
                    watermark_path="media/qr_codes/temp.png",
                )
            )

            # Save watermarked version
            Image.objects.create(
                device_key=device_key,
                image=watermarked_image,
                image_hash=calculate_image_hash(watermarked_image),
                verified=True,
            )

            # Generate download link
            download_url = reverse("image-download", args=[image_object.id])

            return Response(
                {
                    "message": "Image verified successfully",
                    "download_url": download_url,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ImageDownloadView(APIView):
    def get(self, request, image_id):
        try:
            # Get the image object
            image = Image.objects.get(id=image_id)

            if not os.path.exists(image.image.path):
                return Response(
                    {"error": "Image file not found"}, status=status.HTTP_404_NOT_FOUND
                )

            try:
                file_handle = open(image.image.path, "rb")
            except:
                return Response(
                    {"error": "Image not found", "message": str(e)},
                    status=status.HTTP_404_NOT_FOUND,
                )

            response = FileResponse(file_handle)

            response["Content-Type"] = (
                "image/jpeg"  # Considering only image/jpg for now
            )
            response["Content-Disposition"] = (
                f'attachment; filename="{os.path.basename(image.image.name)}"'
            )

            return response

        except Image.DoesNotExist:
            return Response(
                {"error": "Image not found"}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ImageListView(generics.ListAPIView):
    serializer_class = ImageSerializer
    permission_classes = (permissions.IsAuthenticated,)

    def get_queryset(self):
        return Image.objects.filter(user=self.request.user)


class CalculateImageHashView(generics.CreateAPIView):
    def post(self, request):
        """
        Calculate SHA-256 hash of an uploaded image
        """
        # Check if image file is in request
        if "image" not in request.FILES:
            return Response(
                {"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get the image file from request
            image_file = request.FILES["image"]

            # Calculate hash
            sha256_hash = hashlib.sha256()

            # Read file in chunks to handle large files efficiently
            for chunk in image_file.chunks():
                sha256_hash.update(chunk)

            # Get hexadecimal representation of hash
            image_hash = sha256_hash.hexdigest()

            return Response({"hash": image_hash}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Error calculating hash: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SignHashView(generics.CreateAPIView):
    def post(self, request):
        """
        Sign a hash value using user's private key
        """
        # Get username and hash from request
        username = request.data.get("username")
        hash_value = request.data.get("hash")

        if not username or not hash_value:
            return Response(
                {"error": "Username and hash are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Get user's private key
            user = User.objects.get(username=username)
            user_keys = DeviceKeys.objects.get(user=user)
            private_key_pem = user_keys.private_key

            # Load the private key
            private_key = serialization.load_pem_private_key(
                private_key_pem.encode(), password=None
            )

            # Sign the hash
            signature = private_key.sign(
                hash_value.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            # Convert to base64 for transmission
            signature_b64 = base64.b64encode(signature).decode()

            return Response({"signature": signature_b64}, status=status.HTTP_200_OK)

        except User.DoesNotExist:
            return Response(
                {"error": "User not found"}, status=status.HTTP_404_NOT_FOUND
            )
        except DeviceKeys.DoesNotExist:
            return Response(
                {"error": "User keys not found"}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": f"Error signing hash: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class GenerateQRView(generics.CreateAPIView):
    def post(self, request):
        """
        Generate QR code from the first half of a hash value
        """

        # Get string from request
        string_value = request.data.get("string")
        logger.V(3).info(f"string_value: {string_value}")

        if not string_value:
            return Response(
                {"error": "String value is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Create QR code
            qr = pyqrcode.create(string_value)

            # Create a bytes buffer to store the PNG
            buffer = io.BytesIO()
            qr.png(buffer, scale=8)
            buffer.seek(0)

            # Create response with PNG mime type
            response = HttpResponse(buffer.getvalue(), content_type="image/png")
            response["Content-Disposition"] = 'inline; filename="qr_code.png"'

            return response

        except Exception as e:
            return Response(
                {"error": f"Error generating QR code: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class WatermarkImageView(generics.CreateAPIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request: Request):
        """
        Embed QR code watermark into original image
        """
        # Check if both files are in request
        if "original_image" not in request.FILES or "qr_code" not in request.FILES:
            return Response(
                {"error": "Both original image and QR code are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Get files from request
            original_image = request.FILES["original_image"]
            qr_code = request.FILES["qr_code"]

            # Get format from query params, default to None
            output_format = request.data.get("format")

            watermarker = WaveletDCTWatermark()

            # Process watermarking
            watermarked_array = watermarker.fwatermark_image(
                PILImage.open(original_image), PILImage.open(qr_code)
            )

            watermarked_image = PILImage.fromarray(watermarked_array)

            if output_format is None:
                output_format = "JPEG"

            # Save to bytes buffer
            buffer = io.BytesIO()
            watermarked_image.save(buffer, format=output_format.upper())
            buffer.seek(0)

            # Create response
            response = HttpResponse(
                buffer.getvalue(), content_type=f"image/{output_format.lower()}"
            )
            response["Content-Disposition"] = (
                f'inline; filename="watermarked_image.{output_format.lower()}"'
            )

            return response

        except Exception as e:
            return Response(
                {"error": f"Error in watermarking process: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class WatermarkRecoveryView(generics.CreateAPIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        """
        Recover QR code watermark from a watermarked image
        """
        if "image" not in request.FILES:
            return Response(
                {"error": "Watermarked image is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Get watermarked image from request
            watermarked_image = request.FILES["image"]

            # Recover watermark
            watermarker = WaveletDCTWatermark()
            recovered_image_arr = watermarker.frecover_watermark(
                PILImage.open(watermarked_image)
            )

            recovered_image = PILImage.fromarray(recovered_image_arr)

            buffer = io.BytesIO()
            recovered_image.save(buffer, format="JPEG")
            buffer.seek(0)

            response = HttpResponse(buffer.getvalue(), content_type="image/jpeg")
            response["Content-Disposition"] = (
                'inline; filename="recovered_watermark.jpg"'
            )

            return response

        except Exception as e:
            return Response(
                {"error": f"Error in watermark recovery process: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ImageExifView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request: Request):
        """
        Extract EXIF data from an uploaded image
        """
        try:
            # Check if image file is in request
            if "image" not in request.FILES:
                return Response(
                    {"error": "No image file provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            image_file = request.FILES["image"]
            img = PILImage.open(image_file)

            exif_data = extract_exif_data_PIEXIF(img)

            return Response(
                {"message": "EXIF data extracted successfully", "exif_data": exif_data},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"error": f"Error extracting EXIF data: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


def landing_page(request):
    return render(request, "landing.html")
