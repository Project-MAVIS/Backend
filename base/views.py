# Python standard library
import io
import os
import base64
import hashlib
from pathlib import Path
from datetime import datetime

# Django imports
from django.urls import reverse
from django.conf import settings
from django.http import FileResponse, HttpResponse
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.core.files import File

# Django REST Framework
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.request import Request

# Cryptography
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet

# QR Code related
import pyqrcode

# Image processing
import numpy as np
from PIL import Image as PILImage

# Local imports
from .models import DeviceKeys, Image
from .serializers import UserSerializer, ImageSerializer
from .utils import *
from .watermark import WaveletDCTWatermark
from .certificate import *


@api_view(["GET"])
def ping(_) -> Response:
    return Response("pong")


@api_view(["GET"])
def truncate(_):
    try:
        Image.objects.all().delete()
    except Exception as e:
        return Response({"Status": "Exception", "message": str(e)})
    return Response({"Status": "Complete"})


class RegisterUserView(generics.CreateAPIView):
    serializer_class = UserSerializer
    permission_classes = (permissions.AllowAny,)

    def create(self, request: Request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # Generate and save key pair
        private_key, public_key = generate_key_pair()
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


class ImageVerifyView(generics.CreateAPIView):
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request: Request):
        try:

            # Check if the request contains the file
            if "image" not in request.FILES:
                return Response({"error": "No file uploaded"}, status=400)

            file = request.FILES["image"]

            # Define the directory where you want to save the uploaded file
            upload_dir = os.path.join(settings.MEDIA_ROOT, "verify")

            # Ensure the directory exists
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            # Save the file to the directory
            file_path = os.path.join(upload_dir, file._name)
            with open(file_path, "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)

            watermarker = WaveletDCTWatermark()
            watermarker.recover_watermark(
                image_path=f"{settings.MEDIA_ROOT}/verify/{file._name}"
            )

            hash = watermarker.read_qr_code(
                f"{settings.MEDIA_ROOT}/result/recovered_watermark.jpg"
            )

            values = Image.objects.filter(image_hash=hash[0]["data"])
            # Return a success response
            if len(values) >= 1:
                return Response({"status": "verified"})
            else:
                return Response({"status": "Not verified"})
        except Exception as e:
            return Response({"message": f"There was an error: {str(e)}"}, status=200)


class ImageUploadView(generics.CreateAPIView):
    # Device create image hash, device sign image hash, server verifies signature and saves image
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        # #=======================#=======================#=======================
        serializer.is_valid(raise_exception=True)
        # #=======================#=======================#=======================

        try:
            signature = base64.b64decode(request.data.get("image_hash"))
            image_obj = serializer.validated_data["image"]
            image_hash = hashlib.sha256(image_obj.read()).hexdigest()
            # image_hash = request.data.get('image_hash')

            device_name = request.data.get("device_name")

            if not device_name:
                return Response(
                    {"error": "device name is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                # user = User.objects.get(username=device_name)
                device_key = DeviceKeys.objects.get(name=device_name)
            except DeviceKeys.DoesNotExist:
                return Response(
                    {"error": "DeviceKeys not found"}, status=status.HTTP_404_NOT_FOUND
                )

            if verify_signature(device_key.public_key, signature, image_hash.encode()):
                # Save the original image first

                image_object = serializer.save(
                    device_key=device_key, 
                    verified=False, 
                    image_hash=image_hash
                )

                # # Signs the image_hash
                # server_signed_hash = encrypt_string(
                #     image_hash, settings.SERVER_PUBLIC_KEY
                # )

                # Make the certificate, sign the certificate
                certificate = create_certificate(image_object, device_key)
                signed_certificate = encrypt_string(
                    certificate, settings.SERVER_PUBLIC_KEY
                )
                # then in the response send the signed certficate and signed hash

                final_cert_qr_code = pyqrcode.create(signed_certificate)
                buffer = io.BytesIO()
                final_cert_qr_code.png(buffer, scale=8)
                buffer.seek(0)

                # Create watermarked version with embedded certificate
                watermarker = WaveletDCTWatermark()
                watermarked_image = PILImage.open(
                    watermarker.fwatermark_image(
                        original_image=PILImage.open(image_object.image.read()),
                        watermark=PILImage.open(buffer),
                    )
                )

                # Save watermarked version
                Image.objects.create(
                    device_key=device_key,
                    image=watermarked_image,
                    image_hash=calculate_image_hash(watermarked_image),
                    original_image_hash = image_hash,
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
            return Response(
                {"error": "Invalid signature"}, status=status.HTTP_400_BAD_REQUEST
            )
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


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
            buffer.seek(0)

            # Create watermarked version with embedded certificate
            watermarker = WaveletDCTWatermark()
            watermarked_image = PILImage.open(
                watermarker.fwatermark_image(
                    original_image=PILImage.open(image_object.image),
                    watermark=PILImage.open(buffer),
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

        # Get hash from request
        hash_value = request.data.get("hash")

        if not hash_value:
            return Response(
                {"error": "Hash value is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get first half of hash
            half_hash = hash_value[: len(hash_value) // 2]

            # Create QR code
            qr = pyqrcode.create(half_hash)

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
