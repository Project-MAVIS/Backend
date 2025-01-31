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

# QR Code related
import pyqrcode

# Image processing
from PIL import Image as PILImage

# Local imports
from .models import UserKeys, Image
from .serializers import UserSerializer, ImageSerializer
from .utils import generate_key_pair, verify_signature
from .watermark import WaveletDCTWatermark


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
        UserKeys.objects.create(
            user=user, private_key=private_key, public_key=public_key
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
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            signature = base64.b64decode(request.data.get("signature"))
            image_obj = serializer.validated_data["image"]

            image_hash = hashlib.sha256(image_obj.read()).hexdigest()
            # image_hash = request.data.get('image_hash')

            username = request.data.get("username")

            if not username:
                return Response(
                    {"error": "Username is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                user = User.objects.get(username=username)
                user_keys = UserKeys.objects.get(user=user)
            except User.DoesNotExist:
                return Response(
                    {"error": "User not found"}, status=status.HTTP_404_NOT_FOUND
                )
            except UserKeys.DoesNotExist:
                return Response(
                    {"error": "User keys not found"}, status=status.HTTP_404_NOT_FOUND
                )

            # If Image hash is already present in the models and is verified, then there immediately send the link from here
            # =======================# UNCOMMENT BEFORE PROD #=======================
            # try:
            #     prev_image = Image.objects.get(image_hash=image_hash, user= user)
            #     print("Image already exists")
            #     return Response({
            #                 'message': 'Already exists.',
            #                 'image_id': (prev_image.id + 1),
            #                 'verified': True,
            #                 'image_url': request.build_absolute_uri(reverse('download-image', kwargs={'image_id': prev_image.id}))

            #             }, status=status.HTTP_200_OK)
            # except Exception as e:
            #     print(f"New image, {str(e)}")
            # #=======================#=======================#=======================

            if verify_signature(user_keys.public_key, signature, image_hash.encode()):
                # Save the original image first
                image = serializer.save(user=user, verified=False)

                try:
                    # Setup paths for watermarking
                    media_root = Path(settings.MEDIA_ROOT)

                    s = image_hash[: len(image_hash) // 2]
                    url = pyqrcode.create(s)

                    default_qr = f"qr_for_{username}_{datetime.now()}"
                    url.png(f"{media_root}/qr_codes/{default_qr}.png", scale=8)
                    watermark_path = f"{media_root}/qr_codes/{default_qr}.png"

                    # Create a watermarking instance
                    watermarker = WaveletDCTWatermark()

                    # Get the path of the saved image
                    original_image_path = image.image.path

                    # Apply watermark
                    watermarked_image = watermarker.watermark_image(
                        original_image_path, str(watermark_path)
                    )

                    # Get the watermarked image path from the watermarker
                    watermarked_image_path = (
                        watermarker.result_path / "image_with_watermark.jpg"
                    )
                    print(f"watermarked_image_path > {watermarked_image_path}")

                    if os.path.exists(watermarked_image_path):
                        print("Image exists")
                    else:
                        print("Image not exist")

                    # Read the watermarked image and update the model
                    with open(watermarked_image_path, "rb") as f:
                        print("Saving image")
                        image_file = File(
                            f, name="image_with_watermark.png"
                        )  # 'name' is the filename to store

                        Image.objects.create(
                            user=user,
                            image=image_file,
                            image_hash=s,
                            verified=True,
                        )
                        print("Image saved")

                    # Clean up temporary files if needed
                    if os.path.exists(watermarked_image_path):
                        os.remove(watermarked_image_path)

                    return Response(
                        {
                            "message": "Image uploaded, verified, and watermarked successfully",
                            "image_id": (image.id + 1),
                            "verified": True,
                            # 'image_url': request.build_absolute_uri(f"images/{image.id}")
                            "image_url": request.build_absolute_uri(
                                reverse("download-image", kwargs={"image_id": image.id})
                            ),
                        },
                        status=status.HTTP_201_CREATED,
                    )

                except Exception as e:
                    return Response(
                        {"error": f"Error during watermarking: {str(e)}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )

            return Response(
                {"error": "Invalid signature"}, status=status.HTTP_400_BAD_REQUEST
            )

        except Exception as e:
            print("Error here")
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


class CalculateImageHashView(APIView):
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


class SignHashView(APIView):
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
            user_keys = UserKeys.objects.get(user=user)
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
        except UserKeys.DoesNotExist:
            return Response(
                {"error": "User keys not found"}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": f"Error signing hash: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class GenerateQRView(APIView):
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


class WatermarkImageView(APIView):
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


class WatermarkRecoveryView(APIView):
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
