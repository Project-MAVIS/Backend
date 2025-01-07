import os
from pathlib import Path
from django.http import FileResponse
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from .models import UserKeys, Image
from .serializers import UserSerializer, ImageSerializer
from .utils import generate_key_pair, verify_signature
import base64
from .watermark import WaveletDCTWatermark
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from rest_framework.views import APIView


class RegisterUserView(generics.CreateAPIView):
    serializer_class = UserSerializer
    permission_classes = (permissions.AllowAny,)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        # Generate and save key pair
        private_key, public_key = generate_key_pair()
        UserKeys.objects.create(
            user=user,
            private_key=private_key,
            public_key=public_key
        )
        
        return Response({
            'message': 'Registration successful',
            'user_id': user.id,
            'username': user.username,
            'public_key': public_key,
            'private_key': private_key
        }, status=status.HTTP_201_CREATED)

class ImageUploadView(generics.CreateAPIView):
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            signature = base64.b64decode(request.data.get('signature'))
            image_hash = request.data.get('image_hash')
            username = request.data.get('username')
            
            if not username:
                return Response({
                    'error': 'Username is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            try:
                user = User.objects.get(username=username)
                user_keys = UserKeys.objects.get(user=user)
            except User.DoesNotExist:
                return Response({
                    'error': 'User not found'
                }, status=status.HTTP_404_NOT_FOUND)
            except UserKeys.DoesNotExist:
                return Response({
                    'error': 'User keys not found'
                }, status=status.HTTP_404_NOT_FOUND)

            if verify_signature(user_keys.public_key, signature, image_hash.encode()):
                # Save the original image first
                image = serializer.save(user=user, verified=True)
                
                try:
                    # Setup paths for watermarking
                    media_root = Path(settings.MEDIA_ROOT)
                    watermark_path = media_root / 'qr_codes' / 'default_qr.png'
                    
                    # Create a watermarking instance
                    watermarker = WaveletDCTWatermark(base_path=str(media_root))
                    
                    # Get the path of the saved image
                    original_image_path = image.image.path
                    
                    # Apply watermark
                    watermarked_image = watermarker.watermark_image(
                        original_image_path,
                        str(watermark_path)
                    )
                    
                    # Get the watermarked image path from the watermarker
                    watermarked_image_path = watermarker.result_path / 'image_with_watermark.jpg'
                    
                    # Read the watermarked image and update the model
                    with open(watermarked_image_path, 'rb') as f:
                        image.image.save(
                            f'watermarked_{Path(original_image_path).name}',
                            ContentFile(f.read()),
                            save=True
                        )
                    
                    # Clean up temporary files if needed
                    if os.path.exists(watermarked_image_path):
                        os.remove(watermarked_image_path)
                    
                    return Response({
                        'message': 'Image uploaded, verified, and watermarked successfully',
                        'image_id': image.id,
                        'verified': True,
                        'image_url': request.build_absolute_uri(image.image.url)
                    }, status=status.HTTP_201_CREATED)
                
                except Exception as e:
                    return Response({
                        'error': f'Error during watermarking: {str(e)}'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            return Response({
                'error': 'Invalid signature'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)


class ImageDownloadView(APIView):
    def get(self, request, image_id):
        try:
            # Get the image object
            image = Image.objects.get(id=image_id)
            
            if not os.path.exists(image.image.path):
                return Response(
                    {"error": "Image file not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            try:
                file_handle = open(image.image.path, 'rb')
            except:
                return Response(
                {"error": "Image not found", "message": str(e)}, 
                status=status.HTTP_404_NOT_FOUND
            )
            
            response = FileResponse(file_handle)
            
            response['Content-Type'] = 'image/jpeg'  # Considering only image/jpg for now
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(image.image.name)}"'
            
            return response
            
        except Image.DoesNotExist:
            return Response(
                {"error": "Image not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ImageListView(generics.ListAPIView):
    serializer_class = ImageSerializer
    permission_classes = (permissions.IsAuthenticated,)

    def get_queryset(self):
        return Image.objects.filter(user=self.request.user)