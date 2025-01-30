import os
from django.urls import reverse
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
import pyqrcode 
import png 
from pyqrcode import QRCode 
from datetime import datetime
from django.core.files import File
from django.conf import settings
import hashlib
from rest_framework.decorators import api_view

@api_view(['GET'])
def truncate(request):
    try:
        Image.objects.all().delete()
    except Exception as e:
        return Response({"Status":"Exception", "message":str(e)})
    return Response({"Status":"Complete"})

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

class ImageVerifyView(generics.CreateAPIView):
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        try:
                
            # Check if the request contains the file
            if 'image' not in request.FILES:
                return Response({'error': 'No file uploaded'}, status=400)
            
            file = request.FILES['image']

            # Define the directory where you want to save the uploaded file
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'verify')

            # Ensure the directory exists
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            # Save the file to the directory
            file_path = os.path.join(upload_dir, file._name)
            with open(file_path, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)

            watermarker = WaveletDCTWatermark()
            watermarker.recover_watermark(image_path=f'{settings.MEDIA_ROOT}/verify/{file._name}')
            
            hash = watermarker.read_qr_code(f'{settings.MEDIA_ROOT}/result/recovered_watermark.jpg')
                    
            values = Image.objects.filter(image_hash=hash[0]['data'])
            # Return a success response
            if len(values) >= 1:
                return Response({
                    "status": "verified"
                })
            else:
                return Response({
                    "status": "Not verified"
                })
        except Exception as e:
            return Response({"message": f"There was an error: {str(e)}"}, status=200)

class ImageUploadView(generics.CreateAPIView):
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            signature = base64.b64decode(request.data.get('signature'))
            image_obj = serializer.validated_data['image']

            image_hash = hashlib.sha256(image_obj.read()).hexdigest()
            # image_hash = request.data.get('image_hash')            
            
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

                    s = image_hash[:len(image_hash)//2]
                    url = pyqrcode.create(s)   

                    default_qr = f"qr_for_{username}_{datetime.now()}"
                    url.png(f'{media_root}/qr_codes/{default_qr}.png', scale = 8)
                    watermark_path = f'{media_root}/qr_codes/{default_qr}.png'
                    
                    # Create a watermarking instance
                    watermarker = WaveletDCTWatermark()
                    
                    # Get the path of the saved image
                    original_image_path = image.image.path
                    
                    # Apply watermark
                    watermarked_image = watermarker.watermark_image(
                        original_image_path,
                        str(watermark_path)
                    )
                    
                    # Get the watermarked image path from the watermarker
                    watermarked_image_path = watermarker.result_path / 'image_with_watermark.jpg'
                    print(f"watermarked_image_path > {watermarked_image_path}")
                    
                    if os.path.exists(watermarked_image_path):
                        print("Image exists")
                    else:
                        print("Image not exist")

                    # Read the watermarked image and update the model                    
                    with open(watermarked_image_path, 'rb') as f:
                        print("Saving image")
                        image_file = File(f, name='image_with_watermark.png')  # 'name' is the filename to store

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
                    
                    return Response({
                        'message': 'Image uploaded, verified, and watermarked successfully',
                        'image_id': (image.id + 1),
                        'verified': True,
                        # 'image_url': request.build_absolute_uri(f"images/{image.id}")
                        'image_url': request.build_absolute_uri(reverse('download-image', kwargs={'image_id': image.id}))

                    }, status=status.HTTP_201_CREATED)
                
                except Exception as e:
                    return Response({
                        'error': f'Error during watermarking: {str(e)}'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            return Response({
                'error': 'Invalid signature'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        except Exception as e:
            print("Error here")
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