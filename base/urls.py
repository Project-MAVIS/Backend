from django.urls import path
from . import views
from .views import DeviceRegistrationView

# Black formatter is disabled here because it messes up the formatting of the code
# fmt: off
urlpatterns = [
    # Setup
    path('ping', views.ping, name="ping"),
    path('register-user/', views.UserRegistrationView.as_view(), name='register-user'),
    path('register-device/', DeviceRegistrationView.as_view(), name='register-device'),

    path('upload-image/', views.ImageUploadView.as_view(), name='upload-image'),
    path('verify/', views.ImageVerifierView.as_view(), name="ImageVerifier"),

    path('register/', views.Demo_RegisterUserView.as_view(), name='register'),
    path('get-image/', views.ImageLinkProvider.as_view(), name='get-image'),
    path('images/', views.ImageListView.as_view(), name='image-list'),
    path('images/<int:image_id>/', views.ImageDownloadView.as_view(), name='download-image'),
    path('calculate-hash/', views.CalculateImageHashView.as_view(), name='calculate-hash'),
    path('sign-hash/', views.SignHashView.as_view(), name='sign-hash'),
    path('generate-qr/', views.GenerateQRView.as_view(), name='generate-qr'),
    path('watermark-image/', views.WatermarkImageView.as_view(), name='watermark-image'),
    path('recover-watermark/', views.WatermarkRecoveryView.as_view(), name='recover-watermark'),
    path('truncate', views.truncate, name="truncate"),
    path('extract-exif/', views.ImageExifView.as_view(), name='extract-exif'),
]
