from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.RegisterUserView.as_view(), name='register'),
    path('upload-image/', views.ImageUploadView.as_view(), name='upload-image'),
    path('watermark-extract/<int:image_id>/extract', views.ImageVerifyView.as_view(), name='watermark-extract'),
    path('watermark-extract/', views.ImageVerifyView.as_view(), name='watermark-extract'),
    path('images/', views.ImageListView.as_view(), name='image-list'),
    path('images/<int:image_id>/', views.ImageDownloadView.as_view(), name='download-image'),

    path('truncate', views.truncate, name="truncate")
]