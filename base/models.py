from django.db import models
from django.contrib.auth.models import User
import os
from django.db.models.signals import pre_delete
from django.dispatch import receiver

class DeviceKeys(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.TextField()
    public_key = models.TextField()
    private_key = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.user.username

class Image(models.Model):
    device_key = models.ForeignKey(DeviceKeys, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/')
    image_hash = models.TextField()
    original_image_hash = models.TextField(null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    verified = models.BooleanField(default=False)

    def __str__(self) -> str:
        return f"{self.device_key.user.username} -> {self.uploaded_at}"

@receiver(pre_delete, sender=Image)
def delete_image_file(sender, instance, **kwargs):
    """Delete the image file when the Image model instance is deleted."""
    if instance.image:
        # Check if the file exists
        if os.path.isfile(instance.image.path):
            os.remove(instance.image.path)  # Delete the file