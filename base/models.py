from django.db import models
from django.contrib.auth.models import User
import os
from django.db.models.signals import pre_delete
from django.dispatch import receiver
import boto3


class DeviceKeys(models.Model):
    name = models.TextField()
    uuid = models.TextField()
    public_key = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.user.username


class Image(models.Model):
    device_key = models.ForeignKey(DeviceKeys, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="images/")
    image_hash = models.TextField()
    original_image_hash = models.TextField(null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    verified = models.BooleanField(default=False)

    def __str__(self) -> str:
        return f"{self.device_key.uuid} -> {self.uploaded_at}"
    
    def save(self, *args, **kwargs):
        """
        Upload the image to Supabase Storage when saving the model.
        """
        try:
            print("SAVING HERE")
            # Call the parent save method
            super().save(*args, **kwargs)
            print("SAVED")
        except Exception as e:
            print(f"❌ Error during save: {e}")
            raise

        # Check if we have a file in memory
        if self.image and hasattr(self.image, 'file') and self.image.file:
            try:
                import copy
                image = copy.deepcopy(self.image)
                # Upload to Supabase
                self.upload_to_supabase(image)

                # Reset file position for Django to save it properly
                # if hasattr(self.image.file, 'seek') and callable(self.image.file.seek):
                #     self.image.file.seek(0)
            except Exception as e:
                print(f"❌ Error during upload: {e}")
        else:
            print("⚠️ No valid image file found for upload")
        
        print(f"✓ Image record saved: {self.id}")
        

    def upload_to_supabase(self, image):
        """
        Uploads the image to Supabase Storage using boto3.
        """
        # Verify the file exists and can be read
        if not hasattr(image, 'file') or not image.file:
            raise ValueError(f"No valid file handle for {image.name}")
            
        # Create S3 session
        session = boto3.session.Session()
        s3_client = session.client(
            "s3",
            aws_access_key_id=os.getenv("SUPABASE_S3_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SUPABASE_S3_ACCESS_KEY"),
            endpoint_url=os.getenv("SUPABASE_S3_ENDPOINT"),  
            region_name=os.getenv("SUPABASE_S3_REGION_NAME"),
        )

        bucket_name = os.getenv("SUPABASE_S3_BUCKET_NAME")
        
        # Use basename to avoid absolute path issues
        filename = os.path.basename(image.name)
        image_path = f"images/{filename}"  # Path inside bucket

        try:
            # Make sure we're at the beginning of the file
            # if hasattr(image.file, 'seek') and callable(image.file.seek):
            #     image.file.seek(0)
                
            # Upload file to Supabase Storage
            s3_client.upload_fileobj(
                image.file,
                bucket_name,
                image_path,
                ExtraArgs={"ContentType": getattr(image.file, 'content_type', 'image/jpeg')}
            )
            print(f"✅ Uploaded {filename} to {bucket_name}/{image_path}")
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            # raise            

@receiver(pre_delete, sender=Image)
def delete_image_file(sender, instance, **kwargs):
    """Delete the image file when the Image model instance is deleted."""
    if instance.image:
        # Check if the file exists
        if os.path.isfile(instance.image.path):
            os.remove(instance.image.path)  # Delete the file
