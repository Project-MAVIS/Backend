from rest_framework import serializers
from django.contrib.auth.models import User
from .models import DeviceKeys, Image
from datetime import datetime


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ("id", "username", "email", "password")

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"],
        )
        return user


class ImageSerializer(serializers.ModelSerializer):
    device_name = serializers.CharField(write_only=True)

    class Meta:
        model = Image
        fields = ("id", "image", "image_hash", "verified", "uploaded_at", "device_name")
        read_only_fields = ("verified", "uploaded_at")

    def create(self, validated_data):
        # Remove username from validated_data as it's not a model field
        device_name = validated_data.pop("device_name")
        device_key = DeviceKeys.objects.get(name=device_name)
        image_file = validated_data["image"]
        original_filename = image_file.name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_filename = f"{device_name}_{timestamp}_{original_filename}"
        image_file.name = new_filename
        return super().create(validated_data)
