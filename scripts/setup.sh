#!/bin/bash

# Create .keys directory if it doesn't exist
sh scripts/keys.sh

pip install --upgrade pip

pip install --no-cache-dir cryptography django django-rest-framework numpy pywavelets scipy pillow pyqrcode pypng opencv-python pyzbar python-dotenv exif pytz piexif
