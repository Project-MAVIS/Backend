#!/bin/bash

# Create .keys directory if it doesn't exist
sh scripts/keys.sh

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

pip install cryptography django django-rest-framework numpy pywavelets scipy pillow pyqrcode pypng opencv-python pyzbar python-dotenv exif pytz piexif
pip install ipykernel pre-commit black ipython ipykernel requests

sudo apt install libzbar0