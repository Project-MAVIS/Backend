#!/bin/bash

# Create .keys directory if it doesn't exist
sh scripts/keys.sh

pip install --upgrade pip

pip install --no-cache-dir -r requirements.txt
