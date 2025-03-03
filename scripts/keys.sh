#!/bin/bash

# Create .keys directory if it doesn't exist
mkdir -p .keys

# Generate ECC private key using P-256 curve
openssl ecparam -name prime256v1 -genkey -noout -out .keys/private.key

# Extract public key from private key
openssl ec -in .keys/private.key -pubout -out .keys/public.key

# Set appropriate permissions
chmod 600 .keys/private.key
chmod 644 .keys/public.key

echo "Key pair generated successfully:"
echo "Private key: .keys/private.key"
echo "Public key:  .keys/public.key"

# Base 64 Encode the keys and add them to the .env file
echo "SERVER_PRIVATE_KEY=$(cat .keys/private.key | base64 -w 0)" >>.env
echo "SERVER_PUBLIC_KEY=$(cat .keys/public.key | base64 -w 0)" >>.env

echo "Keys added to .env file"
