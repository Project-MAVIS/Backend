#!/bin/bash

# Create .keys directory if it doesn't exist
mkdir -p .keys

# Generate RSA private key
openssl genpkey -algorithm RSA -out .keys/private.key -pkeyopt rsa_keygen_bits:2048

# Extract public key from private key
openssl rsa -pubout -in .keys/private.key -out .keys/public.key

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
