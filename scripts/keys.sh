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
