from django.test import TestCase
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status

from .utils import *


class UtilsTests(TestCase):
    def test_generate_key_pair(self):
        # Generate key pair
        private_pem, public_pem = generate_key_pair()

        # Test 1: Verify return types are strings
        self.assertIsInstance(private_pem, str)
        self.assertIsInstance(public_pem, str)

        # Test 2: Verify PEM format
        self.assertTrue(private_pem.startswith("-----BEGIN PRIVATE KEY-----"))
        self.assertTrue(private_pem.endswith("-----END PRIVATE KEY-----\n"))
        self.assertTrue(public_pem.startswith("-----BEGIN PUBLIC KEY-----"))
        self.assertTrue(public_pem.endswith("-----END PUBLIC KEY-----\n"))

        # Test 3: Verify keys can be loaded by cryptography library
        try:
            private_key = serialization.load_pem_private_key(
                private_pem.encode(), password=None, backend=default_backend()
            )
            public_key = serialization.load_pem_public_key(
                public_pem.encode(), backend=default_backend()
            )
        except Exception as e:
            self.fail(f"Failed to load generated keys: {str(e)}")

        # Test 4: Verify key size is 2048 bits
        self.assertEqual(private_key.key_size, 2048)
        self.assertEqual(public_key.key_size, 2048)

    def test_calculate_image_hash(self):
        img = Image.open("data/samples/jpeg/Wadapav.jpeg")

        # Test 1: Verify return type is string
        self.assertIsInstance(calculate_image_hash(img), str)

        # Test 2: Verify hash is correct
        self.assertEqual(
            calculate_image_hash(img),
            # Calculated independently
            "b1081a80c0d45eb442ee85bdfc86459ca4770bb482496bf62a33365ae125ebb5",
        )

    def test_calculate_string_hash(self):
        # Test 1: Verify return type is string
        self.assertIsInstance(calculate_string_hash("Project MAVIS"), str)

        # Test 2: Verify hash is correct
        self.assertEqual(
            calculate_string_hash("Project MAVIS"),
            "d33aeb62f8d14f71f6b847594c5023d473040edbdf8bf839f15bf110c2cb2999",
        )


class EndpointTests(APITestCase):
    def test_ping(self):
        response = self.client.get(reverse("ping"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, "pong")
