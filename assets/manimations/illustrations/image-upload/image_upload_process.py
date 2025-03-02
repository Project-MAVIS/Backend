from manim import *
import numpy as np


class ImageUploadAndSignatureVerificationProcess(Scene):
    def construct(self):
        # Title
        title = Text("Image Upload & Signature Verification Process", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Step 1: Fetch Image, Signed Hash, and Public Key
        step1_title = Text(
            "Step 1: Fetch Image, Signed Hash & Public Key", font_size=28
        )
        step1_title.to_edge(UP)
        self.play(Write(step1_title))

        # Create phone and server
        phone = RoundedRectangle(height=4, width=2, corner_radius=0.2)
        phone_label = Text("User Device", font_size=20).next_to(phone, DOWN)
        phone_group = VGroup(phone, phone_label).shift(LEFT * 4)

        server = Rectangle(height=3, width=2)
        server_label = Text("Server", font_size=20).next_to(server, DOWN)
        server_group = VGroup(server, server_label).shift(RIGHT * 4)

        self.play(Create(phone_group), Create(server_group))

        # Create components of the request
        image_component = Rectangle(height=2, width=2, color=WHITE)
        image_text = Text("Image", font_size=20)
        image_text.move_to(image_component.get_center())

        signed_hash_component = Rectangle(height=0.5, width=2, color=WHITE)
        signed_hash_component.next_to(image_component, DOWN, buff=0.2)
        signed_hash_text = Text("Signed Hash", font_size=16)
        signed_hash_text.move_to(signed_hash_component.get_center())

        public_key_component = Rectangle(height=0.5, width=2, color=WHITE)
        public_key_component.next_to(signed_hash_component, DOWN, buff=0.2)
        public_key_text = Text("Public Key", font_size=16)
        public_key_text.move_to(public_key_component.get_center())

        # Group all components
        request_contents = VGroup(
            image_component,
            image_text,
            signed_hash_component,
            signed_hash_text,
            public_key_component,
            public_key_text,
        )
        request_contents.next_to(phone, RIGHT, buff=1)

        # Animate components appearing one by one
        self.play(Create(image_component), Write(image_text))
        self.play(Create(signed_hash_component), Write(signed_hash_text))
        self.play(Create(public_key_component), Write(public_key_text))

        # Animate request moving to server
        self.play(request_contents.animate.next_to(server, LEFT, buff=0.5))
        self.wait(1)

        # Step 2: Verify Signed Hash - IMPROVED LAYOUT
        self.play(FadeOut(step1_title))
        step2_title = Text("Step 2: Verify Signed Hash with Public Key", font_size=28)
        step2_title.to_edge(UP)
        self.play(Write(step2_title))

        # Fade out phone to create more space
        self.play(FadeOut(phone_group))

        # Move server and components to better positions
        self.play(
            server_group.animate.move_to(RIGHT * 5),
            request_contents.animate.move_to(LEFT * 2),
        )

        # Create verification process visualization with better spacing
        verification_box = Rectangle(height=1.5, width=3, color=YELLOW)
        verification_box.move_to(
            RIGHT * 1.5 + UP * 1.5
        )  # Positioned above and between components
        verification_text = Text("Verification", font_size=20)
        verification_text.move_to(verification_box.get_center())

        # Arrows showing the verification process - better positioned
        arrow1 = Arrow(
            signed_hash_component.get_right(),
            verification_box.get_left() + DOWN * 0.5,
            buff=0.1,
        )
        arrow2 = Arrow(
            public_key_component.get_right(),
            verification_box.get_left() + DOWN * 0.8,
            buff=0.1,
        )

        self.play(Create(verification_box), Write(verification_text))

        self.play(Create(arrow1), Create(arrow2))

        # Show verification result with better positioning
        verification_result = Text("✓ Verified", font_size=24, color=GREEN)
        verification_result.next_to(verification_box, RIGHT, buff=0.5)
        self.play(Write(verification_result))

        # Add a connecting arrow from verification to server
        verify_to_server_arrow = Arrow(
            verification_box.get_right(), server.get_left() + UP * 0.5, buff=0.1
        )
        self.play(Create(verify_to_server_arrow))

        self.wait(1)

        self.play(
            FadeOut(step2_title),
            FadeOut(verification_box),
            FadeOut(verification_text),
            FadeOut(arrow1),
            FadeOut(arrow2),
            FadeOut(verification_result),
            FadeOut(verify_to_server_arrow),
            FadeOut(server_group),
            FadeOut(request_contents),
        )


class MavisCertificate(Scene):
    def construct(self):
        # Step 3: Create MAVIS Certificate
        step3_title = Text("Step 3: Create MAVIS Certificate", font_size=28)
        step3_title.to_edge(UP)
        self.play(Write(step3_title))

        # Certificate components
        cert_components = [
            ("cert_len", "Certificate Length"),
            ("timestamp", "Timestamp"),
            ("image_id", "Image ID"),
            ("user_id", "User ID"),
            ("device_id", "Device ID"),
            ("username", "Username"),
            ("device_name", "Device Name"),
        ]

        # Create certificate structure
        cert_box = Rectangle(height=4.5, width=3, color=BLUE)
        cert_box.move_to(ORIGIN)
        cert_title = Text("MAVIS Certificate", font_size=24)
        cert_title.next_to(cert_box, UP, buff=0.2)

        # Create individual fields
        field_boxes = []
        field_texts = []

        for i, (field_id, field_name) in enumerate(cert_components):
            field_box = Rectangle(height=0.5, width=2.8, color=WHITE)
            field_box.move_to(cert_box.get_center() + UP * (1.8 - i * 0.6))
            field_text = Text(field_name, font_size=16)
            field_text.move_to(field_box.get_center())

            field_boxes.append(field_box)
            field_texts.append(field_text)

        # Animate certificate creation
        self.play(Create(cert_box), Write(cert_title))

        for box, text in zip(field_boxes, field_texts):
            self.play(Create(box), Write(text))

        # Show hex encoding
        hex_text = Text("Encoded as Hex: 0x7a3f...", font_size=16, color=YELLOW)
        hex_text.next_to(cert_box, DOWN, buff=0.3)
        self.play(Write(hex_text))
        self.wait(1)

        self.play(
            FadeOut(step3_title),
            *[FadeOut(mob) for mob in field_texts],
            *[FadeOut(mob) for mob in field_boxes],
            FadeOut(hex_text),
            FadeOut(cert_box),
            FadeOut(cert_title),
        )

        step4_title = Text("Step 4: Certificate Binary Structure", font_size=28)
        step4_title.to_edge(UP)
        self.play(Write(step4_title))

        # Binary packet visualization
        binary_fields = [
            ("1 byte", "cert_len"),
            ("8 bytes", "timestamp"),
            ("8 bytes", "image_id"),
            ("4 bytes", "user_id"),
            ("4 bytes", "device_id"),
            ("1 byte", "username_len"),
            ("var bytes", "username"),
            ("1 byte", "device_name_len"),
            ("var bytes", "device_name"),
        ]

        # Create binary packet visualization
        packet_boxes = []
        packet_labels = []
        packet_sizes = []

        total_width = 10
        current_x = -total_width / 2

        for size_text, field_name in binary_fields:
            # Determine width based on byte size
            if "var" in size_text:
                width = 1.5
            elif "1 byte" in size_text:
                width = 0.8
            elif "4 bytes" in size_text:
                width = 1.2
            else:  # 8 bytes
                width = 1.6

            box = Rectangle(height=0.8, width=width, color=WHITE)
            box.move_to(np.array([current_x + width / 2, 0, 0]))

            field_label = Text(field_name, font_size=12)
            field_label.move_to(box.get_center())

            size_label = Text(size_text, font_size=10)
            size_label.next_to(box, DOWN, buff=0.1)

            packet_boxes.append(box)
            packet_labels.append(field_label)
            packet_sizes.append(size_label)

            current_x += width

        packet_group = VGroup(*packet_boxes, *packet_labels, *packet_sizes)
        packet_group.move_to(ORIGIN)

        # Animate packet visualization
        for box, label, size in zip(packet_boxes, packet_labels, packet_sizes):
            self.play(Create(box), Write(label), Write(size), run_time=0.3)

        self.wait(1)

        # Step 5: Calculate Hash and Encrypt Certificate
        self.play(
            FadeOut(step4_title),
            *[FadeOut(mob) for mob in packet_boxes + packet_labels + packet_sizes],
        )


class QRCodeGenerationAndEmbedding(Scene):
    def construct(self):
        step5_title = Text("Step 5: Calculate Hash & Encrypt Certificate", font_size=28)
        step5_title.to_edge(UP)
        self.play(Write(step5_title))

        # Show certificate again
        cert_box_small = Rectangle(height=2, width=1.5, color=BLUE)
        cert_box_small.move_to(LEFT * 3)
        cert_text = Text("Certificate", font_size=16)
        cert_text.move_to(cert_box_small.get_center())

        # Hash calculation
        hash_arrow = Arrow(cert_box_small.get_right(), LEFT * 1, buff=0.1)
        hash_box = Rectangle(height=0.8, width=2, color=GREEN)
        hash_box.move_to(ORIGIN)
        hash_text = Text("Hash", font_size=16)
        hash_text.move_to(hash_box.get_center())

        # Encryption - Create encrypt_box BEFORE referencing it in arrows
        encrypt_box = Rectangle(height=1.2, width=2, color=YELLOW)
        encrypt_box.move_to(RIGHT * 3)
        encrypt_text = Text("Encrypted\nCertificate", font_size=16)
        encrypt_text.move_to(encrypt_box.get_center())

        # Server key
        server_key = Rectangle(height=0.8, width=1.5, color=RED)
        server_key.move_to(UP * 1.5 + RIGHT * 2)
        server_key_text = Text("Server\nPublic Key", font_size=14)
        server_key_text.move_to(server_key.get_center())

        key_arrow = Arrow(server_key.get_bottom(), encrypt_box.get_top(), buff=0.1)

        # Animate hash and encryption
        self.play(Create(cert_box_small), Write(cert_text))
        self.play(Create(hash_arrow), Create(hash_box), Write(hash_text))
        self.play(Create(server_key), Write(server_key_text))
        self.play(
            Create(encrypt_box),
            Write(encrypt_text),
            Create(key_arrow),
        )
        self.wait(1)

        self.play(
            FadeOut(step5_title),
            *[
                FadeOut(mob)
                for mob in [
                    cert_box_small,
                    cert_text,
                    hash_arrow,
                    server_key,
                    server_key_text,
                    key_arrow,
                ]
            ],
        )

        # Step 6: Create QR Code and Watermark
        step6_title = Text("Step 6: Create QR Code & Watermark Image", font_size=28)
        step6_title.to_edge(UP)
        self.play(Write(step6_title))

        # Keep the hash box and text visible from previous step
        # Position them on the left side
        hash_box.move_to(LEFT * 3)
        hash_text.move_to(hash_box.get_center())

        # Add a more detailed hash text to show what's being converted
        hash_value = Text("2fd4e1c67a2d28...", font_size=14, color=GREEN)
        hash_value.next_to(hash_box, DOWN, buff=0.2)

        self.play(
            hash_box.animate.move_to(LEFT * 3),
            hash_text.animate.move_to(LEFT * 3),
            Write(hash_value),
        )

        # QR code from hash
        qr_code = Square(side_length=1.5)
        qr_code.set_fill(BLACK, opacity=1)
        qr_code.set_stroke(WHITE, width=2)

        # Create QR code pattern
        qr_pattern = VGroup()
        for i in range(5):
            for j in range(5):
                if np.random.random() > 0.5:  # Random pattern
                    square = Square(side_length=0.2)
                    square.set_fill(WHITE, opacity=1)
                    square.move_to(
                        qr_code.get_center()
                        + RIGHT * (j - 2) * 0.25
                        + UP * (i - 2) * 0.25
                    )
                    qr_pattern.add(square)

        # Position QR code in center
        qr_group = VGroup(qr_code, qr_pattern)
        qr_group.move_to(ORIGIN)
        qr_label = Text("QR Code", font_size=16)
        qr_label.next_to(qr_group, DOWN, buff=0.2)

        # Add conversion arrow from hash to QR
        hash_to_qr_arrow = Arrow(hash_box.get_right(), qr_code.get_left(), buff=0.1)
        conversion_text = Text("Convert to QR", font_size=14)
        conversion_text.next_to(hash_to_qr_arrow, UP, buff=0.1)

        # Animate the conversion from hash to QR code
        self.play(Create(hash_to_qr_arrow), Write(conversion_text))
        self.play(Create(qr_code))
        self.play(Create(qr_pattern))
        self.play(Write(qr_label))

        # Fade out the hash elements after conversion
        self.play(
            FadeOut(hash_box),
            FadeOut(hash_text),
            FadeOut(hash_value),
            FadeOut(hash_to_qr_arrow),
            FadeOut(conversion_text),
        )

        # Original image
        image_rect = Rectangle(height=3, width=3, color=WHITE)
        image_rect.move_to(RIGHT * 3)
        image_label = Text("Original Image", font_size=16)
        image_label.next_to(image_rect, UP, buff=0.2)

        # Watermarking process
        watermark_arrow = Arrow(qr_group.get_right(), image_rect.get_left(), buff=0.1)
        watermark_text = Text("DCT Watermarking", font_size=16)
        watermark_text.next_to(watermark_arrow, UP, buff=0.1)

        # Animate image creation and watermarking
        self.play(Create(image_rect), Write(image_label))
        self.play(Create(watermark_arrow), Write(watermark_text))

        # Show watermarked image
        watermarked_image = Rectangle(height=3, width=3, color=BLUE_A)
        watermarked_image.move_to(image_rect.get_center())
        watermarked_label = Text("Watermarked Image", font_size=16, color=BLUE)
        watermarked_label.next_to(watermarked_image, UP, buff=0.2)

        self.play(
            Transform(image_rect, watermarked_image),
            Transform(image_label, watermarked_label),
        )
        self.wait(1)

        self.play(
            FadeOut(step6_title),
            *[
                FadeOut(mob)
                for mob in [qr_group, qr_label, watermark_arrow, watermark_text]
            ],
        )


class Extras(Scene):
    def construct(self):
        # Step 7: Attach Encrypted Certificate as EXIF
        step7_title = Text("Step 7: Attach Encrypted Certificate as EXIF", font_size=28)
        step7_title.to_edge(UP)
        self.play(Write(step7_title))

        # Show EXIF data structure
        exif_box = Rectangle(height=3.5, width=4, color=YELLOW)
        exif_box.move_to(LEFT * 3)
        exif_title = Text("EXIF Data", font_size=20)
        exif_title.next_to(exif_box, UP, buff=0.2)

        # EXIF fields
        exif_fields = [
            "Camera Model",
            "Date Taken",
            "GPS Data",
            "Encrypted Certificate",
        ]

        exif_field_boxes = []
        exif_field_texts = []

        for i, field in enumerate(exif_fields):
            field_box = Rectangle(height=0.6, width=3.5, color=WHITE)
            field_box.move_to(exif_box.get_center() + UP * (1.2 - i * 0.7))
            field_text = Text(field, font_size=16)
            field_text.move_to(field_box.get_center())

            # Highlight the certificate field
            if i == 3:
                field_box.set_color(RED)
                field_text.set_color(WHITE)

            exif_field_boxes.append(field_box)
            exif_field_texts.append(field_text)

        # Animate EXIF creation
        self.play(Create(exif_box), Write(exif_title))

        for box, text in zip(exif_field_boxes, exif_field_texts):
            self.play(Create(box), Write(text))

        # Arrow from encrypted certificate to EXIF
        cert_to_exif_arrow = Arrow(
            encrypt_box.get_left(), exif_field_boxes[3].get_right(), buff=0.1
        )
        self.play(Create(cert_to_exif_arrow))

        # Attach EXIF to image
        exif_to_image_arrow = Arrow(
            exif_box.get_right(), image_rect.get_left(), buff=0.1
        )
        self.play(Create(exif_to_image_arrow))

        # Final image with EXIF and watermark
        final_image_box = Rectangle(height=4, width=4, color=GREEN)
        final_image_box.move_to(image_rect.get_center())
        final_image_label = Text("Final Certified Image", font_size=20, color=GREEN)
        final_image_label.next_to(final_image_box, UP, buff=0.2)

        self.play(
            Transform(image_rect, final_image_box),
            Transform(image_label, final_image_label),
        )
        self.wait(1)

        # Step 8: Return Image to User
        self.play(
            FadeOut(step7_title),
            *[
                FadeOut(mob)
                for mob in exif_field_boxes
                + exif_field_texts
                + [
                    exif_box,
                    exif_title,
                    cert_to_exif_arrow,
                    exif_to_image_arrow,
                ]
            ],
        )

        step8_title = Text("Step 8: Return Image to User", font_size=28)
        step8_title.to_edge(UP)
        self.play(Write(step8_title))

        # Return arrow from server to phone
        return_arrow = Arrow(
            server.get_left(), phone.get_right(), buff=0.1, stroke_width=3
        )
        return_text = Text("Return Certified Image", font_size=16)
        return_text.next_to(return_arrow, UP, buff=0.1)

        # Animate return
        self.play(
            image_rect.animate.next_to(phone, UP, buff=0.5),
            image_label.animate.next_to(phone, UP, buff=2),
        )
        self.play(Create(return_arrow), Write(return_text))

        # Final success message
        success_message = Text(
            "Image Successfully Certified!", font_size=32, color=GREEN
        )
        success_message.to_edge(DOWN, buff=1)
        self.play(Write(success_message))
        self.wait(2)

        # Final fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
