from manim import *


class PhotoAuthenticationFlow(Scene):
    def construct(self):
        # Scene 1: Phone taking photo
        phone = RoundedRectangle(
            height=4, width=2, corner_radius=0.2, stroke_color=WHITE
        )
        camera_lens = Circle(radius=0.2).move_to(phone.get_top() + DOWN * 0.5)
        phone_group = VGroup(phone, camera_lens)

        # Start phone in center
        phone_group.move_to(ORIGIN)

        # Create a "flash" effect - using ShowCreation instead of Flash
        flash_lines = VGroup(
            *[
                Line(
                    camera_lens.get_center(),
                    camera_lens.get_center() + RIGHT * 0.3 + UP * 0.3,
                ).rotate(angle=i * PI / 6, about_point=camera_lens.get_center())
                for i in range(12)
            ]
        )
        flash_lines.set_color(WHITE)

        # Scene 1 Animation
        self.play(Create(phone_group))
        self.play(Create(flash_lines))
        self.play(FadeOut(flash_lines))

        # Move phone further to left before showing photo
        self.play(phone_group.animate.shift(LEFT * 4))  # Moved further left

        # Photo representation - increased buffer space
        photo = Rectangle(height=2, width=2)
        photo.next_to(phone_group, RIGHT, buff=2)  # Increased buffer from 1 to 2
        photo_text = Text("Photo", font_size=24).move_to(photo.get_center())

        self.play(Create(photo), Write(photo_text))
        self.wait()

        # Scene 2: Hash Calculation
        hash_text = Text("Calculating Hash...", font_size=24)
        hash_text.next_to(photo, DOWN, buff=0.5)

        # Create binary flow effect - adjusted position
        binary = VGroup(*[Text("01", font_size=12) for _ in range(20)]).arrange(
            RIGHT, buff=0.1
        )
        binary.next_to(hash_text, DOWN, buff=0.3)

        hash_result = Text("Hash: 2fd4e1c67a2d28...", font_size=20)
        hash_result.next_to(binary, DOWN, buff=0.3)

        # Scene 2 Animation
        self.play(Write(hash_text))
        self.play(Write(binary))
        self.play(Write(hash_result))
        self.wait()

        # Scene 3: Encryption with Private Key
        # Clear previous elements
        self.play(*[FadeOut(mob) for mob in [photo, hash_text, binary, photo_text]])

        # Zoom into phone's secure element
        secure_element = Rectangle(height=1, width=1.5, color=YELLOW)
        secure_element.move_to(phone.get_center())
        secure_text = Text("Secure\nEnclave", font_size=16)
        secure_text.move_to(secure_element.get_center())

        private_key = Text("Private Key", font_size=16, color=RED)
        private_key.next_to(secure_element, DOWN, buff=0.2)

        # Scene 3 Animation
        self.play(
            phone_group.animate.scale(1.5),
            Create(secure_element),
            Write(secure_text),
            Write(private_key),
        )
        self.wait()

        # Encryption visualization
        encrypted_hash = Text("Encrypted Hash", font_size=20, color=GREEN)
        encrypted_hash.next_to(hash_result, RIGHT, buff=1)

        encryption_arrow = Arrow(
            hash_result.get_right(), encrypted_hash.get_left(), buff=0.1
        )

        self.play(Create(encryption_arrow), Write(encrypted_hash))
        self.wait()

        # Scene 4: Sending to server
        server = Rectangle(height=3, width=2)
        server.shift(RIGHT * 4)
        server_text = Text("Server", font_size=24)
        server_text.move_to(server.get_center())

        # Create components of the package
        photo_component = Rectangle(height=2, width=2, color=WHITE)
        photo_text = Text("Photo", font_size=20)
        photo_text.move_to(photo_component.get_center())

        encrypted_hash_component = Rectangle(height=0.5, width=2, color=WHITE)
        encrypted_hash_component.next_to(photo_component, DOWN, buff=0.2)
        encrypted_hash_text = Text("Encrypted Hash", font_size=16)
        encrypted_hash_text.move_to(encrypted_hash_component.get_center())

        public_key_component = Rectangle(height=0.5, width=2, color=WHITE)
        public_key_component.next_to(encrypted_hash_component, DOWN, buff=0.2)
        public_key_text = Text("Public Key", font_size=16)
        public_key_text.move_to(public_key_component.get_center())

        # Group all components
        package_contents = VGroup(
            photo_component,
            photo_text,
            encrypted_hash_component,
            encrypted_hash_text,
            public_key_component,
            public_key_text,
        )
        package_contents.next_to(phone_group, RIGHT, buff=2)

        # Outer package rectangle
        package = Rectangle(
            height=package_contents.height + 0.4,
            width=package_contents.width + 0.4,
            color=BLUE,
        )
        package.move_to(package_contents.get_center())

        # Scene 4 Animation
        self.play(Create(server), Write(server_text))

        # Animate components appearing one by one
        self.play(Create(photo_component), Write(photo_text))
        self.play(Create(encrypted_hash_component), Write(encrypted_hash_text))
        self.play(Create(public_key_component), Write(public_key_text))

        # Create the package border
        self.play(Create(package))

        # Animate package moving to server
        package_group = VGroup(package, package_contents)
        self.play(package_group.animate.next_to(server, LEFT, buff=0.1))
        self.wait()

        # Final fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
