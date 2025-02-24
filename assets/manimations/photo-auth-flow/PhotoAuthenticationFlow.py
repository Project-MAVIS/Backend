from manim import *

class PhotoAuthenticationFlow(MovingCameraScene):
    def construct(self):
        # Title
        title = Text("Photo Authentication Flow", font_size=36, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # MOBILE CLIENT BOX
        mobile_client_box = Rectangle(height=4, width=6, color=WHITE)
        mobile_client_box.to_edge(LEFT, buff=1)
        mobile_label = Text("Mobile Client", font_size=24)
        mobile_label.next_to(mobile_client_box, DOWN, buff=0.3)

        self.play(Create(mobile_client_box), Write(mobile_label))

        # STEP 1: User Captures Image
        image = Square(side_length=1.5, color=BLUE)
        image_text = Text("Image", font_size=20).move_to(image.get_center())
        image_group = VGroup(image, image_text)
        # image_group.move_to(mobile_client_box.get_left() + RIGHT * 1.5)

        capture_text = Text("User Captures Image", font_size=18)
        capture_text.next_to(image_group, UP, buff=0.3)

        self.play(Create(image), Write(image_text), Write(capture_text))

        # STEP 2: User Calculates Image Hash
        hash_text = Text("User Calculates\nImage Hash", font_size=18)
        hash_text.next_to(image_group, UP, buff=1)

        hash_arrow = Arrow(start=image.get_top(), end=hash_text.get_bottom(), buff=0.1)

        self.play(Create(hash_arrow), Write(hash_text))

        # STEP 3: User Signs Image Hash
        signed_hash_text = Text("User Signs\nImage Hash", font_size=18)
        signed_hash_text.next_to(hash_text, RIGHT, buff=1.5)

        sign_arrow = Arrow(start=hash_text.get_right(), end=signed_hash_text.get_left(), buff=0.1)

        self.play(Create(sign_arrow), Write(signed_hash_text))

        # STEP 4: Send Image & Signed Hash to Server
        server_input_box = Rectangle(height=1, width=2.5, color=WHITE)
        server_input_box_text = Text("Upload-Image/", font_size=20).move_to(server_input_box)
        server_input_group = VGroup(server_input_box, server_input_box_text)
        server_input_group.move_to(mobile_client_box.get_right() + RIGHT * 5)

        send_arrow = Arrow(start=image.get_right(), end=server_input_group.get_left(), buff=0.1)
        signed_hash_label = Text("Signed Hash", font_size=16)
        signed_hash_label.next_to(send_arrow, UP, buff=0.1)

        self.play(self.camera.frame.animate.move_to(server_input_group).scale(1.2))
        self.play(Create(server_input_box), Write(server_input_box_text))
        self.play(Create(send_arrow), Write(signed_hash_label))

        # Move Camera to Server

        # # SERVER BOX
        # server_box = Rectangle(height=4, width=8, color=WHITE)
        # server_box.to_edge(RIGHT * 5, buff=1)
        # server_label = Text("Server", font_size=24)
        # server_label.next_to(server_box, DOWN, buff=0.3)

        # self.play(Create(server_box), Write(server_label))

        # STEP 5: Server Calculates Independent Hash
        independent_hash_text = Text("Independent hash calculated\nfrom image received", font_size=18)
        independent_hash_text.next_to(server_input_group, UP, buff=1)
        independent_hash_text.shift(RIGHT * 1.5)

        independent_hash_arrow = Arrow(
            start=server_input_group.get_top(),
            end=independent_hash_text.get_bottom(),
            buff=0.1
        )

        self.play(Create(independent_hash_arrow), Write(independent_hash_text))

        # STEP 6: Decrypt Signed Hash
        decrypt_hash_text = Text("Decrypt the signed hash\nfrom the photographer", font_size=18)
        decrypt_hash_text.next_to(server_input_group, DOWN, buff=1)
        decrypt_hash_text.shift(RIGHT * 1.5)


        decrypt_arrow = Arrow(
            start=server_input_group.get_bottom(),
            end=decrypt_hash_text.get_top(),
            buff=0.1
        )

        self.play(Create(decrypt_arrow), Write(decrypt_hash_text))

        # STEP 7: Comparison Decision
        decision_diamond = Rectangle(color=WHITE)
        decision_text = Text("Is same?", font_size=20).move_to(decision_diamond)

        decision_group = VGroup(decision_diamond, decision_text)
        decision_group.next_to(server_input_group, RIGHT, buff=5.5)

        compare_arrow_1 = Arrow(start=independent_hash_text.get_right(), end=decision_group.get_left(), buff=0.1)
        compare_arrow_2 = Arrow(start=decrypt_hash_text.get_right(), end=decision_group.get_left(), buff=0.1)

        self.play(Create(decision_diamond), Write(decision_text))
        self.play(Create(compare_arrow_1), Create(compare_arrow_2))

        # STEP 8: If Not Same, Return Error
        error_text = Text("Return Error to Photographer", font_size=18, color=RED)
        error_text.next_to(decision_group, DOWN, buff=0.5)  # Position the error text below
        
        self.play(self.camera.frame.animate.move_to(decision_group).scale(1))

        error_arrow = Arrow(
            start=decision_group.get_bottom(),
            end=error_text.get_top(),
            buff=0.1
        )
        
        # Add "NO" label for the error path
        no_text = Text("NO", font_size=18, color=RED)
        no_text.next_to(error_arrow, LEFT, buff=0.2)  # Position "NO" near the error arrow

        self.play(Create(error_arrow), Write(error_text), Write(no_text))  # Animate NO with the error arrow

        # STEP 9: If Same, Save Original Image
        # success_text = Text("YES", font_size=18, color=GREEN)
        # success_text.next_to(decision_group, RIGHT, buff=0.3)

        # save_image_text = Text("Save original image", font_size=18)
        # save_image_text.next_to(success_text, RIGHT, buff=0.5)

        # success_arrow = Arrow(
        #     start=decision_group.get_right(),
        #     end=save_image_text.get_left(),
        #     buff=0.1
        # )

        # self.play(Write(success_text), Create(success_arrow), Write(save_image_text))
        save_image_text = Text("Save original image", font_size=18)
        save_image_text.next_to(decision_group, RIGHT, buff=1)  # Space out for clarity

        success_arrow = Arrow(
            start=decision_group.get_right(),
            end=save_image_text.get_left(),
            buff=0.1
        )

        # Add "YES" label on top of the success arrow
        yes_text = Text("YES", font_size=18, color=GREEN)
        yes_text.next_to(success_arrow, UP, buff=0.2)  # Position "YES" above the arrow

        self.play(Create(success_arrow), Write(save_image_text), Write(yes_text)) 

        # STEP 10: Make Certificate
        certificate_text = Text("Make certificate", font_size=18)
        certificate_text.next_to(save_image_text, RIGHT, buff=0.5)

        certificate_arrow = Arrow(
            start=save_image_text.get_right(),
            end=certificate_text.get_left(),
            buff=0.1
        )

        self.play(Create(certificate_arrow), Write(certificate_text))
        self.play(self.camera.frame.animate.move_to(certificate_text).scale(1))

        # STEP 11: Sign Certificate
        sign_certificate_text = Text("Sign certificate with\nserver private key", font_size=18)
        sign_certificate_text.next_to(certificate_text, RIGHT, buff=0.5)

        sign_certificate_arrow = Arrow(
            start=certificate_text.get_right(),
            end=sign_certificate_text.get_left(),
            buff=0.1
        )

        self.play(Create(sign_certificate_arrow), Write(sign_certificate_text))

        # STEP 12: Embed Certificate in Image
        embed_certificate_text = Text("Embed/add certificate\nto copy of image", font_size=18)
        embed_certificate_text.next_to(sign_certificate_text, RIGHT, buff=0.5)

        embed_certificate_arrow = Arrow(
            start=sign_certificate_text.get_right(),
            end=embed_certificate_text.get_left(),
            buff=0.1
        )

        self.play(Create(embed_certificate_arrow), Write(embed_certificate_text))
        self.play(self.camera.frame.animate.move_to(embed_certificate_text).scale(1))

        # STEP 13: Provide Image Download Links
        download_text = Text("Provide response with\nimage download links", font_size=18)
        download_text.next_to(embed_certificate_text, RIGHT, buff=0.5)

        download_arrow = Arrow(
            start=embed_certificate_text.get_right(),
            end=download_text.get_left(),
            buff=0.1
        )

        self.play(Create(download_arrow), Write(download_text))

        # Final Wait
        self.wait(2)

        # Fixed FadeOut Issue
        all_objects = [
            title, mobile_client_box, mobile_label, image_group, capture_text,
            hash_text, hash_arrow, signed_hash_text, sign_arrow, server_input_group, send_arrow,
            # server_box, 
            # server_label, 
            independent_hash_text, independent_hash_arrow,
            decrypt_hash_text, decrypt_arrow, decision_group, compare_arrow_1, compare_arrow_2,
            error_text, error_arrow, 
            # success_text, 
            yes_text,
            save_image_text, success_arrow,
            certificate_text, certificate_arrow, sign_certificate_text, sign_certificate_arrow,
            embed_certificate_text, embed_certificate_arrow, download_text, download_arrow
        ]
        self.play(*[FadeOut(obj) for obj in all_objects])
