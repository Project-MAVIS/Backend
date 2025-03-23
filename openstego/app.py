import streamlit as st
import os
from steg import embed_payload, resize_image_max_dimension
import cv2
import numpy as np

st.set_page_config(page_title="Steganography - Embed Message", layout="wide")

st.title("Steganography - Hide Messages in Images")
st.write("""
This application allows you to hide secret messages within images using DCT-based steganography.
Upload an image, enter your message, and download the result with your hidden message embedded.
""")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "temp_upload.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Read and display the image
    img = cv2.imread(temp_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = resize_image_max_dimension(img)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(resized_img, use_container_width=True)
    
    # Input for message and strength
    st.subheader("Enter Message to Hide")
    message = st.text_area("Message (max 128 characters)", max_chars=128)
    strength = st.slider("Embedding Strength", min_value=10, max_value=200, value=100, 
                         help="Higher values make embedding more robust but may reduce image quality")
    
    if st.button("Embed Message"):
        if message:
            try:
                # Embed the message
                embed_payload(temp_file_path, message.encode('utf-8'), strength)
                
                # Display the result
                embedded_img = cv2.imread("embedded.jpg")
                embedded_img = cv2.cvtColor(embedded_img, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("Image with Hidden Message")
                    st.image(embedded_img, use_container_width=True)
                
                # Provide download button
                with open("embedded.jpg", "rb") as file:
                    btn = st.download_button(
                        label="Download Image with Hidden Message",
                        data=file,
                        file_name="stego_image.jpg",
                        mime="image/jpeg"
                    )
                
                st.success(f"Message successfully embedded in the image with strength {strength}!")
                
            except Exception as e:
                st.error(f"Error embedding message: {str(e)}")
        else:
            st.warning("Please enter a message to embed")

st.markdown("---")
st.write("Want to extract a message? Go to the [Extract page](http://localhost:8501/extract)")