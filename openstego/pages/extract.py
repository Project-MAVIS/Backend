import streamlit as st
import os
from steg import extract_payload
import cv2

st.set_page_config(page_title="Steganography - Extract Message", layout="wide")

st.title("Steganography - Extract Hidden Messages")
st.write("""
Upload an image that contains a hidden message, specify the embedding strength that was used,
and the application will extract and display the hidden message.
""")

uploaded_file = st.file_uploader("Choose an image file with a hidden message...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "temp_extract.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Read and display the image
    img = cv2.imread(temp_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.subheader("Uploaded Image")
    st.image(img, use_container_width=True)
    
    # Input for strength
    strength = st.slider("Embedding Strength", min_value=10, max_value=200, value=100, 
                         help="Must match the strength used during embedding")
    
    if st.button("Extract Message"):
        try:
            # Extract the message
            result = extract_payload(temp_file_path, strength)
            
            if isinstance(result, bytes):
                message = result.decode('utf-8')
                st.success("Message extracted successfully!")
                
                st.subheader("Extracted Message:")
                st.code(message, language=None)
            else:
                st.error(result)  # Display the error message
                
        except Exception as e:
            st.error(f"Error extracting message: {str(e)}")

st.markdown("---")
st.write("Want to embed a message? Go to the [Embed page](http://localhost:8501/)")