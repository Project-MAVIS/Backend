import streamlit as st
import requests
from PIL import Image
import io


def main():
    st.title("Image Verification App")
    st.write("Upload an image to verify it with the API")

    # File uploader for image
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Button to send the image to API
        if st.button("Verify Image"):
            with st.spinner("Sending image to verification service..."):
                # Make API request with the uploaded file
                try:
                    # Create file object for the request
                    files = {
                        "image": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            f'image/{uploaded_file.type.split("/")[1]}',
                        )
                    }

                    # Send POST request to API
                    response = requests.post(
                        "http://4.240.97.237:8000/api/verify/", files=files
                    )

                    # Display results
                    if response.status_code == 200:
                        st.success("Image verified successfully!")
                        with st.expander("Details"):
                            st.json(response.json())
                    else:
                        st.error(f"Error: Image does not have provenance information")
                        with st.expander("Details"):
                            st.write(f"Response: {response.text}")
                        # st.write(f"Response: {response.text}")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
