import streamlit as st
import numpy as np
from rembg import remove
import io
from PIL import Image

st.title("Background Removal with rembg")

uploaded_image = st.file_uploader("Upload your photo", type=['jpg', 'png', 'jpeg'])

if uploaded_image:
    img_data = uploaded_image.read()
    
    # Remove background using rembg
    output = remove(img_data)
    
    # Convert the output to an image for display in Streamlit
    output_image = Image.open(io.BytesIO(output))

    st.image(output_image, caption='Processed Image', use_column_width=True)
