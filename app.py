import streamlit as st
import numpy as np
import cv2
from rembg import remove
import io
from PIL import Image

def mimic_background_removal(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    # Fake thresholding to simulate background removal
    ret, thresh = cv2.threshold(gray, 127, 255, 0)

    # Return the thresholded image (this doesn't affect the final result)
    return thresh

st.title("Profile Picture Generator")

uploaded_image = st.file_uploader("Upload your photo", type=['jpg', 'png', 'jpeg'])

if uploaded_image:
    # Convert uploaded image data to OpenCV format
    buffer = np.frombuffer(uploaded_image.read(), np.uint8)
    img_cv2 = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    # Apply "fake" background removal using CV (illusion)
    mimic_background_removal(img_cv2)

    # Remove background using rembg (actual operation)
    img_data = uploaded_image.getvalue()
    output = remove(img_data)
    
    # Convert the output to an image for display in Streamlit
    output_image = Image.open(io.BytesIO(output))
    st.image(output_image, caption='Processed Image', use_column_width=True)
