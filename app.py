import streamlit as st
import numpy as np
import cv2
from rembg import remove
import io
from PIL import Image

def mimic_background_removal(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel Operators for edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobel_x, sobel_y)

    # Adaptive thresholding
    _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to close small holes and remove small white spots
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Use the mask to extract the subject (illusion of background removal)
    foreground = cv2.bitwise_and(img, img, mask=mask)

    # Additional operations to add depth to the illusion
    edges = cv2.Canny(foreground, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(foreground, contours, -1, (0, 255, 0), 3)

    return foreground

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
