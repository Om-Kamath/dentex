# Contrast Enhancement:

# Color Space Conversion: The image is converted from BGR to Lab color space using cv2.cvtColor.
# CLAHE (Contrast Limited Adaptive Histogram Equalization): This method improves the contrast of the image by redistributing lightness values. It's applied specifically to the L channel of the Lab color space.
# Grayscale Conversion:

# The image is converted to grayscale using cv2.cvtColor.
# Enhancement:

# CLAHE: Again, CLAHE is applied but this time to the grayscale image for further enhancement.
# Noise Reduction:

# Bilateral Filter: This filter is used to reduce noise while preserving edges. It smoothens the image by replacing each pixel's value with a weighted average from its neighborhood.
# Thresholding:

# Binary Inverse Thresholding with OTSU: Otsu's method is used to automatically determine the threshold value. Pixels below this value are set to white, and those above are set to black.
# Morphological Operations:

# Closing: This operation (a dilation followed by an erosion) is used to close small holes in the foreground.
# Edge Detection:

# Canny Edge Detector: It detects edges in the image by looking for regions in the image with rapid intensity changes.
# Contour Detection:

# Finding Contours: Using the cv2.findContours function, contours in the binary image are detected. Only the external contours are retrieved.
# Area-based Filtering: Contours with an area below a certain threshold (100 in this case) are discarded.
# Contour Processing:

# Sorting by Area: The detected contours are sorted based on their area.
# Moment Calculation: The centroid (center of mass) of the contours is calculated using image moments.
# Distance-based Filtering: Contours that are too far from the average position of the largest contours are discarded.
# Visualization:

# Drawing Contours: The remaining contours, which are believed to represent dents, are drawn onto the original image using the cv2.drawContours function.

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# ===================
# UTILITY FUNCTIONS
# ===================

def enhance_contrast(image):
    """Enhance the contrast of the image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged_channels = cv2.merge([cl, a_channel, b_channel])
    return cv2.cvtColor(merged_channels, cv2.COLOR_Lab2BGR)

def compute_dent_score(contours):
    # If no contours are detected, return 0% as the score
    if not contours:
        return 0
    
    avg_dent_perimeter = sum([cv2.arcLength(contour, True) for contour in contours]) / len(contours)
    k = 0.01
    return 100 / (1 + math.exp(-k * avg_dent_perimeter))

# =======================
# DENT DETECTION FUNCTION
# =======================

def advanced_dent_detection(image):
    """Detect dents in the provided image."""
    
    image = enhance_contrast(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    _, binary_thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    edges = cv2.Canny(closing, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 100]
    print("No. of contours",len(contours))
    # Ignore small lone dents
    if len(contours) <= 2:
        average = sum([cv2.contourArea(contour) for contour in contours]) / len(contours) if contours else 0
        print("Average",average)
        image_area = image.shape[0] * image.shape[1]
        print("Image",0.05*image_area)
        if average < 0.05 * image_area:  # Assuming 0.5% as the threshold
            return image, []
    if not contours:  
        return image, []

    # Other contour processing
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    num_significant = max(1, int(0.2 * len(sorted_contours)))
    significant_contours = sorted_contours[:num_significant]
    avg_cx = sum([int(cv2.moments(contour)['m10'] / cv2.moments(contour)['m00']) for contour in significant_contours]) / num_significant
    avg_cy = sum([int(cv2.moments(contour)['m01'] / cv2.moments(contour)['m00']) for contour in significant_contours]) / num_significant
    threshold_distance = max(image.shape[0], image.shape[1]) * 0.325
    filtered_contours = [contour for contour in contours if math.sqrt((avg_cx - int(cv2.moments(contour)['m10'] / cv2.moments(contour)['m00']))**2 + (avg_cy - int(cv2.moments(contour)['m01'] / cv2.moments(contour)['m00']))**2) < threshold_distance]
    
    contour_image = image.copy()
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
    return contour_image, filtered_contours

# ===============
# STREAMLIT APP
# ===============

st.title("Enhanced Dent Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    result, contours = advanced_dent_detection(image_np)
    dent_score = compute_dent_score(contours)
    st.image([image_np, result], caption=["Original Image", "Dents Detected"], use_column_width=True)
    st.subheader(f"Dent Severity Score: {dent_score:.2f}%")
    
    if dent_score < 70:
        evaluation = "Panel is Flawless"
        color = "green"
    elif dent_score < 95:
        evaluation = "Panel Repair"
        color = "orange"
    else:
        evaluation = "Panel Replacement"
        color = "red"
    
    st.subheader(f"Evaluation: {evaluation}")
    st.markdown(f"<div style='background-color: #E6E6E6; padding: 10px; border-radius: 10px; width: 100%;'><div style='background-color: {color}; width: {dent_score}%; height: 20px; border-radius: 8px;'></div></div>", unsafe_allow_html=True)
