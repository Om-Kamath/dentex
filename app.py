import streamlit as st
import cv2
import numpy as np
from rembg import remove

def overlay_on_background(foreground, background):
    # If the foreground image has 4 channels (meaning it has an alpha transparency channel)
    if foreground.shape[2] == 4:
        alpha = foreground[:, :, 3] / 255.0
        combined = (foreground[:, :, :3] * alpha[:, :, np.newaxis] + background * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
        return combined
    return foreground

def circular_profile_picture(image):
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.circle(mask, (image.shape[1]//2, image.shape[0]//2), min(image.shape[0], image.shape[1])//2, (255,255,255), -1)
    circular_img = cv2.bitwise_and(image, image, mask=mask)
    return circular_img

st.title("Profile Pic Generator")

uploaded_image = st.file_uploader("Upload your photo", type=['jpg', 'png', 'jpeg'])
selected_bg = st.selectbox("Choose a background", ["Transparent", "Beach", "Mountains", "City"])
make_circular = st.checkbox("Make it circular")

if uploaded_image:
    img_data = uploaded_image.read()
    
    # Remove background using rembg
    output = remove(img_data)
    alpha_channel = cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_UNCHANGED)
    
    if selected_bg != "Transparent":
        if selected_bg == "Beach":
            bg_image_path = "path_to_beach_image.jpg"
        elif selected_bg == "Mountains":
            bg_image_path = "path_to_mountains_image.jpg"
        elif selected_bg == "City":
            bg_image_path = "path_to_city_image.jpg"

        bg_image = cv2.imread(bg_image_path)
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)  # Convert background image to RGB
        bg_image = cv2.resize(bg_image, (alpha_channel.shape[1], alpha_channel.shape[0]))
        
        combined = overlay_on_background(alpha_channel, bg_image)
    else:
        combined = alpha_channel

    if make_circular:
        combined = circular_profile_picture(combined)

    if combined.shape[2] == 4:  # If the image has an alpha channel (i.e., is RGBA)
        st.image(combined, channels="RGBA", use_column_width=True)
    else:
        st.image(combined, channels="RGB", use_column_width=True)
