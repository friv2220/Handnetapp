import streamlit as st
import numpy as np
import cv2
from PIL import Image
from model import analyze_sign

st.title("HandNet")

st.write("Turn on camera and show your sign")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    img = Image.open(camera_image)
    st.image(img, caption="Captured Image")

    img_cv = np.array(img)

    result = analyze_sign(img_cv)

    st.success(result)
