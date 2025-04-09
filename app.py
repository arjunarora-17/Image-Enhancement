import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
import cv2
import tempfile
from ultralytics import YOLO

# Add model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'LIME_MODEL'))
sys.path.append(os.path.join(BASE_DIR, 'DARK_IR_MODEL'))
sys.path.append(os.path.join(BASE_DIR, 'ZERO_DCE'))
sys.path.append(os.path.join(BASE_DIR, 'BIMEF_MODEL'))
sys.path.append(BASE_DIR)

from lime_wrapper import enhance_lime_image
from dark_ir_wrapper import enhance_darkir_image
from zero_dce_wrapper import enhance_zero_dce_image
from bimef_wrapper import enhance_bimef_image

# Load YOLO model once
@st.cache_resource
def load_model():
    return YOLO("yolov5s.pt")



# Default Streamlit theme and layout
st.set_page_config(page_title="Low-Light Image Enhancement", layout="wide")

# Custom CSS for layout and styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
        }
        button:hover {
            background-color: #FAF9F6 !important;
            transition: all 0.2s ease-in-out;
            color: black !important;
        }
        .bottom-right {
            position: fixed;
            bottom: 20px;
            right: 30px;
            z-index: 9999;
        }
    </style>
""", unsafe_allow_html=True)

# Bottom-right Reset button using Streamlit-native method
with st.container():
    if st.button("🔄 Reset", key="reset_button"):
        st.session_state.clear()
        st.rerun()


if "reset" in st.session_state:
    st.session_state.clear()
    st.rerun()

# Title (centered and styled)
st.markdown("""
    <h1 style='text-align: center; font-size: 48px; color: white;'>
        Image Restoration and Object Recognition in Low-Light Environments
        <hr style='margin-top: 10px; margin-bottom: -10px; border: 1px solid #ccc;' />
    </h1>
""", unsafe_allow_html=True)

# First row layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("<div style='font-size:20px; font-weight:bold; color:white;'>Enhancement Method</div>", unsafe_allow_html=True)
    method = st.selectbox("", ["LIME", "DarkIR", "Zero-DCE", "BIMEF"])

with col2:
    st.markdown("<div style='font-size:20px; font-weight:bold; color:white;'>Upload Image</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(" ", type=["png", "jpg", "jpeg"])

with col3:
    st.markdown("<div style='font-size:20px; font-weight:bold; color:white;'>Choose Example</div>", unsafe_allow_html=True)
    example_dir = os.path.join(BASE_DIR, "example_images")
    all_examples = [f for f in os.listdir(example_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]
    selected_example = st.selectbox(" ", ["None"] + all_examples)

# Full-width action button
col_full = st.columns([1])[0]
with col_full:
    if st.button("Enhance Image and Detect Objects", use_container_width=True):
        if uploaded_file:
            input_image = Image.open(uploaded_file).convert('RGB')
        elif selected_example != "None":
            input_image = Image.open(os.path.join(example_dir, selected_example)).convert('RGB')
        else:
            st.warning("Please upload or select an image.")
            st.stop()
        MAX_SIZE = (640, 640)
        input_image.thumbnail(MAX_SIZE, Image.LANCZOS)

        with st.spinner(f"Enhancing image using {method}..."):
            if method == "LIME":
                image_bgr = np.array(input_image)[:, :, ::-1]
                enhanced_bgr = enhance_lime_image(image_bgr)
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                output_image = Image.fromarray(enhanced_rgb)
            elif method == "DarkIR":
                output_image = enhance_darkir_image(input_image)
            elif method == "Zero-DCE":
                output_image = enhance_zero_dce_image(input_image)
            elif method == "BIMEF":
                output_image = enhance_bimef_image(input_image)

        st.markdown("<h3 style='text-align: center;'>Input vs Enhanced Image</h3>", unsafe_allow_html=True)

        _, col_a, col_b,_ = st.columns(4)
        with col_a:
            st.image(input_image, caption="Original Image", use_container_width=True)
        with col_b:
            st.image(output_image, caption=f"{method} Enhanced Image", use_container_width=True)

        # Object Detection
        with st.spinner("Running YOLOv5 object detection..."):
            model = load_model()

            temp_orig = os.path.join(tempfile.gettempdir(), "temp_orig.png")
            input_image.save(temp_orig)
            results_orig = model(temp_orig)
            annotated_orig = results_orig[0].plot()
            annotated_orig_rgb = cv2.cvtColor(annotated_orig, cv2.COLOR_BGR2RGB)
            annotated_orig_pil = Image.fromarray(annotated_orig_rgb)

            temp_enh = os.path.join(tempfile.gettempdir(), "temp_enh.png")
            output_image.save(temp_enh)
            results_enh = model(temp_enh)
            annotated_enh = results_enh[0].plot()
            annotated_enh_rgb = cv2.cvtColor(annotated_enh, cv2.COLOR_BGR2RGB)
            annotated_enh_pil = Image.fromarray(annotated_enh_rgb)

        st.markdown("<h3 style='text-align: center;'>Object Detection Results</h3>", unsafe_allow_html=True)
        _, col_c, col_d,_ = st.columns(4)
        with col_c:
            st.image(annotated_orig_pil, caption="YOLO on Original Image", use_container_width=True)
        with col_d:
            st.image(annotated_enh_pil, caption=f"YOLO on {method} Enhanced Image", use_container_width=True)
        del input_image, output_image, annotated_orig_pil, annotated_enh_pil
