import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Load YOLOv8 model
model = YOLO("best.pt")

# Set page layout and background
st.set_page_config(page_title="Waterborne Bird Detection System", layout="wide")
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #0575E6, #00F260);
        color: white;
    }
    .main {
        background-color: transparent;
    }
    h1 {
        text-align: center;
        color: white;
        font-size: 3em;
        margin-bottom: 0;
    }
    .subheader {
        text-align: center;
        font-size: 1.2em;
        color: #e0e0e0;
        margin-bottom: 30px;
    }
    .upload-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: black;
        margin: auto;
        width: 50%;
    }
    .tag {
        background-color: #A020F0;
        color: white;
        padding: 4px 8px;
        font-weight: bold;
        border-radius: 8px;
        font-size: 18px;
        display: inline-block;
        margin-bottom: 10px;
    }
    .species-card {
        background-color: #f0f8ff;
        color: #013A63;
        padding: 20px;
        border-radius: 15px;
        font-size: 16px;
        width: 70%;
        margin: 20px auto;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Waterborne Bird Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Upload an image of a bird, and the system will detect and name the bird species.</p>", unsafe_allow_html=True)

# Upload Section
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])
detect_button = st.button("Detect Bird")
st.markdown("</div>", unsafe_allow_html=True)

# Clear button
if st.button("Clear"):
    st.experimental_rerun()

# Detect logic
if uploaded_file and detect_button:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if "image" in uploaded_file.type:
        image = Image.open(tmp_path).convert("RGB")
        image_np = np.array(image)
        results = model.predict(source=image_np, conf=0.3)
        boxes = results[0].boxes

        if boxes and len(boxes.cls) > 0:
            names = results[0].names
            cls_id = int(boxes.cls[0].item())
            label = names[cls_id]
            conf = float(boxes.conf[0].item()) * 100

            # Draw bounding box and label
            draw = ImageDraw.Draw(image)
            xyxy = boxes.xyxy[0].tolist()
            draw.rectangle(xyxy, outline="magenta", width=3)

            # Label text background
            font = ImageFont.load_default()
            draw.rectangle([xyxy[0], xyxy[1] - 20, xyxy[0] + 160, xyxy[1]], fill="magenta")
            draw.text((xyxy[0] + 5, xyxy[1] - 18), label, fill="white", font=font)

            st.image(image, caption="", use_container_width=True)

            # Stylish tag and species info card
            st.markdown(f"<div class='tag'>{label}</div>", unsafe_allow_html=True)

            st.markdown(f"""
                <div class='species-card'>
                    <h2>{label}</h2>
                    <i><b>Scientific name:</b> Unknown (add if available)</i>
                    <p>The <strong>{label}</strong> is a water bird species. Description is unavailable here, but you can enrich this card with actual bird info later.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No birds detected.")
    else:
        st.warning("Please upload an image. Video handling is not yet styled.")
