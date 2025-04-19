import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2
from collections import Counter

# Load model once
model = YOLO("best.pt")

st.set_page_config(page_title="Water Bird Detection", layout="wide")

# Title
st.markdown("""
    <h1 style='text-align: center; color: #0A81D1;'>🦆 Water Bird Detection System</h1>
    <p style='text-align: center;'>Upload an image or video to detect water birds.</p>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

# Buttons
detect_button = st.button("Detect")
clear_button = st.button("Clear")

if clear_button:
    st.experimental_rerun()

if uploaded_file and detect_button:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    file_type = uploaded_file.type

    if "image" in file_type:
        # Show uploaded image
        image = Image.open(temp_path).convert("RGB")
        st.subheader("📤 Uploaded Image")
        st.image(image, use_container_width=True)

        # Predict
        results = model.predict(np.array(image), conf=0.25)
        boxes = results[0].boxes

        if boxes and len(boxes.cls) > 0:
            # Plot and display
            result_img = results[0].plot()
            class_ids = boxes.cls.tolist()
            names = model.names
            labels = [names[int(id)] for id in class_ids]
            label_counts = Counter(labels)
            summary = ', '.join(f"{v}x {k}" for k, v in label_counts.items())

            st.subheader("✅ Detection Result")
            st.image(result_img, caption="Detected Image", use_container_width=True)
            st.success(f"Total birds detected: {len(class_ids)}")
            st.info(f"Birds identified: {summary}")
        else:
            st.warning("❌ No birds detected in this image.")

    elif "video" in file_type:
        st.info("🔄 Processing video... Please wait.")

        cap = cv2.VideoCapture(temp_path)
        frame_count = 0
        detections = []
        preview = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 15 == 0:
                results = model.predict(frame, conf=0.25)
                boxes = results[0].boxes
                if boxes and len(boxes.cls) > 0:
                    if preview is None:
                        preview = results[0].plot()
                    class_ids = boxes.cls.tolist()
                    names = model.names
                    labels = [names[int(id)] for id in class_ids]
                    detections.extend(labels)
            frame_count += 1
        cap.release()

        if detections:
            label_counts = Counter(detections)
            summary = ', '.join(f"{v}x {k}" for k, v in label_counts.items())
            st.subheader("🎥 Detection Summary")
            if preview is not None:
                st.image(preview, caption="Sample Frame", use_container_width=True)
            st.success(f"Total birds detected: {len(detections)}")
            st.info(f"Birds identified: {summary}")
        else:
            st.warning("❌ No birds detected in this video.")

