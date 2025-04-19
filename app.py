import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from collections import Counter
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO("best.pt")  # Update path if in a subfolder

# Optional: custom class name list (if needed)
# with open("class_names.txt") as f:
#     class_names = [line.strip() for line in f]

st.set_page_config(page_title="Water Bird Detection System", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #0A81D1;'>🦆 Water Bird Detection System</h1>
    <p style='text-align: center;'>Upload an image or a video to identify water birds using AI.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

detect_button = st.button("Detect")
clear_button = st.button("Clear")

if clear_button:
    st.experimental_rerun()

if uploaded_file and detect_button:
    file_type = uploaded_file.type

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if "image" in file_type:
        image = Image.open(tmp_path).convert("RGB").resize((640, 640))
        results = model.predict(source=np.array(image), conf=0.1)
        boxes = results[0].boxes

        # Debug output
        st.write("🔍 Detection Boxes:", boxes)

        try:
            names = results[0].names
        except:
            names = model.names  # fallback

        if boxes and boxes.cls.numel() > 0:
            class_ids = boxes.cls.tolist()
            labels = [names[int(cls_id)] for cls_id in class_ids]
            label_counts = Counter(labels)
            summary = ', '.join(f"{count}x {label}" for label, count in label_counts.items())

            st.image(image, caption="Detected Image", use_column_width=True)
            st.success(f"✅ Total birds detected: {len(class_ids)}")
            st.info(f"Birds identified: {summary}")
        else:
            st.warning("⚠️ No birds detected.")

    elif "video" in file_type:
        cap = cv2.VideoCapture(tmp_path)
        frame_count = 0
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 15 == 0:  # Process every 15th frame
                resized_frame = cv2.resize(frame, (640, 640))
                results = model.predict(source=resized_frame, conf=0.1)
                boxes = results[0].boxes

                try:
                    names = results[0].names
                except:
                    names = model.names

                if boxes and boxes.cls.numel() > 0:
                    class_ids = boxes.cls.tolist()
                    labels = [names[int(cls_id)] for cls_id in class_ids]
                    detections.extend(labels)

            frame_count += 1
        cap.release()

        if detections:
            label_counts = Counter(detections)
            summary = ', '.join(f"{count}x {label}" for label, count in label_counts.items())
            st.success(f"✅ Total birds detected: {len(detections)}")
            st.info(f"Birds identified: {summary}")
        else:
            st.warning("⚠️ No birds detected.")
