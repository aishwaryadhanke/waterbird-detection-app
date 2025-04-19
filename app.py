import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from collections import Counter
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO("best.pt")

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

def draw_boxes(image, results):
    annotated_image = image.copy()
    names = results.names
    for box in results[0].boxes:
        cls_id = int(box.cls)
        label = names[cls_id]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(annotated_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return annotated_image

if uploaded_file and detect_button:
    file_type = uploaded_file.type

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if "image" in file_type:
        image = Image.open(tmp_path).convert("RGB")
        image_np = np.array(image)
        results = model.predict(source=image_np, conf=0.25)
        boxes = results[0].boxes
        class_ids = boxes.cls.tolist() if boxes else []

        if class_ids:
            labels = [results[0].names[int(cls_id)] for cls_id in class_ids]
            label_counts = Counter(labels)
            summary = ', '.join(f"{count}x {label}" for label, count in label_counts.items())

            annotated = draw_boxes(image_np, results)
            st.image(annotated, caption="Detected Image", channels="RGB", use_column_width=True)
            st.success(f"✅ Total birds detected: {len(class_ids)}")
            st.info(f"Birds identified: {summary}")
        else:
            st.image(image, caption="Original Image", use_column_width=True)
            st.warning("No birds detected.")

    elif "video" in file_type:
        cap = cv2.VideoCapture(tmp_path)
        frame_count = 0
        detections = []

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 15 == 0:
                results = model.predict(source=frame, conf=0.25)
                boxes = results[0].boxes
                class_ids = boxes.cls.tolist() if boxes else []
                labels = [results[0].names[int(cls_id)] for cls_id in class_ids]
                detections.extend(labels)

                annotated = draw_boxes(frame, results)
                stframe.image(annotated, channels="BGR", use_column_width=True)

            frame_count += 1
        cap.release()

        if detections:
            label_counts = Counter(detections)
            summary = ', '.join(f"{count}x {label}" for label, count in label_counts.items())
            st.success(f"✅ Total birds detected: {len(detections)}")
            st.info(f"Birds identified: {summary}")
        else:
            st.warning("No birds detected.")
