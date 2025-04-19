import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO("best.pt")  # Make sure this is the correct path

st.set_page_config(page_title="Waterborne Bird Detection System", layout="wide")

# Page header
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(to right, #0A81D1, #00B4DB); border-radius: 20px; color: white;'>
        <h1 style='margin-bottom: 0;'>🦆 Waterborne Bird Detection System</h1>
        <p style='font-size: 18px;'>Upload an image of a bird, and the system will detect and name the bird species.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image or a video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
detect_button = st.button("Detect Bird")

if detect_button and uploaded_file is not None:
    file_type = uploaded_file.type

    # Save file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if "image" in file_type:
        image = cv2.imread(tfile.name)
        results = model(image)

        # Check for detection
        if results[0].boxes and len(results[0].boxes.cls) > 0:
            boxes = results[0].boxes
            top_idx = boxes.conf.argmax()
            cls_id = int(boxes.cls[top_idx])
            confidence = float(boxes.conf[top_idx])
            xyxy = boxes.xyxy[top_idx].cpu().numpy().astype(int)

            # ✅ Get class name from model directly
            class_name = model.names[cls_id]

            # Draw bounding box and label
            label = f"{class_name} ({confidence:.2f})"
            color = (255, 0, 255)
            cv2.rectangle(image, tuple(xyxy[:2]), tuple(xyxy[2:]), color, 3)
            cv2.putText(image, class_name, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Display result
            st.image(image, caption="Detected Bird", channels="BGR", use_container_width=True)

            st.markdown(f"""
                <div style='margin-top: 20px; padding: 20px; background-color: #f0f8ff; border-radius: 10px;'>
                    <h3 style='color: #0A81D1;'>{class_name}</h3>
                    <p style='font-style: italic;'>Confidence: {confidence:.2f}</p>
                    <p><b>Total birds detected:</b> 1</p>
                    <p><b>Birds identified:</b> 1x {class_name}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No birds detected. Try another image.")

    elif "video" in file_type:
        st.warning("Video detection is not implemented in this version.")
    else:
        st.error("Unsupported file type.")

