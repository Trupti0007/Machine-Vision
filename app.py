import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

st.set_page_config(page_title="Bat Detection System")

st.title("AI Bat Detection System")

# Load YOLO model
model = YOLO("model.pt")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:

    # Save uploaded video temporarily
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_video.name)

    frame_placeholder = st.empty()
    count_placeholder = st.empty()

    total_objects = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Run YOLO detection
        results = model(frame)

        boxes = results[0].boxes
        total_objects += len(boxes)

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        # Convert BGR → RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(annotated_frame, channels="RGB")

        count_placeholder.write(f"Detected objects: {len(boxes)}")

    cap.release()

    st.success(f"Total objects detected in video: {total_objects}")
