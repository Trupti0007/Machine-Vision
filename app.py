import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Cricket Bat AI Detector")

st.title("🏏 Cricket Bat Detection System")

model = YOLO("yolov8n.pt")

menu = st.sidebar.selectbox(
    "Select Mode",
    ["Image Detection","Video Detection"]
)

# ---------- IMAGE DETECTION ----------
if menu == "Image Detection":

    st.header("Upload Bat Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        results = model(image)

        boxes = results[0].boxes
        count = len(boxes)

        st.success(f"Bats Detected: {count}")

        brand = st.selectbox(
            "Select Bat Brand",
            ["SS","SG","MRF","Other"]
        )

        if st.button("Confirm Brand"):
            st.success(f"{count} bats recorded as {brand}")

# ---------- VIDEO DETECTION ----------
if menu == "Video Detection":

    st.header("Upload Bat Video")

    video = st.file_uploader("Upload Video", type=["mp4","mov","avi"])

    if video:

        st.video(video)

        if st.button("Analyze Video"):

            st.info("Processing video...")

            results = model.predict(source=video, save=False)

            frames = len(results)

            st.success(f"Frames analyzed: {frames}")
