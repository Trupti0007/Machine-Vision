import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile
import cv2

st.set_page_config(page_title="Cricket Bat AI System", layout="wide")

st.title("🏏 Smart Cricket Bat Detection & Inventory")

model = YOLO("yolov8n.pt")

# Load brand memory
try:
    data = pd.read_csv("brands.csv")
except:
    data = pd.DataFrame(columns=["image","brand"])

menu = st.sidebar.selectbox(
    "Mode",
    ["Image Detection","Video Detection","Inventory"]
)

# ---------------- IMAGE DETECTION ----------------

if menu == "Image Detection":

    st.header("Upload Bat Image")

    file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if file:

        image = Image.open(file)

        st.image(image, use_column_width=True)

        if st.button("Detect Bats"):

            results = model(image)

            boxes = results[0].boxes
            count = len(boxes)

            st.success(f"Bats detected: {count}")

            brand = st.selectbox(
                "Select Brand",
                ["SS","SG","MRF","Other"]
            )

            if st.button("Save Record"):

                new = pd.DataFrame({
                    "image":[file.name],
                    "brand":[brand]
                })

                data2 = pd.concat([data,new])

                data2.to_csv("brands.csv", index=False)

                st.success("Saved to inventory memory")

# ---------------- VIDEO DETECTION ----------------

if menu == "Video Detection":

    st.header("Upload Bat Video")

    video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if video:

        st.video(video)

        if st.button("Analyze Video"):

            st.info("Processing...")

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video.read())

            cap = cv2.VideoCapture(tfile.name)

            frames = 0
            bats = 0

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frames += 1

                results = model(frame)

                bats += len(results[0].boxes)

            cap.release()

            st.success(f"Frames analyzed: {frames}")
            st.success(f"Total bats detected: {bats}")

# ---------------- INVENTORY ----------------

if menu == "Inventory":

    st.header("Bat Brand Records")

    try:
        inventory = pd.read_csv("brands.csv")
        st.dataframe(inventory)
    except:
        st.warning("No inventory yet")
