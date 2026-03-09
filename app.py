import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import numpy as np

st.title("🏏 Cricket Bat Detection System")

st.write("Upload an image of bats and the AI will count them.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Detecting bats...")

    model = YOLO("yolov8n.pt")

    results = model(image)

    boxes = results[0].boxes

    count = len(boxes)

    st.success(f"Total Objects Detected: {count}")

    brands = ["SS","SG","MRF","Other"]

    brand = st.selectbox("Select Bat Brand", brands)

    if st.button("Save Inventory"):

        data = {
            "Brand":[brand],
            "Quantity":[count]
        }

        df = pd.DataFrame(data)

        df.to_csv("inventory.csv",mode="a",index=False,header=False)

        st.success("Inventory saved successfully")
