import streamlit as st
import cv2
import tempfile
from detect import detect_products
from barcode import scan_barcode

st.title("Cricket Inventory AI System")

uploaded_video = st.file_uploader("Upload Warehouse Video")

if uploaded_video:

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp.name)

    inventory = {}

    frame_area = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        detections = detect_products(frame)

        for item in detections:

            if item not in inventory:
                inventory[item] = 0

            inventory[item] += 1

        barcodes = scan_barcode(frame)

        frame_area.image(frame, channels="BGR")

    cap.release()

    st.subheader("Inventory Count")

    st.write(inventory)
