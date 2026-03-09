import streamlit as st
import cv2
import tempfile
from detect import detect_products
from barcode import scan_barcode

st.set_page_config(page_title="Cricket Inventory AI")

st.title("Cricket Warehouse Inventory AI")

uploaded_video = st.file_uploader("Upload Warehouse Video", type=["mp4","avi","mov"])

if uploaded_video:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_file.name)

    frame_window = st.empty()

    inventory = {}

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

        frame_window.image(frame, channels="BGR")

    cap.release()

    st.subheader("Inventory Count")

    st.write(inventory)
