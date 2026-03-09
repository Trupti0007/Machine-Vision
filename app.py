import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import tempfile
import os
import json
import numpy as np
from moviepy.editor import VideoFileClip

# --- Configurations ---
st.set_page_config(page_title="Cricket Bat Detector", layout="wide")
KNOWLEDGE_FILE = "bat_knowledge.csv"
BAT_CLASSES = [34] # YOLO COCO class 34 is 'baseball bat' (used as a proxy for cricket bats)

# --- Knowledge Base / Learning Functions ---
def load_knowledge():
    """Load previously saved bat features and brands."""
    if os.path.exists(KNOWLEDGE_FILE):
        return pd.read_csv(KNOWLEDGE_FILE)
    return pd.DataFrame(columns=["features", "brand"])

def extract_features(img_rgb):
    """Extract a 2D Hue-Saturation color histogram as a feature vector."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Calculate histogram and normalize
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def predict_brand(features, df, threshold=0.7):
    """Find the closest matching brand from the knowledge database."""
    if df.empty:
        return "Unknown"
    
    min_dist = float('inf')
    best_brand = "Unknown"
    
    for _, row in df.iterrows():
        db_feats = np.array(json.loads(row['features']))
        dist = np.linalg.norm(features - db_feats) # Euclidean distance
        if dist < min_dist:
            min_dist = dist
            best_brand = row['brand']
            
    if min_dist < threshold:
        return best_brand
    return "Unknown"

def update_knowledge(track_crops, updated_labels):
    """Save user's manual labels into the Pandas CSV database to learn for next time."""
    df = load_knowledge()
    new_rows = []
    
    for tid, brand in updated_labels.items():
        if brand != "Unknown":
            img = track_crops[tid]
            feats = extract_features(img)
            new_rows.append({
                "features": json.dumps(feats.tolist()), 
                "brand": brand
            })
            
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(KNOWLEDGE_FILE, index=False)

# --- Video Processing Function ---
def process_video(input_path, output_path, model, knowledge_df, custom_labels=None):
    """Run YOLOv8 tracking, draw boxes, and assign labels based on DB or user input."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Temporary openCV output
    temp_cv2_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_cv2_out, fourcc, fps, (width, height))
    
    track_crops = {}  # Store the cropped image of each unique bat
    track_brands = {} # Store the assigned brand of each unique bat
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Run YOLO tracking
        results = model.track(frame, persist=True, classes=BAT_CLASSES, verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # If we haven't seen this bat ID yet, process it
                if track_id not in track_crops:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        track_crops[track_id] = crop_rgb
                        feats = extract_features(crop_rgb)
                        
                        # Determine the brand (Manual Override > Database Prediction)
                        if custom_labels and track_id in custom_labels:
                            brand = custom_labels[track_id]
                        else:
                            brand = predict_brand(feats, knowledge_df)
                        track_brands[track_id] = brand
                
                # Draw bounding box and label
                brand_label = track_brands.get(track_id, "Unknown")
                text = f"Bat {int(track_id)}: {brand_label}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        out.write(frame)
        
    cap.release()
    out.release()
    
    # Convert OpenCV output to HTML5 friendly H264 format using MoviePy
    clip = VideoFileClip(temp_cv2_out)
    clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
    
    # Cleanup Temp File
    clip.close()
    os.remove(temp_cv2_out)
    
    return track_crops, track_brands

# --- Streamlit UI ---
st.title("🏏 Cricket Bat Detector & Brand Learner")
st.markdown("Upload a video. The system uses YOLOv8 to detect bats, count them, and will try to identify the brand based on previous manual labeling. *Note: Uses COCO 'baseball bat' class as a generic proxy.*")

# Initialize Session State
if "processed" not in st.session_state:
    st.session_state.processed = False
if "file_name" not in st.session_state:
    st.session_state.file_name = None

uploaded_file = st.file_uploader("Upload Cricket Video", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    # Check if a new file was uploaded
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.file_name = uploaded_file.name
        st.session_state.processed = False
        st.session_state.custom_labels = {}

    # Paths setup
    tfile_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile_input.write(uploaded_file.read())
    input_path = tfile_input.name
    output_path = "output_video.mp4"

    # Processing trigger
    if not st.session_state.processed:
        with st.spinner("Analyzing video with YOLOv8..."):
            model = YOLO("yolov8n.pt") # Loads the lightweight YOLOv8 model
            knowledge_df = load_knowledge()
            
            crops, brands = process_video(
                input_path, 
                output_path, 
                model, 
                knowledge_df, 
                custom_labels=st.session_state.get("custom_labels", {})
            )
            
            st.session_state.track_crops = crops
            st.session_state.track_brands = brands
            st.session_state.output_path = output_path
            st.session_state.processed = True
            st.rerun()

    # Once Processed: Display results and Labeling UI
    if st.session_state.processed:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Analyzed Video")
            st.video(st.session_state.output_path)
            
        with col2:
            st.subheader("Detection Stats")
            total_bats = len(st.session_state.track_crops)
            st.metric("Total Bats Detected", total_bats)
            
            st.subheader("Manual Labeling")
            st.markdown("Label brands to teach the system for next time:")
            
            BRAND_OPTIONS = ["Unknown", "SG", "SS", "MRF", "Other"]
            
            with st.form("label_form"):
                user_labels = {}
                for tid, crop in st.session_state.track_crops.items():
                    st.image(crop, caption=f"Bat ID: {int(tid)}", width=120)
                    
                    # Pre-select based on earlier prediction or custom selection
                    current_pred = st.session_state.track_brands[tid]
                    default_idx = BRAND_OPTIONS.index(current_pred) if current_pred in BRAND_OPTIONS else 0
                    
                    user_labels[tid] = st.selectbox(
                        f"Brand for Bat {int(tid)}", 
                        BRAND_OPTIONS, 
                        index=default_idx
                    )
                    st.divider()
                    
                submit = st.form_submit_button("Save Labels & Re-process")
                
                if submit:
                    # Save to database to learn for next time
                    update_knowledge(st.session_state.track_crops, user_labels)
                    st.success("Labels saved! The system has learned these profiles.")
                    
                    # Re-render video with new labels
                    st.session_state.custom_labels = user_labels
                    st.session_state.processed = False
                    st.rerun()