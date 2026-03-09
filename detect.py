from ultralytics import YOLO
import cv2

model = YOLO("models/cricket_model.pt")

def detect_products(frame):

    results = model(frame)

    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:

            cls = int(box.cls)
            label = model.names[cls]

            detections.append(label)

    return detections
