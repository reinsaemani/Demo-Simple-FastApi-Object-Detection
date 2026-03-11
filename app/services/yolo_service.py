from ultralytics import YOLO
import cv2
import numpy as np

class YOLOService:

    def __init__(self):
        self.model = YOLO("../models/best.pt")
        self.conf_threshold = 0.5

    def detect(self, image):

        results = self.model(image)

        detections = []

        for r in results:
            for box in r.boxes:

                conf = float(box.conf[0])

                if conf < self.conf_threshold:
                    continue

                cls = int(box.cls[0])
                label = self.model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        return detections