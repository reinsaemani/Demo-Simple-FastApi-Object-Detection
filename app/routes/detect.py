from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
from app.services.yolo_service import YOLOService
from fastapi.responses import StreamingResponse
import io
import json
import time
import os
from typing import List
import base64



router = APIRouter()

yolo = YOLOService()

@router.post("/detect")
async def detect(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = yolo.detect(image)

    # logging
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    filename = f"{LOG_DIR}/detect_{int(time.time())}.json"

    log_data = {
    "filename": file.filename,
    "timestamp": int(time.time()),
    "detections": detections
}

    with open(filename, "w") as f:
        json.dump(log_data, f, indent=2)

    return {
        "total_objects": len(detections),
        "detections": detections
    }

@router.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = yolo.model(image)

    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )

@router.get("/classes")
def classes():
    return yolo.model.names



@router.post("/detect-batch")
async def detect_batch(files: List[UploadFile] = File(...)):

    results = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detections = yolo.detect(image)

        # buat annotated image
        results_yolo = yolo.model(image)
        annotated = results_yolo[0].plot()

        _, buffer = cv2.imencode(".jpg", annotated)
        img_str = base64.b64encode(buffer).decode("utf-8")

        results.append({
            "filename": file.filename,
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{img_str}"
        })

    return results