import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

# Initialize FastAPI and load the object detection model
app = FastAPI(title="Indian Traffic Detection API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
model = YOLO("yolov8n.pt")  # Swap with custom-trained weights (e.g., 'best.pt') for specific Indian vehicles


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # 1. Read the uploaded image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run Object Detection inference
    results = model(img)

    # 3. Annotate the image with bounding boxes and labels
    annotated_img = results[0].plot()

    # 4. Stream the processed image back in real time
    _, encoded_img = cv2.imencode(".jpg", annotated_img)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
