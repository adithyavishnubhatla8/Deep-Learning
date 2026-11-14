from collections import defaultdict, deque
from pathlib import Path
import io
import math
import shutil
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

for folder in (STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
CLASS_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_COLORS = {
    1: (255, 220, 0),    # Cyan-Blue for bicycle
    2: (50, 220, 80),    # Green for car
    3: (255, 180, 0),    # Sky Blue for motorcycle
    5: (0, 165, 255),    # Orange for bus
    7: (220, 60, 220),   # Magenta for truck
}


def create_browser_video_writer(output_dir: Path, job_id: str, fps: float, width: int, height: int):
    """Creates a VideoWriter using a codec natively playable in HTML5 <video> tags."""
    candidates = [
        ("VP80", ".webm", "video/webm"),
        ("avc1", ".mp4", "video/mp4"),
        ("H264", ".mp4", "video/mp4"),
        ("mp4v", ".mp4", "video/mp4"),
    ]
    for fourcc_name, ext, mime_type in candidates:
        out_path = output_dir / f"{job_id}_output{ext}"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, out_path, mime_type
    raise RuntimeError("OpenCV could not initialize any compatible video writer.")


def save_uploaded_video(file: UploadFile, job_id: str) -> Path:
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Please upload MP4, AVI, MOV, MKV, or WEBM.",
        )
    in_path = UPLOAD_DIR / f"{job_id}{ext}"
    with in_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return in_path


def draw_hud_box(frame, lines, top_left=(15, 15), accent_color=(40, 220, 90)):
    """Draws a clean semi-transparent dark HUD panel with crisp text lines."""
    x, y = top_left
    line_height = 28
    max_w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0][0] for line in lines)
    box_w = max_w + 30
    box_h = len(lines) * line_height + 18

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (18, 22, 30), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x, y), (x + 5, y + box_h), accent_color, -1)

    for idx, line in enumerate(lines):
        ty = y + 26 + idx * line_height
        color = accent_color if idx == 0 else (240, 244, 248)
        cv2.putText(
            frame, line, (x + 16, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
        )

app = FastAPI(title="VisionPort - Task 8: Vehicle Labels & Confidence Scores")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

model = YOLO(str(BASE_DIR / "yolov8n.pt"))


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    in_path = save_uploaded_video(file, job_id)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        in_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not read uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer, out_path, _ = create_browser_video_writer(OUTPUT_DIR, job_id, fps, width, height)
    frames = 0
    conf_scores = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
            result = model.predict(frame, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)[0]

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    conf_scores.append(conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    name = CLASS_NAMES.get(cls_id, model.names[cls_id]).upper()
                    badge_text = f"{name}: {conf * 100:.1f}%"
                    color = CLASS_COLORS.get(cls_id, (56, 189, 248))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    top_y = max(th + 8, y1)
                    cv2.rectangle(frame, (x1, top_y - th - 8), (x1 + tw + 8, top_y), color, -1)
                    cv2.putText(
                        frame, badge_text, (x1 + 4, top_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2, cv2.LINE_AA
                    )

            avg_frame_conf = (sum(conf_scores[-20:]) / len(conf_scores[-20:]) * 100) if conf_scores else 0.0
            draw_hud_box(
                frame,
                [
                    "VisionPort | Task 8: Labels & Confidence Scores",
                    f"Recent Avg Confidence: {avg_frame_conf:.1f}%",
                ],
                accent_color=(56, 189, 248),
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
        in_path.unlink(missing_ok=True)

    avg_conf = round((sum(conf_scores) / len(conf_scores)) * 100, 1) if conf_scores else 0.0
    max_conf = round(max(conf_scores) * 100, 1) if conf_scores else 0.0

    return {
        "video_url": f"/outputs/{out_path.name}",
        "download_name": f"Task8_Labels_Confidence{out_path.suffix}",
        "total_detections": len(conf_scores),
        "avg_confidence": avg_conf,
        "max_confidence": max_conf,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
