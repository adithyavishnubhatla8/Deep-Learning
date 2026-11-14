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
import zipfile

app = FastAPI(title="VisionPort - Task 5: Extract Video Frames")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/extract-frames/")
async def extract_frames(file: UploadFile = File(...), step: int = 25):
    step = max(1, min(step, 300))
    job_id = uuid.uuid4().hex
    in_path = save_uploaded_video(file, job_id)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        in_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="OpenCV could not open the uploaded video.")

    frames_dir = OUTPUT_DIR / f"{job_id}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    zip_path = OUTPUT_DIR / f"{job_id}_frames.zip"

    frame_idx = 0
    saved_urls = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                fname = f"frame_{len(saved_urls):04d}.jpg"
                fpath = frames_dir / fname
                cv2.imwrite(str(fpath), frame)
                saved_urls.append(f"/outputs/{job_id}_frames/{fname}")
            frame_idx += 1

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for img_file in sorted(frames_dir.glob("*.jpg")):
                zf.write(img_file, img_file.name)
    finally:
        cap.release()
        in_path.unlink(missing_ok=True)

    return {
        "total_video_frames": frame_idx,
        "extracted_count": len(saved_urls),
        "step": step,
        "zip_url": f"/outputs/{zip_path.name}",
        "frame_urls": saved_urls,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
