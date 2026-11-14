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

app = FastAPI(title="VisionPort - Task 14: Traffic Signal Status Detection")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

model = YOLO(str(BASE_DIR / "yolov8n.pt"))


def classify_signal_hsv(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255]))
    y_mask = cv2.inRange(hsv, np.array([18, 120, 120]), np.array([34, 255, 255]))
    g_mask = cv2.inRange(hsv, np.array([40, 80, 100]), np.array([90, 255, 255]))

    r_px = int(cv2.countNonZero(r1 | r2))
    y_px = int(cv2.countNonZero(y_mask))
    g_px = int(cv2.countNonZero(g_mask))
    mx = max(r_px, y_px, g_px)

    if mx < 25:
        return "UNKNOWN", (200, 200, 200)
    if mx == r_px:
        return "RED", (40, 40, 255)
    if mx == y_px:
        return "YELLOW", (0, 220, 255)
    return "GREEN", (50, 230, 80)


def draw_signal_widget(frame, status):
    x, y = 18, 95
    cv2.rectangle(frame, (x, y), (x + 56, y + 148), (20, 24, 32), -1)
    cv2.rectangle(frame, (x, y), (x + 56, y + 148), (120, 130, 145), 2)

    red_col = (40, 40, 255) if status == "RED" else (45, 45, 70)
    yel_col = (0, 220, 255) if status == "YELLOW" else (55, 55, 45)
    grn_col = (50, 230, 80) if status == "GREEN" else (45, 65, 45)

    cv2.circle(frame, (x + 28, y + 28), 16, red_col, -1)
    cv2.circle(frame, (x + 28, y + 74), 16, yel_col, -1)
    cv2.circle(frame, (x + 28, y + 120), 16, grn_col, -1)


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
    counts = {"RED": 0, "YELLOW": 0, "GREEN": 0, "UNKNOWN": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect traffic lights (COCO class 9) if visible, otherwise inspect upper signal zone
            res = model.predict(frame, classes=[9], conf=0.25, verbose=False)[0]
            detected_status = "UNKNOWN"
            accent = (200, 200, 200)

            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        st, col = classify_signal_hsv(roi)
                        if st != "UNKNOWN":
                            detected_status, accent = st, col
                        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                        cv2.putText(frame, f"Signal: {st}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            else:
                upper_roi = frame[0:max(1, height // 2), :]
                detected_status, accent = classify_signal_hsv(upper_roi)

            counts[detected_status] += 1
            action_text = {
                "RED": "RED (STOP)",
                "YELLOW": "YELLOW (GET READY)",
                "GREEN": "GREEN (GO)",
                "UNKNOWN": "SCANNING SIGNAL...",
            }[detected_status]

            draw_hud_box(
                frame,
                [
                    "VisionPort | Task 14: Traffic Signal Status",
                    f"Active Signal: {action_text}",
                ],
                accent_color=accent,
            )
            draw_signal_widget(frame, detected_status)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
        in_path.unlink(missing_ok=True)

    dominant = max(counts, key=counts.get)
    return {
        "video_url": f"/outputs/{out_path.name}",
        "download_name": f"Task14_Signal_Status{out_path.suffix}",
        "dominant_signal": dominant,
        "counts": counts,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
