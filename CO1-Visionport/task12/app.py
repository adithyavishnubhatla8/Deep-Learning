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

app = FastAPI(title="VisionPort - Task 12: Vehicle Tracking IDs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

model = YOLO(str(BASE_DIR / "yolov8n.pt"))


class SimpleCentroidTracker:
    """Fallback tracker if ByteTrack dependencies are unavailable offline."""
    def __init__(self, max_dist=65):
        self.next_id = 1
        self.objects = {}
        self.max_dist = max_dist

    def update(self, boxes_xyxy):
        assigned = []
        new_objects = {}
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            best_id, best_d = None, self.max_dist
            for oid, (ox, oy) in self.objects.items():
                d = math.hypot(cx - ox, cy - oy)
                if d < best_d:
                    best_d, best_id = d, oid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            new_objects[best_id] = (cx, cy)
            assigned.append(best_id)
        self.objects = new_objects
        return assigned


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
    track_trails = defaultdict(lambda: deque(maxlen=25))
    unique_ids = set()
    fallback_tracker = SimpleCentroidTracker()
    use_yolo_tracker = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes_list = []
            ids_list = []
            cls_list = []

            if use_yolo_tracker:
                try:
                    res = model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)[0]
                    if res.boxes is not None and len(res.boxes) > 0:
                        boxes_list = res.boxes.xyxy.cpu().tolist()
                        cls_list = res.boxes.cls.int().cpu().tolist()
                        if res.boxes.id is not None:
                            ids_list = res.boxes.id.int().cpu().tolist()
                        else:
                            ids_list = fallback_tracker.update(boxes_list)
                except Exception:
                    use_yolo_tracker = False

            if not use_yolo_tracker:
                res = model.predict(frame, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes_list = res.boxes.xyxy.cpu().tolist()
                    cls_list = res.boxes.cls.int().cpu().tolist()
                    ids_list = fallback_tracker.update(boxes_list)

            for box, tid, cls_id in zip(boxes_list, ids_list, cls_list):
                unique_ids.add(tid)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track_trails[tid].append((cx, cy))

                color = ((tid * 67) % 200 + 55, (tid * 131) % 200 + 55, (tid * 197) % 200 + 55)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID #{tid} {CLASS_NAMES.get(cls_id, 'vehicle').upper()}"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                pts = list(track_trails[tid])
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], color, 2)

            draw_hud_box(
                frame,
                [
                    "VisionPort | Task 12: Multi-Object Vehicle Tracking",
                    f"Active Vehicles: {len(ids_list)} | Total Unique IDs: {len(unique_ids)}",
                ],
                accent_color=(168, 85, 247),
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
        in_path.unlink(missing_ok=True)

    return {
        "video_url": f"/outputs/{out_path.name}",
        "download_name": f"Task12_Vehicle_Tracking{out_path.suffix}",
        "unique_vehicles_tracked": len(unique_ids),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
