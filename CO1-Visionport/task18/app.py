from collections import defaultdict, deque
from pathlib import Path
import math
import shutil
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

for folder in (STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(BASE_DIR / "yolov8n.pt") if (BASE_DIR / "yolov8n.pt").exists() else "yolov8n.pt"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
CLASS_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


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
    max_w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for line in lines)
    box_w = max_w + 30
    box_h = len(lines) * line_height + 18

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (18, 22, 30), -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)
    cv2.rectangle(frame, (x, y), (x + 5, y + box_h), accent_color, -1)

    for idx, line in enumerate(lines):
        ty = y + 26 + idx * line_height
        color = accent_color if idx == 0 else (240, 244, 248)
        cv2.putText(
            frame, line, (x + 16, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.63, color, 2, cv2.LINE_AA
        )


class RobustCentroidTracker:
    """Reliable multi-object tracker used directly or as fallback."""
    def __init__(self, max_dist=75, max_missing=12):
        self.next_id = 1
        self.objects = {}
        self.missing = {}
        self.max_dist = max_dist
        self.max_missing = max_missing

    def update(self, boxes_xyxy):
        if not boxes_xyxy:
            for oid in list(self.missing.keys()):
                self.missing[oid] += 1
                if self.missing[oid] > self.max_missing:
                    del self.missing[oid]
                    self.objects.pop(oid, None)
            return []

        centroids = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes_xyxy]
        if not self.objects:
            assigned = []
            for c in centroids:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = c
                self.missing[oid] = 0
                assigned.append(oid)
            return assigned

        obj_ids = list(self.objects.keys())
        obj_pts = [self.objects[oid] for oid in obj_ids]

        pairs = []
        for bi, (cx, cy) in enumerate(centroids):
            for oi, (ox, oy) in enumerate(obj_pts):
                d = math.hypot(cx - ox, cy - oy)
                if d <= self.max_dist:
                    pairs.append((d, bi, oi))
        pairs.sort(key=lambda x: x[0])

        used_boxes = set()
        used_objs = set()
        assigned_map = {}

        for _, bi, oi in pairs:
            if bi in used_boxes or oi in used_objs:
                continue
            oid = obj_ids[oi]
            assigned_map[bi] = oid
            self.objects[oid] = centroids[bi]
            self.missing[oid] = 0
            used_boxes.add(bi)
            used_objs.add(oi)

        for oi, oid in enumerate(obj_ids):
            if oi not in used_objs:
                self.missing[oid] = self.missing.get(oid, 0) + 1
                if self.missing[oid] > self.max_missing:
                    del self.missing[oid]
                    del self.objects[oid]

        assigned = []
        for bi, c in enumerate(centroids):
            if bi in assigned_map:
                assigned.append(assigned_map[bi])
            else:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = c
                self.missing[oid] = 0
                assigned.append(oid)
        return assigned

app = FastAPI(title="VisionPort - Task 18: Wrong-Way Vehicle Alerts")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

model = YOLO(MODEL_PATH)

FLOW_VECTORS = {
    "down": (0.0, 1.0, "TOP -> BOTTOM"),
    "up": (0.0, -1.0, "BOTTOM -> TOP"),
    "right": (1.0, 0.0, "LEFT -> RIGHT"),
    "left": (-1.0, 0.0, "RIGHT -> LEFT"),
}


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/process-video/")
async def process_video(
    file: UploadFile = File(...),
    flow_dir: str = Form("auto"),
):
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
    tracker = RobustCentroidTracker(max_dist=max(65, width // 11))
    history = defaultdict(lambda: deque(maxlen=15))
    wrong_way_streak = defaultdict(int)
    wrong_way_ids = set()
    dominant_accum = [0.0, 0.0]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res = model.predict(frame, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)[0]
            boxes_list, cls_list = [], []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes_list = res.boxes.xyxy.cpu().tolist()
                cls_list = res.boxes.cls.int().cpu().tolist()

            ids_list = tracker.update(boxes_list)

            # Determine active allowed flow unit vector
            if flow_dir in FLOW_VECTORS:
                fx, fy, flow_label = FLOW_VECTORS[flow_dir]
            else:
                mag = math.hypot(dominant_accum[0], dominant_accum[1])
                if mag > 10.0:
                    fx, fy = dominant_accum[0] / mag, dominant_accum[1] / mag
                    if abs(fy) >= abs(fx):
                        flow_label = "AUTO: TOP -> BOTTOM" if fy > 0 else "AUTO: BOTTOM -> TOP"
                    else:
                        flow_label = "AUTO: LEFT -> RIGHT" if fx > 0 else "AUTO: RIGHT -> LEFT"
                else:
                    fx, fy, flow_label = 0.0, 1.0, "AUTO: CALIBRATING..."

            active_wrong_way = 0
            for box, tid, cls_id in zip(boxes_list, ids_list, cls_list):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                history[tid].append((cx, cy))

                pts = list(history[tid])
                if len(pts) >= 4:
                    dx = pts[-1][0] - pts[0][0]
                    dy = pts[-1][1] - pts[0][1]
                    dist = math.hypot(dx, dy)
                    if dist >= 14.0:
                        if flow_dir == "auto":
                            dominant_accum[0] += dx
                            dominant_accum[1] += dy
                        proj = (dx / dist) * fx + (dy / dist) * fy
                        if proj < -0.55:
                            wrong_way_streak[tid] += 1
                            if wrong_way_streak[tid] >= 3:
                                wrong_way_ids.add(tid)
                        else:
                            wrong_way_streak[tid] = max(0, wrong_way_streak[tid] - 1)

                if tid in wrong_way_ids:
                    active_wrong_way += 1
                    color = (0, 0, 255)
                    label = f"ALERT! ID #{tid} WRONG WAY"
                else:
                    color = (50, 230, 90)
                    label = f"ID #{tid} {CLASS_NAMES.get(cls_id, 'veh').upper()} (OK)"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if tid in wrong_way_ids else 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Draw Allowed Flow indicator arrow on top-right
            ax, ay = width - 70, 65
            end_x = int(ax + fx * 32)
            end_y = int(ay + fy * 32)
            cv2.circle(frame, (ax, ay), 40, (18, 22, 30), -1)
            cv2.arrowedLine(frame, (int(ax - fx * 22), int(ay - fy * 22)), (end_x, end_y), (50, 235, 90), 3, tipLength=0.35)

            if active_wrong_way > 0:
                cv2.rectangle(frame, (3, 3), (width - 4, height - 4), (0, 0, 255), 7)

            draw_hud_box(
                frame,
                [
                    "VisionPort | Task 18: Wrong-Way Vehicle Detector",
                    f"Allowed Flow: {flow_label}",
                    f"Wrong-Way Alerts: {len(wrong_way_ids)} | Active Now: {active_wrong_way}",
                ],
                accent_color=(0, 0, 255) if active_wrong_way > 0 else (50, 235, 90),
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
        in_path.unlink(missing_ok=True)

    return {
        "video_url": f"/outputs/{out_path.name}",
        "download_name": f"Task18_Wrong_Way_Alerts{out_path.suffix}",
        "wrong_way_count": len(wrong_way_ids),
        "flow_direction": flow_label,
        "status": "WRONG-WAY ALERT" if len(wrong_way_ids) > 0 else "NORMAL FLOW",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
