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

app = FastAPI(title="VisionPort - Task 16: Junction Entry & Exit Vehicle Count")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

model = YOLO(MODEL_PATH)


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/process-video/")
async def process_video(
    file: UploadFile = File(...),
    entry_pct: int = Form(38),
    exit_pct: int = Form(70),
):
    entry_pct = max(10, min(entry_pct, 85))
    exit_pct = max(15, min(exit_pct, 92))
    if exit_pct <= entry_pct + 5:
        exit_pct = min(92, entry_pct + 20)

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

    entry_y = int(height * (entry_pct / 100.0))
    exit_y = int(height * (exit_pct / 100.0))

    writer, out_path, _ = create_browser_video_writer(OUTPUT_DIR, job_id, fps, width, height)
    tracker = RobustCentroidTracker(max_dist=max(60, width // 12))
    prev_y = {}
    entered_ids = set()
    exited_ids = set()
    all_seen_ids = set()
    entry_flash = 0
    exit_flash = 0

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

            # Draw shaded junction zone between Entry and Exit lines
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, entry_y), (width, exit_y), (30, 80, 110), -1)
            cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)

            for box, tid, cls_id in zip(boxes_list, ids_list, cls_list):
                all_seen_ids.add(tid)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if tid in prev_y:
                    py = prev_y[tid]
                    # Crossing Entry line
                    if ((py < entry_y <= cy) or (py > entry_y >= cy)) and tid not in entered_ids:
                        entered_ids.add(tid)
                        entry_flash = 6
                    # Crossing Exit line
                    if ((py < exit_y <= cy) or (py > exit_y >= cy)) and tid not in exited_ids:
                        exited_ids.add(tid)
                        exit_flash = 6

                prev_y[tid] = cy

                if tid in exited_ids:
                    color = (0, 165, 255)
                    tag = f"ID #{tid} EXITED"
                elif tid in entered_ids:
                    color = (50, 230, 90)
                    tag = f"ID #{tid} ENTERED"
                else:
                    color = (230, 210, 80)
                    tag = f"ID #{tid} {CLASS_NAMES.get(cls_id, 'veh').upper()}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.putText(frame, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Draw Entry Line (Green) & Exit Line (Orange)
            e_thick = 4 if entry_flash > 0 else 2
            x_thick = 4 if exit_flash > 0 else 2
            if entry_flash > 0:
                entry_flash -= 1
            if exit_flash > 0:
                exit_flash -= 1

            cv2.line(frame, (0, entry_y), (width, entry_y), (50, 235, 90), e_thick)
            cv2.putText(
                frame, f"JUNCTION ENTRY LINE ({len(entered_ids)})", (16, max(22, entry_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 235, 90), 2, cv2.LINE_AA
            )

            cv2.line(frame, (0, exit_y), (width, exit_y), (0, 165, 255), x_thick)
            cv2.putText(
                frame, f"JUNCTION EXIT LINE ({len(exited_ids)})", (16, max(22, exit_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
            )

            draw_hud_box(
                frame,
                [
                    "VisionPort | Task 16: Junction Entry & Exit Counter",
                    f"Entered: {len(entered_ids)} | Exited: {len(exited_ids)} | Active in Junction: {max(0, len(entered_ids) - len(exited_ids))}",
                ],
                accent_color=(50, 235, 90),
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
        in_path.unlink(missing_ok=True)

    return {
        "video_url": f"/outputs/{out_path.name}",
        "download_name": f"Task16_Entry_Exit_Count{out_path.suffix}",
        "entered_count": len(entered_ids),
        "exited_count": len(exited_ids),
        "net_inside": max(0, len(entered_ids) - len(exited_ids)),
        "total_unique": len(all_seen_ids),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
