"""
SharedAlbum face server (version 3)
===================================
Finds every face in a photo and turns each one into 512 numbers (an
embedding). Photos of the same person give embeddings pointing in almost
the same direction, so they're compared with cosine similarity.

Models (InsightFace "buffalo_l", downloaded on first run, about 300 MB):
  - SCRFD-10G (RetinaFace family): finds faces and 5 landmarks (eyes, nose, mouth)
  - ArcFace ResNet-50 (trained on WebFace600K): turns an aligned face into 512 numbers

What this version fixes:
  1. Small faces were missed. Photos used to be shrunk to 640 px before
     detection, so a face in a group photo became about 10 px wide.
     Now the whole photo is checked at 1024 px AND, for big photos, in
     overlapping tiles at full detail. Duplicates are merged (NMS).
  2. Turned heads matched poorly. Each face is now aligned with its 5
     landmarks and its embedding is the average of the face and its
     mirror image, which makes it steadier across angles.
  3. Blurry or sideways faces caused wrong groups. Each face gets a
     quality score (size, sharpness, how frontal, detector confidence) so
     the website can be more careful with poor faces.

Run it:   uvicorn server:app --port 8000
Check it: http://127.0.0.1:8000/health
Stronger (slower, ~400 MB) model: set FACE_MODEL=antelopev2 before starting.
"""

import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
MODEL_NAME = os.environ.get("FACE_MODEL", "buffalo_l")
DETECT_SIZE = 1024          # the whole photo is checked at this size
TILE_SIZE = 640             # big photos are also checked in tiles this size...
TILE_OVERLAP = 0.25         # ...overlapping by this much
TILE_MIN_SIDE = 1300        # only photos bigger than this are tiled
MAX_IMAGE_SIDE = 3000       # larger photos are shrunk first
MIN_SCORE = 0.45            # how sure the detector must be that it's a face
MIN_FACE_SIZE = 18          # faces smaller than this (pixels) are too small to recognise

app = FastAPI(title="SharedAlbum Face Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

detector = None
recognizer = None
norm_crop = None


def load_models():
    """Load the detector and the recogniser once."""
    global detector, recognizer, norm_crop
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    print(f"Loading face models ({MODEL_NAME}). The first time downloads them...")
    fa = FaceAnalysis(name=MODEL_NAME, allowed_modules=["detection", "recognition"],
                      providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=-1, det_size=(DETECT_SIZE, DETECT_SIZE), det_thresh=MIN_SCORE)
    detector = fa.det_model
    recognizer = fa.models["recognition"]
    norm_crop = face_align.norm_crop
    print("Face models ready!")


# ---------------------------------------------------------------------
# Detection: whole photo + tiles, then remove duplicates
# ---------------------------------------------------------------------
def detect_all(bgr):
    """Returns boxes (N, 5: x1, y1, x2, y2, score) and landmarks (N, 5, 2)."""
    h, w = bgr.shape[:2]
    boxes, points = [], []

    b, k = detector.detect(bgr, input_size=(DETECT_SIZE, DETECT_SIZE), max_num=0)
    if b is not None and len(b):
        boxes.append(b); points.append(k)

    if max(h, w) > TILE_MIN_SIDE:
        step = int(TILE_SIZE * (1 - TILE_OVERLAP))
        for y0 in range(0, max(1, h - TILE_SIZE + step), step):
            for x0 in range(0, max(1, w - TILE_SIZE + step), step):
                y1, x1 = min(h, y0 + TILE_SIZE), min(w, x0 + TILE_SIZE)
                tile = bgr[y0:y1, x0:x1]
                if tile.shape[0] < 64 or tile.shape[1] < 64:
                    continue
                tb, tk = detector.detect(tile, input_size=(TILE_SIZE, TILE_SIZE), max_num=0)
                if tb is None or not len(tb):
                    continue
                tb = tb.copy(); tk = tk.copy()
                tb[:, [0, 2]] += x0; tb[:, [1, 3]] += y0
                tk[:, :, 0] += x0; tk[:, :, 1] += y0
                boxes.append(tb); points.append(tk)

    if not boxes:
        return np.zeros((0, 5)), np.zeros((0, 5, 2))
    boxes = np.concatenate(boxes); points = np.concatenate(points)
    keep = nms(boxes, 0.4)
    return boxes[keep], points[keep]


def nms(boxes, iou_limit):
    """Keep the most confident box where boxes overlap."""
    order = boxes[:, 4].argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / np.maximum(1e-6, area_i + area_o - inter)
        order = order[1:][iou < iou_limit]
    return keep


# ---------------------------------------------------------------------
# Recognition: aligned face + its mirror image, averaged
# ---------------------------------------------------------------------
def embed(bgr, kps):
    crop = norm_crop(bgr, landmark=kps, image_size=112)
    feats = recognizer.get_feat([crop, cv2.flip(crop, 1)])
    v = feats.mean(axis=0)
    return v / max(np.linalg.norm(v), 1e-9), crop


def quality(crop, box_size, score, kps):
    """0 to 1: bigger, sharper, more frontal and more certain faces score higher."""
    size_q = min(1.0, box_size / 112.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp_q = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 150.0)
    # How far the nose is from the middle of the eyes: 0 = facing the camera
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = max(1.0, np.linalg.norm(right_eye - left_eye))
    yaw = abs(nose[0] - (left_eye[0] + right_eye[0]) / 2) / eye_dist
    pose_q = max(0.0, 1.0 - yaw * 1.6)
    return round(float(0.3 * size_q + 0.25 * sharp_q + 0.25 * pose_q + 0.2 * score), 3)


# ---------------------------------------------------------------------
# Web endpoints
# ---------------------------------------------------------------------
@app.on_event("startup")
def startup():
    if detector is None:
        load_models()


@app.get("/")
def home():
    return {"status": "ok", "message": "SharedAlbum face server is running. The website uses /health and /detect."}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "detector": "SCRFD (RetinaFace family)",
            "recognizer": "ArcFace", "embedding_size": 512, "version": 3}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="That file isn't an image the server can read.")

    width, height = image.size
    scale = 1.0
    if max(width, height) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    bgr = np.asarray(image)[:, :, ::-1].copy()

    boxes, points = detect_all(bgr)
    faces = []
    for (x1, y1, x2, y2, score), kps in zip(boxes, points):
        bw, bh = x2 - x1, y2 - y1
        if min(bw, bh) < MIN_FACE_SIZE or score < MIN_SCORE:
            continue
        vec, crop = embed(bgr, kps)
        q = quality(crop, min(bw, bh), float(score), kps)
        X1, Y1 = max(0.0, x1 / scale), max(0.0, y1 / scale)
        X2, Y2 = min(float(width), x2 / scale), min(float(height), y2 / scale)
        faces.append({
            "box": {"x": round(X1), "y": round(Y1), "width": round(X2 - X1), "height": round(Y2 - Y1)},
            "score": round(float(score), 3),
            "quality": q,
            "embedding": [round(float(v), 6) for v in vec],
        })

    faces.sort(key=lambda f: f["box"]["width"] * f["box"]["height"], reverse=True)
    return {"width": width, "height": height, "faces": faces}
