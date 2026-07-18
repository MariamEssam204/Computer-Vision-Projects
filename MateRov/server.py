from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil, tempfile, os, time

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = YOLO("crab_detector_yolo11s100epoch.pt")  # ← your fine-tuned weights

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    conf: float = Form(0.5),
    iou: float = Form(0.45),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    t0 = time.time()
    results = model(tmp_path, conf=conf, iou=iou)[0]
    elapsed_ms = round((time.time() - t0) * 1000)
    os.remove(tmp_path)

    boxes = results.boxes
    count = len(boxes)

    detections = []
    classes = {}
    for box in boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf_score = float(box.conf)
        detections.append({"class": cls_name, "confidence": conf_score})
        classes[cls_name] = classes.get(cls_name, 0) + 1

    avg_conf = sum(d["confidence"] for d in detections) / count if count else 0

    return {
        "count": count,
        "avg_confidence": round(avg_conf, 4),
        "classes": classes,
        "detections": detections,
        "inference_ms": elapsed_ms,
    }