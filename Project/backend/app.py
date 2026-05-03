import io
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

from model import InferenceEngine



_HERE        = Path(__file__).parent
PROJECT_ROOT = _HERE.parent
CKPT_PATH    = PROJECT_ROOT / "artifacts/checkpoints/baseline_best.pt"
NAME_LE_PATH = PROJECT_ROOT / "data/processed/name_le.pkl"
MAT_LE_PATH  = PROJECT_ROOT / "data/processed/mat_le.pkl"

app = FastAPI(
    title="SIMILIS Artifact Description API",
    description="Predicts artifact category, material, and generates auto_description from images.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[InferenceEngine] = None


@app.on_event("startup")
def load_model():
    global engine
    if CKPT_PATH.exists() and NAME_LE_PATH.exists() and MAT_LE_PATH.exists():
        engine = InferenceEngine(CKPT_PATH, NAME_LE_PATH, MAT_LE_PATH)
        print("Model loaded")
    else:
        print("Сheckpoint or encoders not found")


class PredictionResponse(BaseModel):
    pred_name: str
    pred_material: str
    conf_name: float
    conf_material: float
    auto_description: str
    top3_names: list
    top3_materials: list
    model_ready: bool


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": engine is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image.")

    if engine is None:
        return PredictionResponse(
            pred_name="Тарелка",
            pred_material="Фаянс",
            conf_name=0.0,
            conf_material=0.0,
            auto_description="Модель не обучена. Запустите ноутбук для обучения.",
            top3_names=[],
            top3_materials=[],
            model_ready=False,
        )

    result = engine.predict(img)
    result["model_ready"] = True
    return PredictionResponse(**result)


@app.get("/classes")
def get_classes():
    if engine is None:
        return {"error": "Model not loaded"}
    return {
        "name_classes":     engine.name_le.classes_.tolist(),
        "material_classes": engine.mat_le.classes_.tolist(),
    }


@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = _HERE / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h2>SIMILIS API is running. See /docs for API reference.</h2>")
