import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from joblib import load

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/app/artifacts")
PREPROC_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
MODEL_PATH   = os.path.join(ARTIFACTS_DIR, "model.pkl")

app = FastAPI(title="Penguins Inference API", version="1.0.0")
preproc = model = None

class PenguinIn(BaseModel):
    island: Optional[str] = None
    bill_length_mm: Optional[float] = None
    bill_depth_mm: Optional[float] = None
    flipper_length_mm: Optional[float] = None
    body_mass_g: Optional[float] = None
    sex: Optional[str] = None
    year: Optional[float] = None

class PredictRequest(BaseModel):
    records: List[PenguinIn]

@app.on_event("startup")
def load_artifacts():
    global preproc, model
    if not (os.path.exists(PREPROC_PATH) and os.path.exists(MODEL_PATH)):
        return
    preproc = load(PREPROC_PATH)
    model   = load(MODEL_PATH)

@app.get("/health")
def health():
    ok = preproc is not None and model is not None
    return {"status": "ok" if ok else "warmup",
            "preprocessor": bool(preproc), "model": bool(model)}

@app.post("/predict")
def predict(req: PredictRequest):
    if preproc is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    df = pd.DataFrame([r.dict() for r in req.records])
    Xp = preproc.transform(df)
    preds = model.predict(Xp)
    try:
        probs = model.predict_proba(Xp).max(axis=1).tolist()
    except Exception:
        probs = None
    return {"predictions": preds.tolist(), "confidence": probs}
