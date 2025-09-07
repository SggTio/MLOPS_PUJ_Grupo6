# api/main.py
import os, threading, joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
PREP_PATH  = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")

app = FastAPI()
_model = None
_pre   = None
_lock = threading.Lock()

class Record(BaseModel):
    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str
    year: int

class PredictRequest(BaseModel):
    records: List[Record] = Field(..., min_items=1)

def ensure_loaded():
    global _model, _pre
    with _lock:
        if _model is None or _pre is None:
            if not (os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH)):
                raise HTTPException(status_code=503, detail="Artifacts not found")
            try:
                _pre = joblib.load(PREP_PATH)
                _model = joblib.load(MODEL_PATH)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load artifacts: {e}")

def apply_preprocessor(pre, df: pd.DataFrame) -> np.ndarray:
    # dict bundle (our current pipeline)
    if isinstance(pre, dict):
        try:
            num_cols = pre["NUMERIC_COLS"]
            cat_cols = pre["CATEGORICAL_COLS"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Invalid preprocessor bundle: {e}")
        # check columns exist
        missing = [c for c in (num_cols + cat_cols) if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        # numeric
        X_num = df[num_cols].apply(pd.to_numeric, errors="coerce")
        X_num = pre["num_imputer"].transform(X_num)
        X_num = pre["num_scaler"].transform(X_num)
        # categorical
        cats = df[cat_cols].astype("string").fillna(pre.get("cat_fill_value", "missing"))
        X_cat = pre["ohe"].transform(cats)
        return np.hstack([X_num, X_cat])
    # single transformer/pipeline support
    try:
        return pre.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessor transform failed: {e}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts_dir": ARTIFACTS_DIR,
        "model": _model is not None,
        "preprocessor": _pre is not None,
        "paths": {"model": MODEL_PATH, "preprocessor": PREP_PATH},
    }

@app.post("/predict")
def predict(req: PredictRequest):
    ensure_loaded()
    df = pd.DataFrame([r.model_dump() for r in req.records])
    try:
        X = apply_preprocessor(_pre, df)
        preds = _model.predict(X)
        return {"predictions": list(map(str, preds))}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

 