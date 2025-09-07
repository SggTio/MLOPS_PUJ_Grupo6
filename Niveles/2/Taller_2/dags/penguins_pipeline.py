from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
from airflow.utils.dates import days_ago

DATA_DB_URI = os.environ["DATA_DB_URI"]
ARTIFACTS_DIR = Path("/opt/airflow/artifacts")
STATUS_DIR = ARTIFACTS_DIR / "status"
RAW_SCHEMA, PROC_SCHEMA = "raw", "processed"
RAW_TABLE, PROC_TABLE = "penguins_raw", "penguins_processed"

def write_status(task_id: str, status: str, msg: str, extra: dict | None = None):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "status": status,                 # "ok" | "error"
        "message": msg,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if extra: payload.update(extra)
    with open(STATUS_DIR / f"{task_id}.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload  # also returned as XCom

@dag(
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    default_args={"owner": "airflow"},
    tags=["penguins","ml","mysql"],
)
def penguins_pipeline():
    @task
    def wipe_data():
        task_id = "wipe_data"
        try:
            eng = create_engine(DATA_DB_URI)
            with eng.begin() as con:
                #con.execute(text(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}"))
                #con.execute(text(f"CREATE SCHEMA IF NOT EXISTS {PROC_SCHEMA}"))
                con.execute(text(f"DROP TABLE IF EXISTS {RAW_SCHEMA}.{RAW_TABLE}"))
                con.execute(text(f"DROP TABLE IF EXISTS {PROC_SCHEMA}.{PROC_TABLE}"))
            return write_status(task_id, "ok", "raw & processed tables dropped")
        except Exception as e:
            write_status(task_id, "error", f"{e}")
            raise

    @task
    def load_raw():
        task_id = "load_raw"
        try:
            from palmerpenguins import load_penguins
            df = load_penguins()  # no preprocessing
            eng = create_engine(DATA_DB_URI)
            df.to_sql(RAW_TABLE, eng, schema=RAW_SCHEMA, if_exists="replace", index=False)
            return write_status(task_id, "ok", "raw loaded", {"rows": len(df)})
        except Exception as e:
            write_status(task_id, "error", f"{e}")
            raise

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from joblib import dump
    import numpy as np
    
    @task
    def preprocess():
        task_id = "preprocess"
        try:
            eng = create_engine(DATA_DB_URI)
            raw = pd.read_sql_table(RAW_TABLE, con=eng, schema=RAW_SCHEMA)
    
            LABEL_COL = "species"
            NUMERIC_COLS = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year']
            CATEGORICAL_COLS = ['island', 'sex']
    
            # Validate columns
            miss_num = [c for c in NUMERIC_COLS if c not in raw.columns]
            miss_cat = [c for c in CATEGORICAL_COLS if c not in raw.columns]
            if miss_num or miss_cat:
                raise ValueError(f"Missing expected columns. numeric={miss_num}, categorical={miss_cat}")
    
            # Split
            y = raw[LABEL_COL] if LABEL_COL in raw.columns else None
            X = raw.drop(columns=[LABEL_COL], errors='ignore').copy()
    
            # ----- Numeric: coerce -> np.nan, then impute + scale -----
            for c in NUMERIC_COLS:
                X[c] = pd.to_numeric(X[c], errors='coerce')
    
            num_imputer = SimpleImputer(strategy="median")
            X_num = num_imputer.fit_transform(X[NUMERIC_COLS])
            num_scaler = StandardScaler()
            X_num = num_scaler.fit_transform(X_num)  # dense ndarray
    
            # ----- Categorical: fill sentinel (no SimpleImputer) -> OHE -----
            # Convert to pandas string dtype, then fill with a safe token (no pd.NA left)
            cats = X[CATEGORICAL_COLS].astype('string').fillna('missing')
            # OneHotEncoder happily accepts a DataFrame/ndarray of strings
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat = ohe.fit_transform(cats)
    
            # ----- Concatenate -----
            Xp = np.hstack([X_num, X_cat])
    
            # Feature names (robust; no ColumnTransformer)
            try:
                num_names = list(NUMERIC_COLS)
                cat_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
                feature_names = num_names + cat_names
            except Exception:
                feature_names = [f"f{i}" for i in range(Xp.shape[1])]
    
            # Build processed DataFrame
            Xp_df = pd.DataFrame(Xp, columns=feature_names)
            if y is not None:
                Xp_df[LABEL_COL] = y.values
    
            # Persist processed table
            Xp_df.to_sql(PROC_TABLE, eng, schema=PROC_SCHEMA, if_exists="replace", index=False)
    
            # Persist a compact preprocessor bundle
            pre_bundle = {
                "NUMERIC_COLS": NUMERIC_COLS,
                "CATEGORICAL_COLS": CATEGORICAL_COLS,
                "num_imputer": num_imputer,
                "num_scaler": num_scaler,
                "ohe": ohe,
                "cat_fill_value": "missing",
            }
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            dump(pre_bundle, ARTIFACTS_DIR / "preprocessor.pkl")
    
            return write_status(task_id, "ok", "preprocess done",
                                {"rows": int(Xp_df.shape[0]), "features": int(len(feature_names))})
        except Exception as e:
            write_status(task_id, "error", f"{e}")
            raise

 

    @task
    def train_model():
        task_id = "train_model"
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from joblib import dump

            eng = create_engine(DATA_DB_URI)
            df = pd.read_sql_table(PROC_TABLE, con=eng, schema=PROC_SCHEMA)
            y = df["species"].values
            X = df.drop(columns=["species"])

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X, y)
            acc = accuracy_score(y, clf.predict(X))  # demo metric

            dump(clf, ARTIFACTS_DIR / "model.pkl")
            with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
                json.dump({"accuracy_training_set": float(acc)}, f, indent=2)

            return write_status(task_id, "ok", "model trained", {"accuracy_training_set": float(acc)})
        except Exception as e:
            write_status(task_id, "error", f"{e}")
            raise

    w = wipe_data()
    r = load_raw()
    p = preprocess()
    t = train_model()
    w >> r >> p >> t

penguins_pipeline()
