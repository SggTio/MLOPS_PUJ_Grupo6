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
                con.execute(text(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}"))
                con.execute(text(f"CREATE SCHEMA IF NOT EXISTS {PROC_SCHEMA}"))
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

    @task
    def preprocess():
        task_id = "preprocess"
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from joblib import dump

            eng = create_engine(DATA_DB_URI)
            raw = pd.read_sql_table(RAW_TABLE, con=eng, schema=RAW_SCHEMA)

            y = raw["species"]
            X = raw.drop(columns=["species"])

            cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
            num_cols = X.select_dtypes(include=["number"]).columns.tolist()

            pre = ColumnTransformer([
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols),
            ])

            preproc = Pipeline([("prep", pre), ("scale", StandardScaler(with_mean=False))])
            Xp = preproc.fit_transform(X)

            try:
                feature_names = preproc.get_feature_names_out()
            except Exception:
                feature_names = [f"f{i}" for i in range(Xp.shape[1])]

            Xp_df = pd.DataFrame.sparse.from_spmatrix(Xp, columns=feature_names)
            Xp_df["species"] = y.values

            Xp_df.to_sql(PROC_TABLE, eng, schema=PROC_SCHEMA, if_exists="replace", index=False)
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            dump(preproc, ARTIFACTS_DIR / "preprocessor.pkl")

            return write_status(task_id, "ok", "preprocessed saved",
                                {"rows": len(Xp_df), "features": len(feature_names)})
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
