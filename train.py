from pathlib import Path
import json
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "used_cars.csv"
MODEL_DIR = ROOT_DIR / "models"
TARGET = "selling_price"

NUMERIC_FEATURES = ["year", "km_driven", "engine_cc"]
CATEGORICAL_FEATURES = ["make", "model", "fuel", "transmission", "owner"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert values like '35,000' or '1200 cc' to numeric values."""
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[^0-9.\\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

def load_and_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize numeric columns that may contain text.
    for col in ["year", "km_driven", "engine_cc", TARGET]:
        df[col] = clean_numeric(df[col])
    # Remove rows without the value we are trying to predict.
    df = df.dropna(subset=[TARGET])
    # Basic plausibility filters for corrupted/out-of-range values.
    df = df[df[TARGET].between(0, 20_000_000)]
    df = df[df["year"].between(1980, 2026)]
    df = df[df["km_driven"].between(0, 1_000_000)]
    # Clean categorical text; missing values are imputed later inside the pipeline.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
        df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "<NA>": np.nan, "nan": np.nan})
    return df

def build_pipeline(model):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_and_clean_data()
    X = df[FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    trained = {}
    print(f"Rows after cleaning: {len(df)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("\nModel comparison (test set)")
    print("-" * 58)
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))
        results[name] = {"mae_inr": round(mae, 2), "r2": round(r2, 4)}
        trained[name] = pipeline
        print(f"{name:20s} MAE = ₹{mae:,.0f} | R² = {r2:.4f}")

    # MAE is the primary selection metric: lower is better.
    best_name = min(results, key=lambda name: results[name]["mae_inr"])
    joblib.dump(trained[best_name], MODEL_DIR / "best_model.joblib")
    metadata = {
        "selected_model": best_name,
        "selection_metric": "MAE (lower is better)",
        "metrics": results,
        "rows_after_cleaning": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "dataset_note": "Illustrative synthetic dataset; metrics are not real-world market validation."
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("-" * 58)
    print(f"Selected: {best_name}")
    print(f"Saved model to: {MODEL_DIR / 'best_model.joblib'}")
    print(f"Saved metrics to: {MODEL_DIR / 'model_metadata.json'}")

if __name__ == "__main__":
    main()
