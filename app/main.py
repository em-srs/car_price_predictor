from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"

app = FastAPI(
    title="Used Car Price Prediction API",
    description="Predict an estimated used-car selling price in Indian rupees.",
    version="1.0.0",
)

class CarDetails(BaseModel):
    make: str = Field(..., min_length=1, examples=["Maruti"])
    model: str = Field(..., min_length=1, examples=["Swift"])
    year: int = Field(..., ge=1980, le=2026, examples=[2021])
    km_driven: float = Field(..., ge=0, le=1_000_000, examples=[35000])
    fuel: str = Field(..., min_length=1, examples=["Petrol"])
    transmission: str = Field(..., min_length=1, examples=["Manual"])
    owner: str = Field(..., min_length=1, examples=["First"])
    engine_cc: float = Field(..., gt=0, le=10000, examples=[1200])

def load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model not found. From the project root, run: python train.py"
        )
    model = joblib.load(MODEL_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata

try:
    model_pipeline, model_metadata = load_artifacts()
except RuntimeError:
    model_pipeline, model_metadata = None, {}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_pipeline is not None}

@app.get("/model-info")
def model_info():
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not trained yet. Run python train.py.")
    return model_metadata

@app.post("/predict")
def predict(car: CarDetails):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not trained yet. Run python train.py.")
    row = pd.DataFrame([car.model_dump()])
    try:
        predicted_price = float(model_pipeline.predict(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process input: {exc}") from exc
    return {
        "predicted_price_inr": round(max(0.0, predicted_price), 2),
        "currency": "INR",
        "note": "Estimated value from a learning dataset; not a verified market quote."
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Used Car Price Predictor</title>
      <style>
        body{font-family:Arial,sans-serif;max-width:760px;margin:40px auto;padding:0 18px;color:#172033}
        h1{margin-bottom:8px}p{color:#526078;line-height:1.5}
        form{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
        label{display:grid;gap:6px;font-size:14px;font-weight:600}
        input,select,button{font:inherit;padding:10px;border:1px solid #cbd3df;border-radius:8px}
        button{background:#172033;color:white;cursor:pointer;grid-column:1/-1}
        #result{margin-top:20px;padding:14px;border-radius:8px;background:#f1f5f9;white-space:pre-wrap}
        small{color:#657187}
      </style>
    </head>
    <body>
      <h1>Used Car Price Predictor</h1>
      <p>Enter the car details to get an estimated selling price. This demo uses an illustrative learning dataset.</p>
      <form id="car-form">
        <label>Make<input name="make" value="Maruti" required></label>
        <label>Model<input name="model" value="Swift" required></label>
        <label>Year<input name="year" type="number" min="1980" max="2026" value="2021" required></label>
        <label>Kilometres driven<input name="km_driven" type="number" min="0" value="35000" required></label>
        <label>Fuel<select name="fuel"><option>Petrol</option><option>Diesel</option><option>CNG</option><option>Electric</option></select></label>
        <label>Transmission<select name="transmission"><option>Manual</option><option>Automatic</option></select></label>
        <label>Owner<select name="owner"><option>First</option><option>Second</option><option>Third</option><option>Fourth+</option></select></label>
        <label>Engine (cc)<input name="engine_cc" type="number" min="1" value="1200" required></label>
        <button type="submit">Predict price</button>
      </form>
      <div id="result" aria-live="polite">Your estimate will appear here.</div>
      <p><small>For API testing and the request schema, open <a href="/docs">/docs</a>.</small></p>
      <script>
        const form = document.getElementById('car-form');
        const result = document.getElementById('result');
        form.addEventListener('submit', async (event) => {
          event.preventDefault();
          const raw = Object.fromEntries(new FormData(form).entries());
          raw.year = Number(raw.year); raw.km_driven = Number(raw.km_driven); raw.engine_cc = Number(raw.engine_cc);
          result.textContent = 'Predicting...';
          try {
            const response = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(raw)});
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Prediction failed');
            result.textContent = 'Estimated price: ' + new Intl.NumberFormat('en-IN',{style:'currency',currency:'INR',maximumFractionDigits:0}).format(data.predicted_price_inr) + '\\n\\n' + data.note;
          } catch (error) { result.textContent = 'Error: ' + error.message; }
        });
      </script>
    </body></html>
    """
