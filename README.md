# Used Car Price Prediction

A beginner-friendly end-to-end machine learning project that predicts the estimated selling price of a used car from its details.

🚀 **Live Demo:** [https://car-price-predictor-hkae.onrender.com/](https://car-price-predictor-hkae.onrender.com/)  
📖 **Live API Docs:** [https://car-price-predictor-hkae.onrender.com/docs](https://car-price-predictor-hkae.onrender.com/docs)

## What this project does

1. Loads the included CSV dataset.
2. Cleans missing and text-formatted numeric values with pandas.
3. Trains **Linear Regression** and **Random Forest Regressor**.
4. Compares both models using **MAE** and **R²** on a held-out test set.
5. Saves the better model pipeline with `joblib`.
6. Serves predictions through a FastAPI `/predict` endpoint.
7. Includes a small browser form at `/` and API documentation at `/docs`.

> **Dataset note:** `data/used_cars.csv` is a reproducible, illustrative dataset generated for this learning project. It is not a verified market-price feed. Model scores demonstrate the workflow, not real-world pricing accuracy.

## Project structure

```text
used-car-price-prediction/
├── app/
│   ├── __init__.py
│   └── main.py
├── data/
│   └── used_cars.csv
├── models/                 # trained artifacts created by train.py
├── tests/
│   └── test_api.py
├── train.py
├── requirements.txt
├── render.yaml
├── .gitignore
└── README.md
```

## 1. Create and activate a virtual environment

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train and compare the models

Run this from the project root:

```bash
python train.py
```

The script prints MAE and R² for both models, selects the model with the lower MAE, and saves:
- `models/best_model.joblib`
- `models/model_metadata.json`

A lower MAE means predictions are closer to actual prices on average. R² describes how much variation in the test prices is explained by the model; closer to 1 is generally better. Neither score guarantees accurate predictions for every individual car.

## 4. Run the API locally

Train first, then run:

```bash
uvicorn app.main:app --reload
```

Open:
- Home form: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"make\":\"Maruti\",\"model\":\"Swift\",\"year\":2021,\"km_driven\":35000,\"fuel\":\"Petrol\",\"transmission\":\"Manual\",\"owner\":\"First\",\"engine_cc\":1200}"
```

For Linux/macOS, use `\` instead of `^` for multiline commands, or submit the same JSON in `/docs`.

Example JSON:

```json
{
  "make": "Maruti",
  "model": "Swift",
  "year": 2021,
  "km_driven": 35000,
  "fuel": "Petrol",
  "transmission": "Manual",
  "owner": "First",
  "engine_cc": 1200
}
```

The response contains an estimated price in INR. It is a model estimate, not a dealership quote.

## 5. Deploy on Render

The project is deployed live on Render:
- **Live Web App:** [https://car-price-predictor-hkae.onrender.com/](https://car-price-predictor-hkae.onrender.com/)
- **Live Swagger Docs:** [https://car-price-predictor-hkae.onrender.com/docs](https://car-price-predictor-hkae.onrender.com/docs)
- **Live Health Check:** [https://car-price-predictor-hkae.onrender.com/health](https://car-price-predictor-hkae.onrender.com/health)

### Deployment settings reference:
1. Push this folder to a GitHub repository.
2. Sign in to Render and create a **New Web Service** connected to that repository (or public Git URL).
3. Set **Build Command** to `pip install -r requirements.txt && python train.py`.
4. Set **Start Command** to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
5. Choose Python runtime 3.11+ and deploy.

`render.yaml` is included as a reference for a Render Blueprint deployment. On the free tier, service availability and sleep behaviour depend on Render's current plan.

## 6. Run tests

```bash
pytest -q
```

Tests assume that you have run `python train.py` first.

## How the important parts work

- **Train/test split:** training data teaches the model; test data is kept separate to estimate performance on unseen examples.
- **Preprocessing pipeline:** numeric columns get median imputation; categorical columns get most-frequent imputation and one-hot encoding. This avoids manually converting category names into arbitrary numbers.
- **Linear Regression:** learns a linear relationship between input features and price.
- **Random Forest:** combines many decision trees and can learn non-linear patterns.
- **No data leakage:** preprocessing is fitted on training data only, because it is inside the scikit-learn pipeline.
- **Model serving:** FastAPI validates incoming JSON, creates a one-row DataFrame, and calls the saved preprocessing-plus-model pipeline.

## Interview questions to practise

1. **What data problems did you handle?** Some numeric values are stored as text, and a few cells are missing. The training script converts numeric text into numbers and uses median/mode imputation for remaining missing values.
2. **Linear Regression vs Random Forest?** Linear Regression models a straight-line relationship. Random Forest averages predictions from many decision trees and can learn more complex patterns.
3. **How did you choose the better model?** I compared MAE on the same test set and selected the model with the lower MAE. I also reported R².
4. **Why split training and test data?** To check how the model performs on data it did not learn from.
5. **How is the model connected to the API?** The trained pipeline is saved with joblib, loaded when the FastAPI app starts, and called by `/predict`.

## Important limitations

- The included dataset is synthetic, so it is for learning and demonstrating an end-to-end workflow.
- Real prices depend on location, condition, accident history, service records, trim and market timing, which are not represented here.
- Do not present the generated dataset or its metrics as proof of real-world commercial accuracy.
