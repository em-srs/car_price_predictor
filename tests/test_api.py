from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "selected_model" in data
    assert "metrics" in data
    assert data["selected_model"] in data["metrics"]
    assert "mae_inr" in data["metrics"][data["selected_model"]]
    assert "r2" in data["metrics"][data["selected_model"]]

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Used Car Price Predictor" in response.text
    for field in ["make", "model", "year", "km_driven", "fuel", "transmission", "owner", "engine_cc"]:
        assert f'name="{field}"' in response.text

def test_docs_page():
    response = client.get("/docs")
    assert response.status_code == 200

def test_valid_prediction():
    payload = {
        "make": "Maruti",
        "model": "Swift",
        "year": 2021,
        "km_driven": 35000,
        "fuel": "Petrol",
        "transmission": "Manual",
        "owner": "First",
        "engine_cc": 1200
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_inr" in data
    assert isinstance(data["predicted_price_inr"], (int, float))
    assert data["predicted_price_inr"] >= 0
    assert data["currency"] == "INR"

def test_prediction_with_alternative_car():
    payload = {
        "make": "Hyundai",
        "model": "Creta",
        "year": 2019,
        "km_driven": 60000,
        "fuel": "Diesel",
        "transmission": "Automatic",
        "owner": "Second",
        "engine_cc": 1500
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price_inr"] >= 0

def test_prediction_unknown_categorical():
    payload = {
        "make": "Tesla",
        "model": "Model 3",
        "year": 2022,
        "km_driven": 10000,
        "fuel": "Electric",
        "transmission": "Automatic",
        "owner": "First",
        "engine_cc": 1000
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price_inr"] >= 0

def test_invalid_km_driven():
    payload = {
        "make": "Maruti",
        "model": "Swift",
        "year": 2021,
        "km_driven": -5000,
        "fuel": "Petrol",
        "transmission": "Manual",
        "owner": "First",
        "engine_cc": 1200
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_invalid_year_too_early():
    payload = {
        "make": "Maruti",
        "model": "Swift",
        "year": 1970,
        "km_driven": 35000,
        "fuel": "Petrol",
        "transmission": "Manual",
        "owner": "First",
        "engine_cc": 1200
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_invalid_year_future():
    payload = {
        "make": "Maruti",
        "model": "Swift",
        "year": 2035,
        "km_driven": 35000,
        "fuel": "Petrol",
        "transmission": "Manual",
        "owner": "First",
        "engine_cc": 1200
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_invalid_engine_cc_zero():
    payload = {
        "make": "Maruti",
        "model": "Swift",
        "year": 2021,
        "km_driven": 35000,
        "fuel": "Petrol",
        "transmission": "Manual",
        "owner": "First",
        "engine_cc": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_invalid_empty_string():
    payload = {
        "make": "",
        "model": "Swift",
        "year": 2021,
        "km_driven": 35000,
        "fuel": "Petrol",
        "transmission": "Manual",
        "owner": "First",
        "engine_cc": 1200
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
