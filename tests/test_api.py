import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    """Create a test client inside a lifespan context manager."""
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "Welcome" in json_data["message"]
    assert json_data["status"] == "running"

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_red_wine(client):
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "type": "red"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "rounded_prediction" in json_data
    assert json_data["wine_type"] == "red"
    assert isinstance(json_data["prediction"], float)

def test_predict_white_wine(client):
    payload = {
        "fixed_acidity": 6.8,
        "volatile_acidity": 0.26,
        "citric_acid": 0.34,
        "residual_sugar": 6.1,
        "chlorides": 0.046,
        "free_sulfur_dioxide": 30.0,
        "total_sulfur_dioxide": 110.0,
        "density": 0.9945,
        "pH": 3.18,
        "sulphates": 0.47,
        "alcohol": 10.2,
        "type": "white"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert json_data["wine_type"] == "white"

def test_predict_invalid_type(client):
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "type": "beer"  # Invalid
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid wine type" in response.json()["detail"]

def test_predict_missing_fields(client):
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        # citric_acid is missing
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "type": "red"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Validation Error)
