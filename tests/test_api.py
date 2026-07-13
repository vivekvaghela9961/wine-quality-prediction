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

def test_predict_batch(client):
    csv_data = (
        "fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,"
        "free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,type\n"
        "7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,red\n"
        "6.8,0.26,0.34,6.1,0.046,30.0,110.0,0.9945,3.18,0.47,10.2,white\n"
    )
    
    files = {"file": ("wines.csv", csv_data, "text/csv")}
    response = client.post("/predict_batch", files=files)
    
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=predictions.csv" in response.headers["content-disposition"]
    
    # Read the returned CSV
    content = response.text
    assert "predicted_quality" in content
    assert "rounded_predicted_quality" in content
    assert "total_acidity" not in content  # Engineered features should not leak to output if not requested, but output contains original columns plus predictions
    
    lines = content.strip().split("\n")
    assert len(lines) == 3  # Header + 2 data rows

def test_get_model_info(client):
    response = client.get("/model_info")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["model_name"] == "RandomForestRegressor"
    assert "parameters" in json_data
    assert "features" in json_data
    assert len(json_data["features"]) == 15

def test_get_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["MAE"] == pytest.approx(0.5115)
    assert json_data["RMSE"] == pytest.approx(0.6655)
    assert json_data["R2"] == pytest.approx(0.4109)
    assert json_data["CV_RMSE_Mean"] == pytest.approx(0.7423)

