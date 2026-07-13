import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    """Create a test client inside a lifespan context manager."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def auth_headers(client):
    """Sign up and log in a test user to get JWT authorization headers."""
    # Ensure signup works
    client.post("/auth/signup", json={"username": "testuser", "password": "password123"})
    # Log in to get token
    res = client.post("/auth/login", json={"username": "testuser", "password": "password123"})
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

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

def test_predict_red_wine(client, auth_headers):
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
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "rounded_prediction" in json_data
    assert json_data["wine_type"] == "red"
    assert isinstance(json_data["prediction"], float)

def test_predict_white_wine(client, auth_headers):
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
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert json_data["wine_type"] == "white"

def test_predict_unauthorized(client):
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
    # Call without Authorization headers
    response = client.post("/predict", json=payload)
    assert response.status_code in [401, 403]
    
    # Call with invalid headers
    response = client.post("/predict", json=payload, headers={"Authorization": "Bearer badtoken"})
    assert response.status_code == 401  # Unauthorized

def test_predict_invalid_type(client, auth_headers):
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
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 400
    assert "Invalid wine type" in response.json()["detail"]

def test_predict_missing_fields(client, auth_headers):
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
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 422  # Unprocessable Entity (Validation Error)

def test_predict_batch(client, auth_headers):
    csv_data = (
        "fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,"
        "free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,type\n"
        "7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,red\n"
        "6.8,0.26,0.34,6.1,0.046,30.0,110.0,0.9945,3.18,0.47,10.2,white\n"
    )
    
    files = {"file": ("wines.csv", csv_data, "text/csv")}
    response = client.post("/predict_batch", files=files, headers=auth_headers)
    
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=predictions.csv" in response.headers["content-disposition"]
    
    # Read the returned CSV
    content = response.text
    assert "predicted_quality" in content
    assert "rounded_predicted_quality" in content
    assert "total_acidity" not in content
    
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

def test_predict_batch_invalid_extension(client, auth_headers):
    # Uploading a non-csv file
    files = {"file": ("wines.txt", "some,text,data\n1,2,3", "text/plain")}
    response = client.post("/predict_batch", files=files, headers=auth_headers)
    assert response.status_code == 400
    assert "Only CSV files are supported" in response.json()["detail"]

def test_predict_batch_missing_fields(client, auth_headers):
    # Missing alcohol and pH columns
    csv_data = (
        "fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,"
        "free_sulfur_dioxide,total_sulfur_dioxide,density,sulphates,type\n"
        "7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,0.56,red\n"
    )
    files = {"file": ("wines.csv", csv_data, "text/csv")}
    response = client.post("/predict_batch", files=files, headers=auth_headers)
    assert response.status_code == 400
    assert "Missing required columns in CSV" in response.json()["detail"]

def test_predict_batch_invalid_wine_types(client, auth_headers):
    # Containing 'beer' as type
    csv_data = (
        "fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,"
        "free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,type\n"
        "7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,beer\n"
    )
    files = {"file": ("wines.csv", csv_data, "text/csv")}
    response = client.post("/predict_batch", files=files, headers=auth_headers)
    assert response.status_code == 400
    assert "All wines must have type 'red' or 'white'" in response.json()["detail"]

def test_predict_batch_malformed_csv(client, auth_headers):
    # Completely broken CSV format
    files = {"file": ("wines.csv", "this is not,a csv,at all,,,\n\n,\n\n,", "text/csv")}
    response = client.post("/predict_batch", files=files, headers=auth_headers)
    assert response.status_code == 400

def test_predictions_history(client, auth_headers):
    # Retrieve history
    response = client.get("/predictions_history", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

