import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.database import SessionLocal
from api.models import User, PredictionLog


@pytest.fixture
def client():
    """Create a test client inside a lifespan context manager."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session():
    """Create a database session for querying test databases directly."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_signup_login_flow(client, db_session):
    # 1. Signup user
    username = "auth_test_user"
    password = "secretpassword"

    # Clean user if exists
    user_db = db_session.query(User).filter(User.username == username).first()
    if user_db:
        db_session.delete(user_db)
        db_session.commit()

    signup_res = client.post(
        "/auth/signup", json={"username": username, "password": password}
    )
    assert signup_res.status_code == 200
    assert signup_res.json()["username"] == username

    # Verify user exists in database
    user_in_db = db_session.query(User).filter(User.username == username).first()
    assert user_in_db is not None
    assert user_in_db.username == username
    assert user_in_db.hashed_password != password  # Must be hashed!

    # 2. Signup username conflict
    signup_conflict_res = client.post(
        "/auth/signup", json={"username": username, "password": "different_password"}
    )
    assert signup_conflict_res.status_code == 400

    # 3. Login invalid credentials
    login_fail_res = client.post(
        "/auth/login", json={"username": username, "password": "wrong_password"}
    )
    assert login_fail_res.status_code == 401

    # 4. Login successful
    login_success_res = client.post(
        "/auth/login", json={"username": username, "password": password}
    )
    assert login_success_res.status_code == 200
    token_data = login_success_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"


def test_prediction_logging_to_db(client, db_session):
    # 1. Signup & login
    username = "db_logger_user"
    password = "pass"

    user_db = db_session.query(User).filter(User.username == username).first()
    if not user_db:
        client.post("/auth/signup", json={"username": username, "password": password})

    login_res = client.post(
        "/auth/login", json={"username": username, "password": password}
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Check count of logs before prediction
    log_count_before = db_session.query(PredictionLog).count()

    # 2. Make prediction
    payload = {
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.3,
        "citric_acid": 0.3,
        "residual_sugar": 2.0,
        "chlorides": 0.05,
        "free_sulfur_dioxide": 25.0,
        "total_sulfur_dioxide": 100.0,
        "density": 0.995,
        "pH": 3.2,
        "sulphates": 0.5,
        "alcohol": 10.0,
        "type": "red",
    }
    pred_res = client.post("/predict", json=payload, headers=headers)
    assert pred_res.status_code == 200

    # 3. Verify prediction was logged in database
    log_count_after = db_session.query(PredictionLog).count()
    assert log_count_after == log_count_before + 1

    latest_log = (
        db_session.query(PredictionLog).order_by(PredictionLog.id.desc()).first()
    )
    assert latest_log.fixed_acidity == 7.0
    assert latest_log.volatile_acidity == 0.3
    assert latest_log.predicted_quality == pred_res.json()["prediction"]
    assert latest_log.wine_type == "red"
