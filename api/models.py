import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime
from api.database import Base

class User(Base):
    """User table for API client authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

class PredictionLog(Base):
    """Prediction log table tracking all quality prediction requests."""
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    wine_type = Column(String, nullable=False)
    fixed_acidity = Column(Float, nullable=False)
    volatile_acidity = Column(Float, nullable=False)
    citric_acid = Column(Float, nullable=False)
    residual_sugar = Column(Float, nullable=False)
    chlorides = Column(Float, nullable=False)
    free_sulfur_dioxide = Column(Float, nullable=False)
    total_sulfur_dioxide = Column(Float, nullable=False)
    density = Column(Float, nullable=False)
    pH = Column(Float, nullable=False)
    sulphates = Column(Float, nullable=False)
    alcohol = Column(Float, nullable=False)
    predicted_quality = Column(Float, nullable=False)
