"""API endpoints for ML model training and prediction."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.data_fetcher import fetch_historical_klines
from app.services.ml_model import train_model, predict_signal, load_model

router = APIRouter(prefix="/api/model", tags=["model"])


class TrainRequest(BaseModel):
    interval: str = "15m"
    limit: int = 1500


@router.post("/train")
async def train_endpoint(req: TrainRequest):
    """Train the ML model on historical data."""
    try:
        df = fetch_historical_klines(interval=req.interval, limit=req.limit)
        result = train_model(df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict")
async def predict_endpoint():
    """Get current trading signal prediction."""
    try:
        df = fetch_historical_klines(interval="15m", limit=200)
        result = predict_signal(df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def model_status():
    """Check if model is trained."""
    model, scaler = load_model()
    return {
        "trained": model is not None,
    }
