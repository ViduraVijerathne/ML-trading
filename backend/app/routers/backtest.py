"""API endpoints for backtesting."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.data_fetcher import fetch_historical_klines
from app.services.backtester import run_backtest

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    interval: str = "15m"
    limit: int = 1500
    initial_balance: float = 10.0
    trade_amount: float = 1.0
    leverage: int = 10
    take_profit_pct: float = 0.01
    stop_loss_pct: float = 0.005


@router.post("/run")
async def run_backtest_endpoint(req: BacktestRequest):
    """Run backtest with specified parameters."""
    try:
        df = fetch_historical_klines(interval=req.interval, limit=req.limit)
        result = run_backtest(
            df=df,
            initial_balance=req.initial_balance,
            trade_amount=req.trade_amount,
            leverage=req.leverage,
            take_profit_pct=req.take_profit_pct,
            stop_loss_pct=req.stop_loss_pct,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
