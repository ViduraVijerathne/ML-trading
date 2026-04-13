"""API endpoints for live trading."""

import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.trader import get_trader
from app.services.data_fetcher import fetch_historical_klines
from app.services.ml_model import predict_signal

router = APIRouter(prefix="/api/trading", tags=["trading"])

# Background task reference
_trading_task = None


async def trading_loop():
    """Background trading loop."""
    trader = get_trader()
    trader.is_running = True
    trader.setup_leverage()

    while trader.is_running:
        try:
            df = fetch_historical_klines(interval="15m", limit=200)
            signal_data = predict_signal(df)
            trader.execute_signal(signal_data)
        except Exception:
            pass

        await asyncio.sleep(60)  # Check every 60 seconds


@router.post("/start")
async def start_trading():
    """Start live trading."""
    global _trading_task
    trader = get_trader()

    if trader.is_running:
        return {"status": "already_running"}

    _trading_task = asyncio.create_task(trading_loop())
    return {"status": "started"}


@router.post("/stop")
async def stop_trading():
    """Stop live trading."""
    global _trading_task
    trader = get_trader()
    trader.is_running = False

    if _trading_task:
        _trading_task.cancel()
        _trading_task = None

    # Close any open position
    trader.close_position()

    return {"status": "stopped"}


@router.get("/status")
async def trading_status():
    """Get current trading status."""
    trader = get_trader()
    return trader.get_status()


@router.post("/execute-once")
async def execute_once():
    """Execute a single trade signal (manual mode)."""
    trader = get_trader()
    try:
        trader.setup_leverage()
        df = fetch_historical_klines(interval="15m", limit=200)
        signal_data = predict_signal(df)
        result = trader.execute_signal(signal_data)
        return {
            "signal": signal_data,
            "execution": result,
            "status": trader.get_status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close-position")
async def close_position():
    """Close current position."""
    trader = get_trader()
    result = trader.close_position()
    return {"result": result, "status": trader.get_status()}
