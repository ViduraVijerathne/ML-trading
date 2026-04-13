from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SOL Futures Trader API")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# --------------- Model endpoints (lazy imports) ---------------

@app.post("/api/model/train")
async def train_endpoint(req: dict | None = None):
    from app.services.data_fetcher import fetch_historical_klines
    from app.services.ml_model import train_model
    try:
        interval = (req or {}).get("interval", "15m")
        limit = (req or {}).get("limit", 1500)
        df = fetch_historical_klines(interval=interval, limit=limit)
        result = train_model(df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/predict")
async def predict_endpoint():
    from app.services.data_fetcher import fetch_historical_klines
    from app.services.ml_model import predict_signal
    try:
        df = fetch_historical_klines(interval="15m", limit=200)
        return predict_signal(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/status")
async def model_status():
    from app.services.ml_model import load_model
    model, scaler = load_model()
    return {"trained": model is not None}


# --------------- Backtest endpoints ---------------

@app.post("/api/backtest/run")
async def run_backtest_endpoint(req: dict | None = None):
    from app.services.data_fetcher import fetch_historical_klines
    from app.services.backtester import run_backtest
    try:
        params = req or {}
        interval = params.get("interval", "15m")
        limit = params.get("limit", 1500)
        df = fetch_historical_klines(interval=interval, limit=limit)
        result = run_backtest(
            df,
            initial_balance=params.get("initial_balance", 10.0),
            trade_amount=params.get("trade_amount", 1.0),
            leverage=params.get("leverage", 10),
            take_profit_pct=params.get("take_profit_pct", 0.01),
            stop_loss_pct=params.get("stop_loss_pct", 0.005),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------- Trading endpoints ---------------

@app.post("/api/trading/start")
async def start_trading():
    from app.services.trader import get_trader
    try:
        trader = get_trader()
        trader.is_running = True
        trader.setup_leverage()
        return {"message": "Trading started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/stop")
async def stop_trading():
    from app.services.trader import get_trader
    try:
        trader = get_trader()
        trader.is_running = False
        return {"message": "Trading stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trading/status")
async def trading_status():
    from app.services.trader import get_trader
    try:
        trader = get_trader()
        return trader.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/execute-once")
async def execute_once():
    from app.services.trader import get_trader
    from app.services.data_fetcher import fetch_historical_klines
    from app.services.ml_model import predict_signal
    try:
        trader = get_trader()
        df = fetch_historical_klines(interval="15m", limit=200)
        signal_data = predict_signal(df)
        if signal_data["signal"] in ("LONG", "SHORT"):
            result = trader.execute_signal(signal_data)
        else:
            result = {"action": "HOLD", "reason": signal_data.get("reason", "No signal")}
        return {"message": "Signal executed", "signal": signal_data, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/close-position")
async def close_position():
    from app.services.trader import get_trader
    try:
        trader = get_trader()
        trader.close_position()
        return {"message": "Position closed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
