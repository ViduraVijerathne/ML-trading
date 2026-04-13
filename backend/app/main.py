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
            fee_rate=params.get("fee_rate", 0.0004),
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


@app.post("/api/trading/settings")
async def update_trading_settings(req: dict | None = None):
    from app.services.trader import get_trader
    try:
        params = req or {}
        trader = get_trader()
        trader.update_settings(
            leverage=params.get("leverage"),
            trade_amount=params.get("trade_amount"),
            take_profit_pct=params.get("take_profit_pct"),
            stop_loss_pct=params.get("stop_loss_pct"),
            commission_fee=params.get("commission_fee"),
        )
        # Re-apply leverage on Binance if changed
        if "leverage" in params:
            trader.setup_leverage()
        return {
            "message": "Settings updated",
            "settings": {
                "leverage": trader.leverage,
                "trade_amount": trader.trade_amount,
                "take_profit_pct": trader.take_profit_pct,
                "stop_loss_pct": trader.stop_loss_pct,
                "commission_fee": trader.commission_fee,
            },
        }
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


# --------------- Aviator Prediction Bot endpoints ---------------

@app.post("/api/aviator/generate-data")
async def aviator_generate_data(req: dict | None = None):
    """Generate synthetic Aviator game data for training."""
    from app.services.aviator_data import generate_synthetic_data, save_history
    try:
        params = req or {}
        n_rounds = params.get("n_rounds", 10000)
        seed = params.get("seed", 42)
        pattern_strength = params.get("pattern_strength", 0.9)
        df = generate_synthetic_data(
            n_rounds=n_rounds, seed=seed, pattern_strength=pattern_strength,
        )
        # Save to history
        rounds = df.to_dict("records")
        save_history(rounds)
        above_2 = int((df["crash_point"] >= 2.0).sum())
        return {
            "message": f"Generated {n_rounds} synthetic rounds",
            "total_rounds": n_rounds,
            "above_2x": above_2,
            "below_2x": n_rounds - above_2,
            "ratio_above_2x": round(above_2 / n_rounds * 100, 2),
            "mean_crash": round(float(df["crash_point"].mean()), 2),
            "median_crash": round(float(df["crash_point"].median()), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/aviator/import-csv")
async def aviator_import_csv(req: dict | None = None):
    """Import Aviator game data from CSV content."""
    from app.services.aviator_data import import_from_csv, save_history
    try:
        params = req or {}
        csv_content = params.get("csv_content", "")
        if not csv_content:
            raise ValueError("csv_content is required")
        df = import_from_csv(csv_content)
        rounds = df.to_dict("records")
        save_history(rounds)
        above_2 = int((df["crash_point"] >= 2.0).sum())
        return {
            "message": f"Imported {len(df)} rounds from CSV",
            "total_rounds": len(df),
            "above_2x": above_2,
            "below_2x": len(df) - above_2,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/aviator/train")
async def aviator_train(req: dict | None = None):
    """Train the Aviator prediction model."""
    from app.services.aviator_data import history_to_dataframe, generate_synthetic_data, save_history
    from app.services.aviator_model import train_aviator_model
    try:
        params = req or {}
        use_synthetic = params.get("use_synthetic", False)
        n_rounds = params.get("n_rounds", 10000)

        if use_synthetic:
            df = generate_synthetic_data(
                n_rounds=n_rounds,
                seed=params.get("seed", 42),
                pattern_strength=params.get("pattern_strength", 0.9),
            )
            save_history(df.to_dict("records"))
        else:
            df = history_to_dataframe()

        if len(df) < 200:
            raise ValueError(
                f"Not enough data: {len(df)} rounds (need 200+). "
                "Generate synthetic data or import CSV first."
            )
        result = train_aviator_model(df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/aviator/predict")
async def aviator_predict():
    """Get prediction for the next Aviator round."""
    from app.services.aviator_data import history_to_dataframe
    from app.services.aviator_model import predict_next_round
    try:
        df = history_to_dataframe()
        if len(df) < 60:
            raise ValueError(f"Need at least 60 rounds of history, have {len(df)}")
        return predict_next_round(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/aviator/model-status")
async def aviator_model_status():
    """Check if the Aviator model is trained."""
    from app.services.aviator_model import load_aviator_model
    weights = load_aviator_model()
    return {
        "trained": weights is not None,
        "confidence_threshold": weights.get("confidence_threshold", 0.55) * 100 if weights else None,
        "n_features": len(weights.get("features", [])) if weights else 0,
    }


@app.post("/api/aviator/bot/start")
async def aviator_bot_start():
    from app.services.aviator_bot import start_bot
    try:
        return start_bot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/aviator/bot/stop")
async def aviator_bot_stop():
    from app.services.aviator_bot import stop_bot
    try:
        return stop_bot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/aviator/bot/status")
async def aviator_bot_status():
    from app.services.aviator_bot import get_status
    try:
        return get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/aviator/bot/settings")
async def aviator_bot_settings(req: dict | None = None):
    from app.services.aviator_bot import update_bot_settings
    try:
        params = req or {}
        return update_bot_settings(**params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/aviator/bot/reset")
async def aviator_bot_reset(req: dict | None = None):
    from app.services.aviator_bot import reset_bot
    try:
        params = req or {}
        return reset_bot(initial_balance=params.get("initial_balance", 100.0))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/aviator/bot/process-round")
async def aviator_bot_process_round(req: dict | None = None):
    """Feed a new round result to the bot."""
    from app.services.aviator_bot import process_round
    try:
        params = req or {}
        crash_point = params.get("crash_point")
        if crash_point is None:
            raise ValueError("crash_point is required")
        return process_round(
            crash_point=float(crash_point),
            round_id=params.get("round_id"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/aviator/simulate")
async def aviator_simulate(req: dict | None = None):
    """Run a full simulation of the bot on historical/synthetic data."""
    from app.services.aviator_data import generate_synthetic_data, history_to_dataframe
    from app.services.aviator_bot import simulate_session
    try:
        params = req or {}
        use_synthetic = params.get("use_synthetic", True)

        if use_synthetic:
            df = generate_synthetic_data(
                n_rounds=params.get("n_rounds", 10000),
                seed=params.get("seed", 42),
                pattern_strength=params.get("pattern_strength", 0.9),
            )
        else:
            df = history_to_dataframe()
            if len(df) < 200:
                raise ValueError("Not enough history for simulation")

        result = simulate_session(
            df,
            initial_balance=params.get("initial_balance", 100.0),
            bet_amount=params.get("bet_amount", 1.0),
            cashout_target=params.get("cashout_target", 2.0),
            martingale=params.get("martingale", False),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/aviator/history")
async def aviator_history(limit: int = 100):
    """Get recent round history."""
    from app.services.aviator_data import load_history
    try:
        history = load_history()
        return {
            "total_rounds": len(history),
            "rounds": history[-limit:],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/aviator/add-round")
async def aviator_add_round(req: dict | None = None):
    """Manually add a round result."""
    from app.services.aviator_data import append_round
    try:
        params = req or {}
        crash_point = params.get("crash_point")
        if crash_point is None:
            raise ValueError("crash_point is required")
        return append_round(float(crash_point), params.get("round_id"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
