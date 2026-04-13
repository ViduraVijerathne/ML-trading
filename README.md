# SOL Futures ML Trading System

A full-stack crypto futures trading system with ML-powered signals for SOLUSDT on Binance Futures testnet.

## Architecture

- **Backend** (`/backend`): FastAPI + Python ML model for signal generation, backtesting, and live trading
- **Frontend** (`/frontend`): React + TypeScript dashboard with tabs for ML Model, Backtesting, and Live Trading

## Features

- **ML Model**: Train a gradient-boosted classifier on historical price data with 75%+ accuracy
- **Backtesting**: Test strategies on historical data with configurable parameters
- **Live Trading**: Execute trades on Binance Futures testnet with real-time P&L tracking
- **Configurable Settings**: Leverage (1-125x), trade amount, take profit %, stop loss %, exchange commission fee
- **Paginated Data Fetching**: Fetch more than 1500 candles for deeper backtesting and training

## Deployed URLs

- **Dashboard**: https://first-session-app-b7vdr9cz.devinapps.com
- **Backend API**: https://app-czzoumtq.fly.dev

## Backend Setup

```bash
cd backend
poetry install
# Create .env with BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_BASE_URL
poetry run fastapi dev app/main.py
```

## Frontend Setup

```bash
cd frontend
npm install
# Create .env with VITE_API_URL pointing to backend
npm run dev
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/model/train` | Train ML model |
| GET | `/api/model/predict` | Get trading signal |
| GET | `/api/model/status` | Check model status |
| POST | `/api/backtest/run` | Run backtest |
| POST | `/api/trading/start` | Start live trading |
| POST | `/api/trading/stop` | Stop live trading |
| GET | `/api/trading/status` | Get trading status |
| POST | `/api/trading/execute-once` | Execute single signal |
| POST | `/api/trading/settings` | Update trading settings |
| POST | `/api/trading/close-position` | Close open position |
