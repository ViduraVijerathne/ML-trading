"""Fetch historical kline data from Binance Futures testnet via REST API."""

import time
import hashlib
import hmac
from urllib.parse import urlencode

import requests
import pandas as pd

from app.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_BASE_URL, SYMBOL


def _sign(params: dict) -> str:
    query_string = urlencode(params)
    return hmac.new(
        BINANCE_SECRET_KEY.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _headers() -> dict:
    return {"X-MBX-APIKEY": BINANCE_API_KEY}


def fetch_historical_klines(
    symbol: str = SYMBOL,
    interval: str = "15m",
    limit: int = 1500,
) -> pd.DataFrame:
    """Fetch historical kline/candlestick data from Binance Futures testnet."""
    url = f"{BINANCE_BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, headers=_headers(), timeout=30)
    resp.raise_for_status()
    klines = resp.json()

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )

    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df


def futures_account() -> dict:
    url = f"{BINANCE_BASE_URL}/fapi/v2/account"
    params = {"timestamp": int(time.time() * 1000), "recvWindow": 10000}
    params["signature"] = _sign(params)
    resp = requests.get(url, params=params, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def futures_change_leverage(symbol: str, leverage: int) -> dict:
    url = f"{BINANCE_BASE_URL}/fapi/v1/leverage"
    params = {
        "symbol": symbol,
        "leverage": leverage,
        "timestamp": int(time.time() * 1000),
        "recvWindow": 10000,
    }
    params["signature"] = _sign(params)
    resp = requests.post(url, params=params, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def futures_create_order(symbol: str, side: str, order_type: str, quantity: float) -> dict:
    url = f"{BINANCE_BASE_URL}/fapi/v1/order"
    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": round(quantity, 1),
        "timestamp": int(time.time() * 1000),
        "recvWindow": 10000,
    }
    params["signature"] = _sign(params)
    resp = requests.post(url, params=params, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def futures_position_information(symbol: str) -> list:
    url = f"{BINANCE_BASE_URL}/fapi/v2/positionRisk"
    params = {
        "symbol": symbol,
        "timestamp": int(time.time() * 1000),
        "recvWindow": 10000,
    }
    params["signature"] = _sign(params)
    resp = requests.get(url, params=params, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def futures_mark_price(symbol: str) -> dict:
    url = f"{BINANCE_BASE_URL}/fapi/v1/premiumIndex"
    params = {"symbol": symbol}
    resp = requests.get(url, params=params, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()
