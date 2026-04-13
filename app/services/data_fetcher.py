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


def _parse_klines(klines: list) -> pd.DataFrame:
    """Convert raw kline data to a DataFrame."""
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


def fetch_historical_klines(
    symbol: str = SYMBOL,
    interval: str = "15m",
    limit: int = 1500,
) -> pd.DataFrame:
    """Fetch historical kline/candlestick data from Binance Futures testnet.

    Supports fetching more than 1500 candles by paginating backwards
    through the API automatically.
    """
    url = f"{BINANCE_BASE_URL}/fapi/v1/klines"
    max_per_request = 1500

    if limit <= max_per_request:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(url, params=params, headers=_headers(), timeout=30)
        resp.raise_for_status()
        return _parse_klines(resp.json())

    # Paginate backwards to collect more than 1500 candles
    all_klines: list = []
    remaining = limit
    end_time: int | None = None  # None = latest

    while remaining > 0:
        batch_size = min(remaining, max_per_request)
        params: dict = {"symbol": symbol, "interval": interval, "limit": batch_size}
        if end_time is not None:
            params["endTime"] = end_time

        resp = requests.get(url, params=params, headers=_headers(), timeout=30)
        resp.raise_for_status()
        klines = resp.json()

        if not klines:
            break

        all_klines = klines + all_klines  # prepend older candles
        remaining -= len(klines)

        if len(klines) < batch_size:
            break  # no more data available

        # Next batch ends just before the oldest candle we received
        end_time = int(klines[0][0]) - 1

        # Small delay to avoid rate limiting
        time.sleep(0.2)

    return _parse_klines(all_klines)


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
