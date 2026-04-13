"""Technical indicator feature engineering for the ML model."""

import pandas as pd
import ta


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator features to the dataframe."""
    df = df.copy()

    # Trend indicators
    df["sma_7"] = ta.trend.sma_indicator(df["close"], window=7)
    df["sma_25"] = ta.trend.sma_indicator(df["close"], window=25)
    df["sma_99"] = ta.trend.sma_indicator(df["close"], window=99)
    df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # RSI
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["rsi_7"] = ta.momentum.rsi(df["close"], window=7)

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ATR
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

    # ADX
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()

    # Volume indicators
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # Price-based features
    df["price_change"] = df["close"].pct_change()
    df["price_change_3"] = df["close"].pct_change(3)
    df["price_change_5"] = df["close"].pct_change(5)
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_range"] = (df["close"] - df["open"]) / df["open"]

    # Moving average crossovers
    df["sma_cross"] = (df["sma_7"] - df["sma_25"]) / df["close"]
    df["ema_cross"] = (df["ema_9"] - df["ema_21"]) / df["close"]

    # Distance from Bollinger Bands
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    return df


FEATURE_COLUMNS = [
    "sma_7", "sma_25", "sma_99", "ema_9", "ema_21",
    "macd", "macd_signal", "macd_diff",
    "rsi", "rsi_7",
    "stoch_k", "stoch_d",
    "bb_upper", "bb_lower", "bb_mid", "bb_width",
    "atr", "adx",
    "volume_ratio",
    "price_change", "price_change_3", "price_change_5",
    "high_low_range", "close_open_range",
    "sma_cross", "ema_cross", "bb_position",
]
