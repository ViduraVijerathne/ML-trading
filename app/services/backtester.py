"""Backtesting engine for the ML-enhanced trading model.

Uses the same predict_signal() pipeline as live trading so that back-test
results closely reflect real-world performance.
"""

import numpy as np
import pandas as pd
from app.services.feature_engine import add_features, FEATURE_COLUMNS
from app.services.ml_model import (
    load_model, _add_extra_features, _generate_tech_signal, _ALL_FEATURES,
    _predict_proba_lightweight,
)


def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = 10.0,
    trade_amount: float = 1.0,
    leverage: int = 10,
    take_profit_pct: float = 0.01,
    stop_loss_pct: float = 0.005,
    fee_rate: float = 0.0004,
) -> dict:
    """Run backtest on historical data using the trained ML model."""
    weights, ok = load_model()
    if weights is None:
        raise ValueError("Model not trained yet. Train the model first.")

    df = add_features(df)
    df = _add_extra_features(df)
    df = df.dropna(subset=_ALL_FEATURES).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Not enough data for backtesting")

    balance = initial_balance
    trades: list[dict] = []
    equity_curve: list[dict] = []
    wins = 0
    losses = 0
    total_pnl = 0.0

    i = 0
    while i < len(df) - 1:
        if balance < trade_amount:
            break

        row = df.iloc[i]

        # ---- Signal generation (mirrors predict_signal) ----
        tech = _generate_tech_signal(row)
        if tech == 0:
            i += 1
            continue

        X = row[_ALL_FEATURES].values.reshape(1, -1).astype(float)
        probs = _predict_proba_lightweight(weights, X)
        win_prob = float(probs[0, 1])

        if win_prob < 0.55:
            i += 1
            continue

        signal = "LONG" if tech == 1 else "SHORT"
        confidence = win_prob
        entry_price = float(row["close"])
        position_size = trade_amount * leverage
        quantity = position_size / entry_price

        entry_fee = position_size * fee_rate

        # ---- Walk forward to find exit ----
        trade_pnl = 0.0
        exit_price = entry_price
        exit_reason = "timeout"
        bars_held = 0

        for j in range(i + 1, min(i + 20, len(df))):
            high = float(df.at[j, "high"])
            low = float(df.at[j, "low"])
            close = float(df.at[j, "close"])
            bars_held = j - i

            if signal == "LONG":
                if (low - entry_price) / entry_price <= -stop_loss_pct:
                    exit_price = entry_price * (1 - stop_loss_pct)
                    exit_reason = "stop_loss"
                    trade_pnl = -stop_loss_pct * position_size
                    break
                if (high - entry_price) / entry_price >= take_profit_pct:
                    exit_price = entry_price * (1 + take_profit_pct)
                    exit_reason = "take_profit"
                    trade_pnl = take_profit_pct * position_size
                    break
                if j == min(i + 19, len(df) - 1):
                    exit_price = close
                    trade_pnl = (close - entry_price) * quantity
            else:  # SHORT
                if (high - entry_price) / entry_price >= stop_loss_pct:
                    exit_price = entry_price * (1 + stop_loss_pct)
                    exit_reason = "stop_loss"
                    trade_pnl = -stop_loss_pct * position_size
                    break
                if (entry_price - low) / entry_price >= take_profit_pct:
                    exit_price = entry_price * (1 - take_profit_pct)
                    exit_reason = "take_profit"
                    trade_pnl = take_profit_pct * position_size
                    break
                if j == min(i + 19, len(df) - 1):
                    exit_price = close
                    trade_pnl = (entry_price - close) * quantity

        exit_fee = abs(quantity * exit_price) * fee_rate
        total_fees = entry_fee + exit_fee
        net_pnl = trade_pnl - total_fees

        balance += net_pnl
        total_pnl += net_pnl

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

        trades.append({
            "trade_id": len(trades) + 1,
            "signal": signal,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "entry_time": str(row.get("open_time", "")),
            "quantity": round(quantity, 6),
            "pnl": round(net_pnl, 4),
            "fees": round(total_fees, 4),
            "exit_reason": exit_reason,
            "confidence": round(confidence * 100, 2),
            "balance_after": round(balance, 4),
        })
        equity_curve.append({"trade": len(trades), "balance": round(balance, 4)})

        i += bars_held + 1  # skip past trade

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    return {
        "initial_balance": initial_balance,
        "final_balance": round(balance, 4),
        "total_pnl": round(total_pnl, 4),
        "total_trades": total_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": round(win_rate, 2),
        "leverage": leverage,
        "trade_amount": trade_amount,
        "trades": trades,
        "equity_curve": equity_curve,
    }
