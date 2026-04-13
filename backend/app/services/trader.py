"""Live trading engine for Binance Futures testnet."""

import asyncio
from datetime import datetime
from typing import Optional

from app.config import SYMBOL, LEVERAGE, TRADE_AMOUNT
from app.services.data_fetcher import (
    fetch_historical_klines,
    futures_account,
    futures_change_leverage,
    futures_create_order,
    futures_position_information,
    futures_mark_price,
)
from app.services.ml_model import predict_signal


class FuturesTrader:
    def __init__(self):
        self.symbol = SYMBOL
        self.leverage = LEVERAGE
        self.trade_amount = TRADE_AMOUNT
        self.take_profit_pct = 0.01
        self.stop_loss_pct = 0.005
        self.commission_fee = 0.0004  # Binance default taker fee 0.04%
        self.is_running = False
        self.trades: list[dict] = []
        self.current_position: Optional[dict] = None
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0

    def update_settings(
        self,
        leverage: Optional[int] = None,
        trade_amount: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        commission_fee: Optional[float] = None,
    ):
        """Update trading parameters."""
        if leverage is not None:
            self.leverage = leverage
        if trade_amount is not None:
            self.trade_amount = trade_amount
        if take_profit_pct is not None:
            self.take_profit_pct = take_profit_pct
        if stop_loss_pct is not None:
            self.stop_loss_pct = stop_loss_pct
        if commission_fee is not None:
            self.commission_fee = commission_fee

    def setup_leverage(self):
        """Set leverage for the trading pair."""
        try:
            futures_change_leverage(self.symbol, self.leverage)
        except Exception as e:
            if "No need to change" not in str(e):
                raise

    def get_account_balance(self) -> float:
        """Get USDT balance from futures account."""
        try:
            account = futures_account()
            for asset in account.get("assets", []):
                if asset["asset"] == "USDT":
                    return float(asset["walletBalance"])
        except Exception:
            pass
        return 0.0

    def get_current_price(self) -> float:
        """Get current mark price."""
        try:
            ticker = futures_mark_price(self.symbol)
            return float(ticker["markPrice"])
        except Exception:
            return 0.0

    def get_open_positions(self) -> list:
        """Get open positions."""
        try:
            positions = futures_position_information(self.symbol)
            return [p for p in positions if float(p.get("positionAmt", 0)) != 0]
        except Exception:
            return []

    def place_order(self, side: str, quantity: float) -> Optional[dict]:
        """Place a market order."""
        try:
            order = futures_create_order(self.symbol, side, "MARKET", quantity)
            return order
        except Exception as e:
            return {"error": str(e)}

    def close_position(self) -> Optional[dict]:
        """Close current position."""
        positions = self.get_open_positions()
        if not positions:
            self.current_position = None
            return None

        for pos in positions:
            amt = float(pos["positionAmt"])
            if amt > 0:
                return self.place_order("SELL", abs(amt))
            elif amt < 0:
                return self.place_order("BUY", abs(amt))
        return None

    def execute_signal(self, signal_data: dict) -> dict:
        """Execute a trading signal."""
        signal = signal_data["signal"]
        confidence = signal_data["confidence"]
        price = signal_data["price"]

        # Close existing position if direction changed
        if self.current_position:
            if self.current_position["signal"] != signal:
                close_result = self.close_position()
                if close_result and "error" not in close_result:
                    exit_price = self.get_current_price()
                    entry_price = self.current_position["entry_price"]
                    if self.current_position["signal"] == "LONG":
                        pnl = (exit_price - entry_price) / entry_price * self.trade_amount * self.leverage
                    else:
                        pnl = (entry_price - exit_price) / entry_price * self.trade_amount * self.leverage

                    self.total_pnl += pnl
                    if pnl > 0:
                        self.wins += 1
                    else:
                        self.losses += 1

                    trade_record = {
                        "trade_id": len(self.trades) + 1,
                        "signal": self.current_position["signal"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": round(pnl, 4),
                        "confidence": self.current_position["confidence"],
                        "entry_time": self.current_position["entry_time"],
                        "exit_time": datetime.utcnow().isoformat(),
                        "status": "closed",
                    }
                    self.trades.append(trade_record)
                    self.current_position = None

        # Only enter if no position and confidence is sufficient
        if self.current_position is None and confidence >= 60:
            position_size = self.trade_amount * self.leverage
            quantity = position_size / price

            side = "BUY" if signal == "LONG" else "SELL"
            order_result = self.place_order(side, quantity)

            if order_result and "error" not in order_result:
                self.current_position = {
                    "signal": signal,
                    "entry_price": price,
                    "confidence": confidence,
                    "entry_time": datetime.utcnow().isoformat(),
                    "quantity": quantity,
                    "order_id": order_result.get("orderId", ""),
                }
                return {
                    "action": "OPENED",
                    "signal": signal,
                    "price": price,
                    "confidence": confidence,
                    "quantity": round(quantity, 4),
                }
            else:
                return {
                    "action": "FAILED",
                    "error": order_result.get("error", "Unknown error") if order_result else "No response",
                }

        return {"action": "HOLD", "reason": "Position exists or low confidence"}

    def get_status(self) -> dict:
        """Get current trading status."""
        balance = self.get_account_balance()
        total_trades = self.wins + self.losses

        return {
            "is_running": self.is_running,
            "balance": round(balance, 4),
            "total_pnl": round(self.total_pnl, 4),
            "total_trades": total_trades,
            "winning_trades": self.wins,
            "losing_trades": self.losses,
            "win_rate": round(self.wins / total_trades * 100, 2) if total_trades > 0 else 0,
            "current_position": self.current_position,
            "current_price": self.get_current_price(),
            "trade_history": self.trades[-20:],  # Last 20 trades
            "settings": {
                "leverage": self.leverage,
                "trade_amount": self.trade_amount,
                "take_profit_pct": self.take_profit_pct,
                "stop_loss_pct": self.stop_loss_pct,
                "commission_fee": self.commission_fee,
            },
        }


# Singleton trader instance
_trader: Optional[FuturesTrader] = None


def get_trader() -> FuturesTrader:
    global _trader
    if _trader is None:
        _trader = FuturesTrader()
    return _trader
