"""Aviator auto-betting bot.

Manages a simulated or live betting session. The bot:
1. Receives round results.
2. Uses the ML model to predict the next round.
3. Places a bet when the model predicts >= 2x with high confidence.
4. Cashes out at 2x if the round reaches it.
5. Tracks performance (wins, losses, P&L).
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from app.services.aviator_data import (
    append_round,
    history_to_dataframe,
    load_history,
)
from app.services.aviator_model import predict_next_round


@dataclass
class BotSettings:
    bet_amount: float = 1.0
    cashout_target: float = 2.0
    max_consecutive_losses: int = 10
    stop_loss_balance: float = 0.0
    take_profit_balance: float = 0.0
    martingale: bool = False
    martingale_multiplier: float = 2.0
    max_martingale_bet: float = 16.0


@dataclass
class BotState:
    is_running: bool = False
    balance: float = 100.0
    initial_balance: float = 100.0
    total_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_skipped: int = 0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    current_bet_amount: float = 1.0
    last_prediction: Optional[dict] = None
    bet_history: list = field(default_factory=list)


# Singleton bot instance
_bot_state: Optional[BotState] = None
_bot_settings: Optional[BotSettings] = None


def get_bot_state() -> BotState:
    global _bot_state
    if _bot_state is None:
        _bot_state = BotState()
    return _bot_state


def get_bot_settings() -> BotSettings:
    global _bot_settings
    if _bot_settings is None:
        _bot_settings = BotSettings()
    return _bot_settings


def update_bot_settings(
    bet_amount: Optional[float] = None,
    cashout_target: Optional[float] = None,
    max_consecutive_losses: Optional[int] = None,
    stop_loss_balance: Optional[float] = None,
    take_profit_balance: Optional[float] = None,
    martingale: Optional[bool] = None,
    martingale_multiplier: Optional[float] = None,
    max_martingale_bet: Optional[float] = None,
    initial_balance: Optional[float] = None,
) -> dict:
    settings = get_bot_settings()
    state = get_bot_state()

    if bet_amount is not None:
        settings.bet_amount = bet_amount
        state.current_bet_amount = bet_amount
    if cashout_target is not None:
        settings.cashout_target = cashout_target
    if max_consecutive_losses is not None:
        settings.max_consecutive_losses = max_consecutive_losses
    if stop_loss_balance is not None:
        settings.stop_loss_balance = stop_loss_balance
    if take_profit_balance is not None:
        settings.take_profit_balance = take_profit_balance
    if martingale is not None:
        settings.martingale = martingale
    if martingale_multiplier is not None:
        settings.martingale_multiplier = martingale_multiplier
    if max_martingale_bet is not None:
        settings.max_martingale_bet = max_martingale_bet
    if initial_balance is not None:
        state.initial_balance = initial_balance
        state.balance = initial_balance

    return _settings_dict()


def _settings_dict() -> dict:
    s = get_bot_settings()
    return {
        "bet_amount": s.bet_amount,
        "cashout_target": s.cashout_target,
        "max_consecutive_losses": s.max_consecutive_losses,
        "stop_loss_balance": s.stop_loss_balance,
        "take_profit_balance": s.take_profit_balance,
        "martingale": s.martingale,
        "martingale_multiplier": s.martingale_multiplier,
        "max_martingale_bet": s.max_martingale_bet,
    }


def start_bot() -> dict:
    state = get_bot_state()
    state.is_running = True
    return {"message": "Aviator bot started", "status": get_status()}


def stop_bot() -> dict:
    state = get_bot_state()
    state.is_running = False
    return {"message": "Aviator bot stopped", "status": get_status()}


def reset_bot(initial_balance: float = 100.0) -> dict:
    global _bot_state
    _bot_state = BotState(
        initial_balance=initial_balance,
        balance=initial_balance,
        current_bet_amount=get_bot_settings().bet_amount,
    )
    return {"message": "Bot state reset", "status": get_status()}


def get_status() -> dict:
    state = get_bot_state()
    settings = get_bot_settings()
    win_rate = 0.0
    if state.total_bets > 0:
        win_rate = round(state.total_wins / state.total_bets * 100, 2)

    return {
        "is_running": state.is_running,
        "balance": round(state.balance, 2),
        "initial_balance": state.initial_balance,
        "total_pnl": round(state.total_pnl, 2),
        "total_bets": state.total_bets,
        "total_wins": state.total_wins,
        "total_losses": state.total_losses,
        "total_skipped": state.total_skipped,
        "win_rate": win_rate,
        "consecutive_losses": state.consecutive_losses,
        "current_bet_amount": round(state.current_bet_amount, 2),
        "last_prediction": state.last_prediction,
        "settings": _settings_dict(),
        "bet_history": state.bet_history[-50:],  # last 50 bets
    }


def process_round(crash_point: float, round_id: Optional[int] = None) -> dict:
    """Process a new round result and optionally place the next bet.

    This is the main loop of the bot:
    1. Record the round result.
    2. If the bot placed a bet on this round, resolve it.
    3. Predict the next round and decide whether to bet.

    Parameters
    ----------
    crash_point : float
        The crash multiplier for the just-completed round.
    round_id : int, optional
        Round identifier.

    Returns
    -------
    dict with round result, bet outcome (if any), and next prediction.
    """
    state = get_bot_state()
    settings = get_bot_settings()

    # Record the round
    recorded = append_round(crash_point, round_id)

    result: dict = {
        "round": recorded,
        "crash_point": crash_point,
        "bet_placed": False,
        "bet_result": None,
        "next_prediction": None,
    }

    # Resolve previous bet (if any)
    if state.last_prediction and state.last_prediction.get("bet_placed"):
        bet_amount = state.last_prediction["bet_amount"]
        target = settings.cashout_target

        if crash_point >= target:
            # WIN: cashed out at target
            payout = bet_amount * target
            profit = payout - bet_amount
            state.balance += profit
            state.total_pnl += profit
            state.total_wins += 1
            state.consecutive_losses = 0

            # Reset bet amount after win
            if settings.martingale:
                state.current_bet_amount = settings.bet_amount

            bet_result = {
                "outcome": "WIN",
                "bet_amount": bet_amount,
                "payout": round(payout, 2),
                "profit": round(profit, 2),
                "crash_point": crash_point,
                "target": target,
            }
        else:
            # LOSS: crashed before target
            state.balance -= bet_amount
            state.total_pnl -= bet_amount
            state.total_losses += 1
            state.consecutive_losses += 1

            # Martingale: double bet after loss
            if settings.martingale:
                next_bet = min(
                    state.current_bet_amount * settings.martingale_multiplier,
                    settings.max_martingale_bet,
                )
                state.current_bet_amount = next_bet

            bet_result = {
                "outcome": "LOSS",
                "bet_amount": bet_amount,
                "payout": 0,
                "profit": -bet_amount,
                "crash_point": crash_point,
                "target": target,
            }

        state.total_bets += 1
        result["bet_result"] = bet_result

        # Record in history
        state.bet_history.append({
            "round_id": recorded["round_id"],
            "timestamp": recorded["timestamp"],
            **bet_result,
            "balance_after": round(state.balance, 2),
        })

        # Clear resolved prediction so it cannot be re-resolved
        state.last_prediction = None

    # Check stop conditions
    if not state.is_running:
        result["next_prediction"] = {"should_bet": False, "reason": "Bot is stopped"}
        return result

    if settings.stop_loss_balance > 0 and state.balance <= settings.stop_loss_balance:
        state.is_running = False
        result["next_prediction"] = {"should_bet": False, "reason": "Stop-loss hit"}
        return result

    if settings.take_profit_balance > 0 and state.balance >= settings.take_profit_balance:
        state.is_running = False
        result["next_prediction"] = {"should_bet": False, "reason": "Take-profit hit"}
        return result

    if state.consecutive_losses >= settings.max_consecutive_losses:
        state.is_running = False
        result["next_prediction"] = {
            "should_bet": False,
            "reason": f"Max consecutive losses ({settings.max_consecutive_losses}) reached",
        }
        return result

    if state.balance < state.current_bet_amount:
        state.is_running = False
        result["next_prediction"] = {"should_bet": False, "reason": "Insufficient balance"}
        return result

    # Predict next round
    history_df = history_to_dataframe()
    if len(history_df) < 60:
        state.total_skipped += 1
        result["next_prediction"] = {
            "should_bet": False,
            "reason": f"Need more history ({len(history_df)}/60 rounds)",
            "predicted_class": 0,
            "prediction": "N/A",
            "confidence": 0,
            "confidence_threshold": 0,
            "probability_above_2x": 0,
            "probability_below_2x": 0,
            "last_crash_point": 0,
            "features": {},
        }
        state.last_prediction = result["next_prediction"]
        return result

    try:
        prediction = predict_next_round(history_df)
    except Exception as e:
        state.total_skipped += 1
        result["next_prediction"] = {
            "should_bet": False,
            "reason": f"Prediction error: {e}",
            "predicted_class": 0,
            "prediction": "N/A",
            "confidence": 0,
            "confidence_threshold": 0,
            "probability_above_2x": 0,
            "probability_below_2x": 0,
            "last_crash_point": 0,
            "features": {},
        }
        state.last_prediction = result["next_prediction"]
        return result

    if prediction["should_bet"]:
        prediction["bet_placed"] = True
        prediction["bet_amount"] = round(state.current_bet_amount, 2)
        result["bet_placed"] = True
    else:
        prediction["bet_placed"] = False
        state.total_skipped += 1

    state.last_prediction = prediction
    result["next_prediction"] = prediction

    return result


def simulate_session(
    df: pd.DataFrame,
    initial_balance: float = 100.0,
    bet_amount: float = 1.0,
    cashout_target: float = 2.0,
    martingale: bool = False,
) -> dict:
    """Run a full simulation of the bot on historical data.

    Trains the model on the first 75% of data, then simulates
    betting on the remaining 25%.  The simulation runs entirely
    in-memory to avoid destroying the shared round history on disk.
    """
    from app.services.aviator_model import train_aviator_model, build_features, FEATURE_COLS, predict_next_round

    # Train model without persisting to disk (avoid overwriting user's model)
    train_result = train_aviator_model(df, persist=False)
    sim_weights = train_result["weights"]

    # Prepare data for simulation — use last 25%
    split_idx = int(len(df) * 0.75)
    sim_data = df.iloc[split_idx:].reset_index(drop=True)

    # Seed history with first 60 rounds of sim data (in-memory only)
    seed_rounds = min(60, len(sim_data) - 1)
    history = []
    for i in range(seed_rounds):
        row = sim_data.iloc[i]
        history.append({
            "round_id": int(row.get("round_id", i + 1)),
            "crash_point": float(row["crash_point"]),
            "timestamp": int(row.get("timestamp", int(time.time()) - (len(sim_data) - i) * 15)),
        })

    # Simulate remaining rounds (all in-memory, no disk writes)
    balance = initial_balance
    bets = 0
    wins = 0
    losses = 0
    skipped = 0
    pnl = 0.0
    current_bet = bet_amount
    consecutive_losses = 0
    equity_curve = [{"round": 0, "balance": balance}]
    bet_log: list[dict] = []

    for i in range(seed_rounds, len(sim_data)):
        row = sim_data.iloc[i]
        crash = float(row["crash_point"])

        # Record round in history (in-memory only)
        history.append({
            "round_id": int(row.get("round_id", i + 1)),
            "crash_point": crash,
            "timestamp": int(row.get("timestamp", int(time.time()) - (len(sim_data) - i) * 15)),
        })

        # Get prediction using in-memory history
        hist_df = pd.DataFrame(history[:-1])  # exclude current round for prediction
        if len(hist_df) < 60:
            skipped += 1
            continue

        try:
            pred = predict_next_round(hist_df, weights=sim_weights)
        except Exception:
            skipped += 1
            continue

        if not pred["should_bet"]:
            skipped += 1
            equity_curve.append({"round": i - seed_rounds + 1, "balance": round(balance, 2)})
            continue

        # Place bet
        if balance < current_bet:
            break

        actual_bet = current_bet  # capture before modification

        if crash >= cashout_target:
            profit = actual_bet * (cashout_target - 1)
            balance += profit
            pnl += profit
            wins += 1
            consecutive_losses = 0
            if martingale:
                current_bet = bet_amount
            outcome = "WIN"
        else:
            balance -= actual_bet
            pnl -= actual_bet
            losses += 1
            consecutive_losses += 1
            if martingale:
                current_bet = min(current_bet * 2, bet_amount * 16)
            profit = -actual_bet
            outcome = "LOSS"

        bets += 1
        bet_log.append({
            "round": i - seed_rounds + 1,
            "crash_point": crash,
            "outcome": outcome,
            "bet_amount": round(actual_bet, 2),
            "profit": round(profit, 2),
            "balance": round(balance, 2),
            "confidence": pred["confidence"],
        })
        equity_curve.append({"round": i - seed_rounds + 1, "balance": round(balance, 2)})

    win_rate = round(wins / bets * 100, 2) if bets > 0 else 0.0

    return {
        "training": train_result,
        "simulation": {
            "initial_balance": initial_balance,
            "final_balance": round(balance, 2),
            "total_pnl": round(pnl, 2),
            "total_rounds": len(sim_data) - seed_rounds,
            "bets_placed": bets,
            "wins": wins,
            "losses": losses,
            "skipped": skipped,
            "win_rate": win_rate,
            "bet_amount": bet_amount,
            "cashout_target": cashout_target,
            "martingale": martingale,
            "equity_curve": equity_curve[-200:],  # last 200 points
            "bet_log": bet_log[-100:],  # last 100 bets
        },
    }
