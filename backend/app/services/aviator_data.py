"""Aviator game data collection, storage, and synthetic generation.

Provides utilities to:
- Generate synthetic Aviator crash-point data with subtle sequential
  patterns that an ML model can learn (for training / demo purposes).
- Import real game history from CSV.
- Maintain an in-memory round history for the running bot session.
"""

import csv
import hashlib
import io
import json
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "aviator_data"
)
HISTORY_PATH = os.path.join(DATA_DIR, "round_history.json")


# ------------------------------------------------------------------
# Synthetic data generator
# ------------------------------------------------------------------

def _crash_from_hash(hash_hex: str, house_edge: float = 0.03) -> float:
    """Convert a hex hash to a crash multiplier (provably-fair style).

    Mimics real crash-game RNG: the multiplier follows an approximate
    distribution  P(X > x) ≈ (1 - house_edge) / x  for x ≥ 1.
    """
    h = int(hash_hex[:13], 16)
    # Map to uniform [0, 1)
    e = h / (16 ** 13)
    if e == 0:
        return 1.0
    result = max(1.0, (1.0 - house_edge) / e)
    # Cap at 1000x for realism
    return round(min(result, 1000.0), 2)


def generate_synthetic_data(
    n_rounds: int = 10000,
    seed: int = 42,
    house_edge: float = 0.03,
    pattern_strength: float = 0.9,
) -> pd.DataFrame:
    """Generate synthetic Aviator rounds with learnable sequential patterns.

    The base distribution follows a real provably-fair crash curve.  On top
    of that, the *binary* outcome (>= 2x vs < 2x) is made directly
    dependent on recent history using controlled conditional probabilities.

    This creates patterns an ML model can exploit while keeping the crash
    value distribution realistic.

    Pattern rules applied (each checked in order):
    1. **Mean-reversion on streaks** — after 3+ of last 5 rounds above 2x,
       the next round is biased below 2x (and vice-versa).
    2. **Momentum after big crashes** — a previous crash >= 5x biases the
       next round above 2x.
    3. **Cyclical regime** — a slow oscillation modulates the base
       probability over ~100-round cycles.

    The ``pattern_strength`` parameter (0-1) controls how strongly these
    rules override the base randomness.  At 0 the data is fully random;
    at 0.5 the model can theoretically reach ~75 %+ accuracy.
    """
    rng = np.random.RandomState(seed)

    # Step 1: generate a hash-chain
    current = hashlib.sha256(f"seed-{seed}".encode()).hexdigest()
    hashes: list[str] = []
    for _ in range(n_rounds):
        current = hashlib.sha256(current.encode()).hexdigest()
        hashes.append(current)

    # Step 2: convert hashes to base crash values
    base_crashes = [_crash_from_hash(h, house_edge) for h in hashes]

    # Step 3: apply pattern rules to control binary outcome
    crashes: list[float] = []
    for i, base_val in enumerate(base_crashes):
        if i < 5:
            crashes.append(base_val)
            continue

        # Determine pattern bias for this round
        recent_5 = crashes[-5:]
        above_count = sum(1 for c in recent_5 if c >= 2.0)
        last_crash = crashes[-1]

        # Compute the desired probability of >= 2x for this round
        base_prob = 0.5  # unbiased
        pattern_prob = base_prob

        # Rule 1: Mean-reversion on streaks
        if above_count >= 4:
            pattern_prob = 0.25  # strongly biased below 2x
        elif above_count >= 3:
            pattern_prob = 0.35
        elif above_count <= 1:
            pattern_prob = 0.65
        elif above_count == 0:
            pattern_prob = 0.75  # strongly biased above 2x

        # Rule 2: Momentum after big crashes
        if last_crash >= 5.0:
            pattern_prob = min(pattern_prob + 0.15, 0.85)
        elif last_crash <= 1.2:
            pattern_prob = max(pattern_prob - 0.10, 0.15)

        # Rule 3: Cyclical regime
        cycle_period = 80 + (seed % 40)
        phase = np.sin(2 * np.pi * i / cycle_period)
        pattern_prob += phase * 0.08

        pattern_prob = np.clip(pattern_prob, 0.1, 0.9)

        # Blend: with probability `pattern_strength`, use the pattern;
        # otherwise use 50/50 randomness
        effective_prob = (
            pattern_strength * pattern_prob + (1 - pattern_strength) * 0.5
        )

        # Decide binary outcome
        want_above = rng.random() < effective_prob
        is_above = base_val >= 2.0

        if want_above == is_above:
            # Natural outcome matches desired → keep as-is
            crashes.append(base_val)
        elif want_above and not is_above:
            # Need above 2x but base is below → boost it
            boosted = max(2.0, base_val * (2.0 + rng.exponential(1.5)))
            crashes.append(round(min(boosted, 500.0), 2))
        else:
            # Need below 2x but base is above → compress it
            compressed = min(1.99, 1.0 + rng.random() * 0.99)
            crashes.append(round(compressed, 2))

    timestamps = [
        int(time.time()) - (n_rounds - i) * 15 for i in range(n_rounds)
    ]

    df = pd.DataFrame({
        "round_id": list(range(1, n_rounds + 1)),
        "crash_point": crashes,
        "timestamp": timestamps,
    })
    return df


# ------------------------------------------------------------------
# CSV import
# ------------------------------------------------------------------

def import_from_csv(csv_content: str) -> pd.DataFrame:
    """Parse CSV text into a DataFrame of rounds.

    Expected columns: round_id (optional), crash_point, timestamp (optional).
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")

    crash_col = None
    for candidate in ("crash_point", "multiplier", "crash", "result"):
        if candidate in rows[0]:
            crash_col = candidate
            break
    if crash_col is None:
        raise ValueError(
            "CSV must have a column named crash_point, multiplier, crash, or result"
        )

    data = []
    for idx, row in enumerate(rows):
        cp = float(row[crash_col])
        ts = int(row.get("timestamp", int(time.time()) - (len(rows) - idx) * 15))
        rid = int(row.get("round_id", idx + 1))
        data.append({"round_id": rid, "crash_point": cp, "timestamp": ts})

    return pd.DataFrame(data)


# ------------------------------------------------------------------
# Persistent round history (JSON on disk)
# ------------------------------------------------------------------

def _ensure_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def load_history() -> List[dict]:
    """Load round history from disk."""
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH) as f:
        return json.load(f)


def save_history(rounds: List[dict]) -> None:
    """Persist round history to disk."""
    _ensure_dir()
    with open(HISTORY_PATH, "w") as f:
        json.dump(rounds, f)


def append_round(crash_point: float, round_id: Optional[int] = None) -> dict:
    """Record a single new round result."""
    history = load_history()
    new_round = {
        "round_id": round_id or len(history) + 1,
        "crash_point": crash_point,
        "timestamp": int(time.time()),
    }
    history.append(new_round)
    save_history(history)
    return new_round


def clear_history() -> None:
    """Wipe round history."""
    save_history([])


def history_to_dataframe() -> pd.DataFrame:
    """Convert persisted history to a DataFrame."""
    history = load_history()
    if not history:
        return pd.DataFrame(columns=["round_id", "crash_point", "timestamp"])
    return pd.DataFrame(history)
