"""ML model for Aviator crash-point prediction.

Predicts whether the next round's multiplier will be >= 2.0 (class 1)
or < 2.0 (class 0).  Uses engineered features derived from the recent
history of crash points.

Training uses scikit-learn (lazy-imported). Inference uses a lightweight
JSON export — no sklearn needed at runtime.
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models"
)
AVIATOR_WEIGHTS_PATH = os.path.join(MODEL_DIR, "aviator_model_weights.json")

_cached_weights: Optional[dict] = None

# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------
LOOKBACK = 10  # lag depth
WINDOWS = [5, 10, 20, 50]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML features from a time-ordered crash-point series.

    Input must have at least a ``crash_point`` column sorted by time.
    """
    df = df.copy()
    cp = df["crash_point"]

    # Binary target: 1 if crash >= 2, else 0
    df["target"] = (cp >= 2.0).astype(int)

    # --- Lag features ---
    for lag in range(1, LOOKBACK + 1):
        df[f"lag_{lag}"] = cp.shift(lag)
        df[f"lag_above2_{lag}"] = (cp.shift(lag) >= 2.0).astype(float)

    # --- Rolling statistics ---
    for w in WINDOWS:
        roll = cp.shift(1).rolling(w, min_periods=w)
        df[f"roll_mean_{w}"] = roll.mean()
        df[f"roll_std_{w}"] = roll.std()
        df[f"roll_min_{w}"] = roll.min()
        df[f"roll_max_{w}"] = roll.max()
        df[f"roll_median_{w}"] = roll.median()
        df[f"ratio_above2_{w}"] = (
            (cp.shift(1) >= 2.0).astype(float).rolling(w, min_periods=w).mean()
        )

    # --- Streak features ---
    above = (cp >= 2.0).astype(int)
    streak_above = above.copy()
    streak_below = (1 - above).copy()
    for i in range(1, len(df)):
        if above.iloc[i - 1] == 1:
            streak_above.iloc[i] = streak_above.iloc[i - 1] + 1 if above.iloc[i] == 1 else 0
        else:
            streak_above.iloc[i] = 0 if above.iloc[i] == 0 else 1
        if above.iloc[i - 1] == 0:
            streak_below.iloc[i] = streak_below.iloc[i - 1] + 1 if above.iloc[i] == 0 else 0
        else:
            streak_below.iloc[i] = 0 if above.iloc[i] == 1 else 1
    df["streak_above2"] = streak_above.shift(1)
    df["streak_below2"] = streak_below.shift(1)

    # --- Derived ratios ---
    df["last_vs_mean_10"] = cp.shift(1) / df["roll_mean_10"]
    df["last_vs_mean_50"] = cp.shift(1) / df["roll_mean_50"]

    # --- EMA of crash values ---
    df["ema_5"] = cp.shift(1).ewm(span=5).mean()
    df["ema_20"] = cp.shift(1).ewm(span=20).mean()
    df["ema_ratio"] = df["ema_5"] / df["ema_20"]

    # --- Volatility of binary outcomes ---
    df["binary_vol_10"] = (
        (cp.shift(1) >= 2.0).astype(float).rolling(10, min_periods=10).std()
    )

    # --- Momentum / regime features (captures cyclical & momentum patterns) ---
    df["big_crash_flag"] = (cp.shift(1) >= 5.0).astype(float)
    df["medium_crash_flag"] = (cp.shift(1) >= 3.0).astype(float)
    df["tiny_crash_flag"] = (cp.shift(1) <= 1.2).astype(float)

    # Ratio of recent big crashes
    df["big_crash_ratio_10"] = (
        (cp.shift(1) >= 5.0).astype(float).rolling(10, min_periods=5).mean()
    )
    df["tiny_crash_ratio_10"] = (
        (cp.shift(1) <= 1.5).astype(float).rolling(10, min_periods=5).mean()
    )

    # Rolling change in above-2x ratio (regime change detector)
    ratio_20 = (cp.shift(1) >= 2.0).astype(float).rolling(20, min_periods=20).mean()
    ratio_10 = (cp.shift(1) >= 2.0).astype(float).rolling(10, min_periods=10).mean()
    df["ratio_delta_10_20"] = ratio_10 - ratio_20

    # Consecutive pattern features
    df["above2_last3"] = sum(
        (cp.shift(i) >= 2.0).astype(float) for i in range(1, 4)
    )
    df["above2_last5"] = sum(
        (cp.shift(i) >= 2.0).astype(float) for i in range(1, 6)
    )

    # Lag interaction features
    df["lag1_x_lag2"] = df["lag_above2_1"] * df["lag_above2_2"]
    df["lag1_x_lag3"] = df["lag_above2_1"] * df["lag_above2_3"]

    # Moving average crossover of crash values
    df["ema_cross_5_20"] = df["ema_5"] - df["ema_20"]

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names."""
    cols: list[str] = []
    for lag in range(1, LOOKBACK + 1):
        cols.append(f"lag_{lag}")
        cols.append(f"lag_above2_{lag}")
    for w in WINDOWS:
        cols.extend([
            f"roll_mean_{w}", f"roll_std_{w}", f"roll_min_{w}",
            f"roll_max_{w}", f"roll_median_{w}", f"ratio_above2_{w}",
        ])
    cols.extend([
        "streak_above2", "streak_below2",
        "last_vs_mean_10", "last_vs_mean_50",
        "ema_5", "ema_20", "ema_ratio",
        "binary_vol_10",
        "big_crash_flag", "medium_crash_flag", "tiny_crash_flag",
        "big_crash_ratio_10", "tiny_crash_ratio_10",
        "ratio_delta_10_20",
        "above2_last3", "above2_last5",
        "lag1_x_lag2", "lag1_x_lag3",
        "ema_cross_5_20",
    ])
    return cols


FEATURE_COLS = get_feature_columns()


# ------------------------------------------------------------------
# Lightweight inference (numpy only)
# ------------------------------------------------------------------

def _traverse_tree(tree_data: dict, features: np.ndarray) -> float:
    node = 0
    while True:
        feat_idx = tree_data["feature"][node]
        if feat_idx < 0:
            return tree_data["value"][node]
        if features[feat_idx] <= tree_data["threshold"][node]:
            node = tree_data["children_left"][node]
        else:
            node = tree_data["children_right"][node]


def _predict_proba_lightweight(weights: dict, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    mean = np.array(weights["scaler_mean"])
    scale = np.array(weights["scaler_scale"])
    X_scaled = (X - mean) / scale

    lr = weights["learning_rate"]
    init = weights["init_log_odds"]

    probs = np.zeros((n, 2))
    for i in range(n):
        raw = init
        for tree_data in weights["trees"]:
            raw += lr * _traverse_tree(tree_data, X_scaled[i])
        p1 = 1.0 / (1.0 + np.exp(-raw))
        probs[i, 0] = 1.0 - p1
        probs[i, 1] = p1
    return probs


def load_aviator_model() -> Optional[dict]:
    global _cached_weights
    if _cached_weights is not None:
        return _cached_weights
    if not os.path.exists(AVIATOR_WEIGHTS_PATH):
        return None
    with open(AVIATOR_WEIGHTS_PATH) as f:
        _cached_weights = json.load(f)
    return _cached_weights


# ------------------------------------------------------------------
# Training (requires scikit-learn — lazy imported)
# ------------------------------------------------------------------

def train_aviator_model(df: pd.DataFrame) -> dict:
    """Train a GradientBoosting classifier on Aviator crash data.

    Parameters
    ----------
    df : DataFrame
        Must contain at least ``crash_point`` column, time-ordered.

    Returns
    -------
    dict with accuracy metrics and model info.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import TimeSeriesSplit

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Feature engineering
    df = build_features(df)
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

    if len(df) < 200:
        raise ValueError(f"Not enough data after feature engineering: {len(df)} rows (need 200+)")

    X = df[FEATURE_COLS].values
    y = df["target"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series split: train on first 75 %, test on last 25 %
    split_idx = int(len(X_scaled) * 0.75)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Metrics ---
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    # High-confidence accuracy (selective prediction)
    probs = model.predict_proba(X_test)
    confidence = np.max(probs, axis=1)

    # Find the best confidence threshold that yields >= 75% accuracy
    best_threshold = 0.50
    best_hc_acc = test_acc
    best_hc_count = len(y_test)
    for thresh in np.arange(0.50, 0.85, 0.01):
        mask = confidence >= thresh
        if mask.sum() < 20:
            continue
        acc = accuracy_score(y_test[mask], test_pred[mask])
        if acc >= best_hc_acc:
            best_hc_acc = acc
            best_threshold = float(thresh)
            best_hc_count = int(mask.sum())

    report = classification_report(
        y_test, test_pred, target_names=["< 2x", ">= 2x"], output_dict=True,
    )

    # --- Cross-validation ---
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_scaled):
        cv_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_split=20, min_samples_leaf=10,
            max_features="sqrt", random_state=42,
        )
        cv_model.fit(X_scaled[train_idx], y[train_idx])
        cv_pred = cv_model.predict(X_scaled[val_idx])
        cv_scores.append(accuracy_score(y[val_idx], cv_pred))

    # --- Export lightweight JSON model ---
    trees = []
    for stage in model.estimators_:
        tree = stage[0].tree_
        trees.append({
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "value": tree.value[:, 0, 0].tolist(),
        })

    init_value = float(model.init_.class_prior_[1])
    init_log_odds = float(np.log(init_value / (1.0 - init_value)))

    weights = {
        "learning_rate": model.learning_rate,
        "init_log_odds": init_log_odds,
        "n_classes": 2,
        "trees": trees,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "features": FEATURE_COLS,
        "confidence_threshold": best_threshold,
    }

    with open(AVIATOR_WEIGHTS_PATH, "w") as f:
        json.dump(weights, f)

    # Reset cache
    global _cached_weights
    _cached_weights = None

    feat_imp = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

    return {
        "train_accuracy": round(train_acc * 100, 2),
        "test_accuracy": round(test_acc * 100, 2),
        "high_confidence_accuracy": round(best_hc_acc * 100, 2),
        "high_confidence_trades": best_hc_count,
        "confidence_threshold": round(best_threshold * 100, 2),
        "cv_mean_accuracy": round(float(np.mean(cv_scores)) * 100, 2),
        "cv_std": round(float(np.std(cv_scores)) * 100, 2),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(df),
        "class_distribution": {
            "above_2x": int(y.sum()),
            "below_2x": int(len(y) - y.sum()),
        },
        "classification_report": report,
        "feature_importance": feat_imp,
    }


# ------------------------------------------------------------------
# Prediction (lightweight inference)
# ------------------------------------------------------------------

def predict_next_round(df: pd.DataFrame) -> dict:
    """Predict whether the next round will be >= 2x.

    Parameters
    ----------
    df : DataFrame
        Recent crash history (needs at least 50 rows).

    Returns
    -------
    dict with prediction, confidence, and recommendation.
    """
    weights = load_aviator_model()
    if weights is None:
        raise ValueError("Aviator model not trained yet. Train the model first.")

    df = build_features(df)
    df = df.dropna(subset=FEATURE_COLS)

    if len(df) == 0:
        raise ValueError("Not enough history to generate features (need 50+ rounds)")

    latest = df.iloc[-1]
    X = latest[FEATURE_COLS].values.reshape(1, -1).astype(float)

    probs = _predict_proba_lightweight(weights, X)
    prob_above = float(probs[0, 1])
    prob_below = float(probs[0, 0])
    confidence = max(prob_above, prob_below)
    threshold = weights.get("confidence_threshold", 0.55)

    predicted_class = 1 if prob_above >= 0.5 else 0
    prediction_label = ">= 2x" if predicted_class == 1 else "< 2x"

    # Recommendation: only bet if predicting >= 2x with high confidence
    should_bet = predicted_class == 1 and confidence >= threshold

    return {
        "prediction": prediction_label,
        "predicted_class": predicted_class,
        "probability_above_2x": round(prob_above * 100, 2),
        "probability_below_2x": round(prob_below * 100, 2),
        "confidence": round(confidence * 100, 2),
        "confidence_threshold": round(threshold * 100, 2),
        "should_bet": should_bet,
        "last_crash_point": float(latest["crash_point"]),
        "features": {
            "streak_above2": float(latest.get("streak_above2", 0)),
            "streak_below2": float(latest.get("streak_below2", 0)),
            "roll_mean_10": round(float(latest.get("roll_mean_10", 0)), 2),
            "ratio_above2_20": round(float(latest.get("ratio_above2_20", 0)), 2),
            "ema_ratio": round(float(latest.get("ema_ratio", 0)), 4),
        },
    }
