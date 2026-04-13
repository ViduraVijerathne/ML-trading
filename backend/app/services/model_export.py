"""Export sklearn GradientBoosting model to a lightweight JSON format.

This allows inference without importing scikit-learn, dramatically
reducing memory usage for deployment.
"""

import json
import os
import numpy as np


def export_model_to_json(model, scaler, features: list[str], output_dir: str):
    """Export a trained GradientBoostingClassifier + StandardScaler to JSON.

    The exported format stores every decision tree as arrays of node
    attributes (feature, threshold, children_left, children_right, value)
    plus the scaler's mean_ and scale_ vectors.
    """
    trees = []
    for stage in model.estimators_:
        tree = stage[0].tree_
        trees.append({
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "value": tree.value[:, 0, 0].tolist(),
            "n_nodes": int(tree.node_count),
        })

    # Get initial prediction (prior log-odds)
    init_value = float(model.init_.class_prior_[1])
    init_log_odds = float(np.log(init_value / (1 - init_value)))

    data = {
        "learning_rate": model.learning_rate,
        "init_log_odds": init_log_odds,
        "n_classes": 2,
        "trees": trees,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "features": features,
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_weights.json")
    with open(path, "w") as f:
        json.dump(data, f)

    return path


def load_lightweight_model(model_dir: str):
    """Load the exported JSON model weights."""
    path = os.path.join(model_dir, "model_weights.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _traverse_tree(tree_data: dict, features: np.ndarray) -> float:
    """Traverse a single decision tree and return the leaf value."""
    node = 0
    while True:
        feat_idx = tree_data["feature"][node]
        if feat_idx < 0:  # leaf node (TREE_UNDEFINED = -2)
            return tree_data["value"][node]
        if features[feat_idx] <= tree_data["threshold"][node]:
            node = tree_data["children_left"][node]
        else:
            node = tree_data["children_right"][node]


def predict_proba_lightweight(model_data: dict, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities using exported model weights.

    Replicates GradientBoostingClassifier.predict_proba() using only numpy.
    """
    n_samples = X.shape[0]
    # Scale features
    mean = np.array(model_data["scaler_mean"])
    scale = np.array(model_data["scaler_scale"])
    X_scaled = (X - mean) / scale

    lr = model_data["learning_rate"]
    init = model_data["init_log_odds"]

    probs = np.zeros((n_samples, 2))
    for i in range(n_samples):
        raw = init
        for tree_data in model_data["trees"]:
            raw += lr * _traverse_tree(tree_data, X_scaled[i])
        # sigmoid
        p1 = 1.0 / (1.0 + np.exp(-raw))
        probs[i, 0] = 1.0 - p1
        probs[i, 1] = p1

    return probs
