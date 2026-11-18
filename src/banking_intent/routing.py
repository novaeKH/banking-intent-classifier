import numpy as np
import pandas as pd


def routing_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    correct = predictions["category"].eq(predictions["predicted_category"])
    rows = []
    for threshold in np.linspace(0, 1, 101):
        accepted = predictions["confidence"].ge(threshold)
        rows.append(
            {
                "threshold": threshold,
                "accepted": int(accepted.sum()),
                "coverage": accepted.mean(),
                "accuracy": correct[accepted].mean(),
                "error_rate": 1 - correct[accepted].mean(),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(curve, target_accuracy=0.95, min_accepted=100) -> float:
    eligible = curve.loc[
        (curve["accuracy"] >= target_accuracy) & (curve["accepted"] >= min_accepted)
    ]
    if eligible.empty:
        return float("inf")
    return float(
        eligible.sort_values(["coverage", "threshold"], ascending=[False, True]).iloc[
            0
        ]["threshold"]
    )


def routing_metrics(predictions, threshold) -> dict:
    accepted = predictions["confidence"].ge(threshold)
    correct = predictions["category"].eq(predictions["predicted_category"])
    return {
        "threshold": threshold if np.isfinite(threshold) else None,
        "rows": len(predictions),
        "accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": float(correct[accepted].mean())
        if accepted.any()
        else None,
        "wrongly_routed": int((accepted & ~correct).sum()),
        "handoff": int((~accepted).sum()),
    }
