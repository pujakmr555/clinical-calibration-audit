"""Discrimination and calibration metrics. Unit-tested in tests/test_metrics.py."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc(y_true, y_prob):
    return float(roc_auc_score(y_true, y_prob))


def auprc(y_true, y_prob):
    return float(average_precision_score(y_true, y_prob))


def brier(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def _bin_edges(y_prob, n_bins, strategy):
    if strategy == "quantile":  # equal-mass bins (recommended for imbalanced clinical data)
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    return np.linspace(0.0, 1.0, n_bins + 1)  # equal-width


def reliability_bins(y_true, y_prob, n_bins=10, strategy="quantile"):
    """Per-bin (confidence, accuracy, count) for reliability diagrams and ECE."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = _bin_edges(y_prob, n_bins, strategy)
    idx = np.clip(np.searchsorted(edges, y_prob, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "mean_prob": float(y_prob[mask].mean()),
            "frac_pos": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return rows


def ece(y_true, y_prob, n_bins=10, strategy="quantile"):
    """Expected Calibration Error (binary, using predicted prob of positive class)."""
    rows = reliability_bins(y_true, y_prob, n_bins, strategy)
    n = sum(r["count"] for r in rows)
    return float(sum(r["count"] / n * abs(r["mean_prob"] - r["frac_pos"]) for r in rows))


def all_metrics(y_true, y_prob, n_bins=10):
    return {
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "ece_quantile": ece(y_true, y_prob, n_bins, "quantile"),
        "ece_width": ece(y_true, y_prob, n_bins, "uniform"),
        "prevalence": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }
