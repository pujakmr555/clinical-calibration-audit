"""Selective prediction: risk-coverage curves, AURC, deferral simulation."""
import numpy as np
from .metrics import auroc


def confidence_score(p, kind="margin"):
    """Uncertainty->confidence. 'margin': distance from 0.5 (binary MSP equivalent)."""
    p = np.asarray(p, dtype=float)
    if kind == "margin":
        return np.abs(p - 0.5)
    raise ValueError(kind)


def risk_coverage_curve(y_true, y_prob, conf=None, n_points=50, threshold=0.5):
    """Sweep coverage from 100% down; at each point keep the most-confident fraction.
    Risk = 0/1 error rate on retained set (threshold=decision cutoff)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    c = confidence_score(p) if conf is None else np.asarray(conf, dtype=float)
    order = np.argsort(-c)  # most confident first
    y, p = y[order], p[order]
    err = ((p >= threshold).astype(float) != y).astype(float)
    n = len(y)
    curve = []
    for cov in np.linspace(1.0, 0.05, n_points):
        k = max(1, int(round(cov * n)))
        curve.append({"coverage": k / n, "risk": float(err[:k].mean()),
                      "auroc": auroc(y[:k], p[:k]) if 0 < y[:k].mean() < 1 else np.nan})
    return curve


def aurc(curve):
    """Area under risk-coverage curve (lower = better selective prediction)."""
    cov = np.array([r["coverage"] for r in curve])
    risk = np.array([r["risk"] for r in curve])
    o = np.argsort(cov)
    return float(np.trapezoid(risk[o], cov[o]))


def metrics_at_coverage(y_true, y_prob, coverages=(0.8, 0.9, 0.95), conf=None):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    c = confidence_score(p) if conf is None else np.asarray(conf, dtype=float)
    out = {}
    for cov in coverages:
        thr = np.quantile(c, 1 - cov)
        keep = c >= thr
        if 0 < y[keep].mean() < 1:
            out[f"auroc@{cov}"] = auroc(y[keep], p[keep])
        out[f"err@{cov}"] = float((((p[keep] >= 0.5) != y[keep].astype(bool))).mean())
        out[f"true_cov@{cov}"] = float(keep.mean())
    return out


def deferral_utility(y_true, y_prob, expert_acc, coverage, conf=None):
    """System error when deferred cases are judged by a simulated expert
    with accuracy `expert_acc` (independent errors assumption — state in paper)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    c = confidence_score(p) if conf is None else np.asarray(conf, dtype=float)
    thr = np.quantile(c, 1 - coverage)
    keep = c >= thr
    model_err = float((((p[keep] >= 0.5) != y[keep].astype(bool))).mean()) if keep.any() else 0.0
    n_defer = int((~keep).sum())
    system_err = (keep.sum() * model_err + n_defer * (1 - expert_acc)) / len(y)
    return {"coverage": float(keep.mean()), "model_err_kept": model_err,
            "expert_acc": expert_acc, "system_err": float(system_err)}
