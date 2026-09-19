"""Subgroup reliability: per-group metrics with bootstrap CIs, abstention disparity."""
import numpy as np
import pandas as pd
from .metrics import all_metrics, ece
from .selective import confidence_score


def bootstrap_ci(fn, y, p, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if 0 < y[idx].mean() < 1:
            vals.append(fn(y[idx], p[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(vals)), float(lo), float(hi)


def subgroup_metrics(y_true, y_prob, groups, min_n=50, n_boot=1000):
    """Per-group AUROC/ECE/prevalence with bootstrap CIs.
    groups: pd.Series of subgroup labels aligned with y."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    g = pd.Series(groups).reset_index(drop=True)
    rows = []
    for name, idx in g.groupby(g).groups.items():
        idx = np.asarray(idx)
        if len(idx) < min_n:
            continue
        m = all_metrics(y[idx], p[idx])
        e_mean, e_lo, e_hi = bootstrap_ci(lambda a, b: ece(a, b), y[idx], p[idx], n_boot)
        rows.append({"group": name, "n": len(idx), "prevalence": m["prevalence"],
                     "auroc": m["auroc"], "ece": m["ece_quantile"],
                     "ece_ci_lo": e_lo, "ece_ci_hi": e_hi})
    return pd.DataFrame(rows)


def abstention_disparity(y_prob, groups, coverage=0.9, conf=None):
    """At a fixed GLOBAL coverage, what fraction of each subgroup is deferred?
    This is the paper's headline fairness analysis."""
    p = np.asarray(y_prob, dtype=float)
    c = confidence_score(p) if conf is None else np.asarray(conf, dtype=float)
    thr = np.quantile(c, 1 - coverage)
    deferred = c < thr
    g = pd.Series(groups).reset_index(drop=True)
    rows = [{"group": name, "n": len(idx), "deferral_rate": float(deferred[np.asarray(idx)].mean())}
            for name, idx in g.groupby(g).groups.items()]
    df = pd.DataFrame(rows)
    df["global_deferral_rate"] = float(deferred.mean())
    df["disparity_ratio"] = df["deferral_rate"] / df["global_deferral_rate"].clip(lower=1e-9)
    return df
