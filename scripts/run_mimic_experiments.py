#!/usr/bin/env python3
"""Run the COMPLETE experiment suite on MIMIC-IV data.

Usage: python scripts/run_mimic_experiments.py [--data data/mimic_cohort.parquet]

This single script runs everything needed for the paper:
1. Baselines (LR, XGBoost) × 5 seeds × 4 recalibration methods
2. TabPFN × context sizes × 5 seeds × 4 recalibration methods
3. Subgroup analysis with bootstrap CIs
4. Abstention disparity analysis
5. Risk-coverage curves + AURC
6. Saves all results to results/

Runtime estimate: ~30-60 min on CPU (mostly TabPFN)
"""
import sys, os, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.mimic_loader import load_mimic, make_splits, xy, GROUP_COLS, FEATURE_PREFIX
from src.eval.metrics import all_metrics
from src.eval.calibration import recalibrate
from src.eval.selective import risk_coverage_curve, aurc, metrics_at_coverage, deferral_utility
from src.eval.subgroups import subgroup_metrics, abstention_disparity

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ── Config ──
SEEDS = [0, 1, 2]
RECAL_METHODS = ["none", "temperature", "platt", "isotonic"]
EXPERT_ACCS = [0.75, 0.85, 0.95]
COVERAGES = [0.80, 0.90, 0.95]

# Try importing TabPFN — if not installed, skip those experiments
try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
    print("TabPFN available ✓")
except ImportError:
    HAS_TABPFN = False
    print("TabPFN not installed — skipping TabPFN experiments (pip install tabpfn torch)")


def build_model(name, seed, **kwargs):
    if name == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=kwargs.get("C", 1.0), random_state=seed))
    elif name == "xgb":
        return XGBClassifier(n_estimators=kwargs.get("n_estimators", 400),
                             max_depth=kwargs.get("max_depth", 5),
                             learning_rate=kwargs.get("learning_rate", 0.05),
                             subsample=0.9, colsample_bytree=0.9,
                             eval_metric="logloss", random_state=seed, n_jobs=-1)
    elif name == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                           early_stopping=True, random_state=seed))
    elif name == "tabpfn":
        ctx = kwargs.get("context_size", 3000)
        strategy = kwargs.get("strategy", "random")
        return TabPFNWrapper(context_size=ctx, strategy=strategy, seed=seed)
    raise ValueError(name)


class TabPFNWrapper:
    def __init__(self, context_size=3000, strategy="random", seed=0):
        self.clf = TabPFNClassifier(random_state=seed)
        self.context_size = context_size
        self.strategy = strategy
        self.seed = seed

    def _select_context(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        if n <= self.context_size:
            return X, y
        if self.strategy == "balanced":
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            k = self.context_size // 2
            idx = np.concatenate([
                rng.choice(pos, min(k, len(pos)), replace=len(pos) < k),
                rng.choice(neg, self.context_size - min(k, len(pos)), replace=False)])
        else:  # random
            idx = rng.choice(n, self.context_size, replace=False)
        return X[idx], y[idx]

    def fit(self, X, y):
        Xc, yc = self._select_context(np.asarray(X), np.asarray(y))
        self.clf.fit(Xc, yc)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(np.asarray(X))


def run_model(name, Xtr, ytr, Xva, yva, Xte, yte, test_df, seed, **kwargs):
    """Train one model, evaluate with all recalibrations, return result rows."""
    t0 = time.time()
    model = build_model(name, seed, **kwargs)
    model.fit(Xtr, ytr)
    p_va = model.predict_proba(Xva)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    fit_time = time.time() - t0

    label = name if "context_size" not in kwargs else f"{name}_ctx{kwargs['context_size']}"

    metric_rows, sg_rows, disp_rows, rc_rows, defer_rows = [], [], [], [], []

    for recal in RECAL_METHODS:
        p = recalibrate(recal, yva, p_va, p_te)
        m = all_metrics(yte, p)
        rc = risk_coverage_curve(yte, p)

        row = {"model": label, "recal": recal, "seed": seed, "fit_time": fit_time, **m,
               "aurc": aurc(rc), **metrics_at_coverage(yte, p, COVERAGES)}
        metric_rows.append(row)

        # Risk-coverage curve data
        for pt in rc:
            rc_rows.append({"model": label, "recal": recal, "seed": seed, **pt})

        # Deferral simulation
        for ea in EXPERT_ACCS:
            for cov in COVERAGES:
                d = deferral_utility(yte, p, ea, cov)
                defer_rows.append({"model": label, "recal": recal, "seed": seed, **d})

        # Subgroup analysis (only for none + temperature to save time)
        if recal in ("none", "temperature"):
            for gcol in GROUP_COLS:
                if gcol not in test_df.columns:
                    continue
                sg = subgroup_metrics(yte, p, test_df[gcol], n_boot=500)
                sg["group_var"] = gcol
                sg["model"] = label
                sg["recal"] = recal
                sg["seed"] = seed
                sg_rows.append(sg)

                ad = abstention_disparity(p, test_df[gcol], coverage=0.9)
                ad["group_var"] = gcol
                ad["model"] = label
                ad["recal"] = recal
                ad["seed"] = seed
                disp_rows.append(ad)

    return metric_rows, sg_rows, disp_rows, rc_rows, defer_rows, p_te


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mimic_cohort.parquet")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = load_mimic(args.data)
    print(f"Dataset: {len(df)} stays, {df['y'].mean()*100:.1f}% mortality")
    print(f"Features: {len([c for c in df.columns if c.startswith(FEATURE_PREFIX)])}")

    os.makedirs("results/preds", exist_ok=True)

    all_metrics_rows, all_sg, all_disp, all_rc, all_defer = [], [], [], [], []

    # ── Define model configs ──
    model_configs = [
        {"name": "lr"},
        {"name": "xgb", "params": {"n_estimators": 400, "max_depth": 5}},
        {"name": "mlp"},
    ]
    if HAS_TABPFN:
        for ctx in [1000, 3000, 5000]:
            model_configs.append({"name": "tabpfn", "params": {"context_size": ctx}})

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        train, val, test = make_splits(df, seed=seed)
        Xtr, ytr = xy(train)
        Xva, yva = xy(val)
        Xte, yte = xy(test)

        for mcfg in model_configs:
            name = mcfg["name"]
            params = mcfg.get("params", {})
            label = name if "context_size" not in params else f"{name}_ctx{params['context_size']}"
            print(f"\n  Training {label} (seed={seed})...")

            try:
                mr, sg, disp, rc, defer, p_te = run_model(
                    name, Xtr, ytr, Xva, yva, Xte, yte, test, seed, **params)
                all_metrics_rows.extend(mr)
                all_sg.extend(sg)
                all_disp.extend(disp)
                all_rc.extend(rc)
                all_defer.extend(defer)

                # Save predictions
                np.save(f"results/preds/{label}_s{seed}.npy", np.vstack([yte, p_te]))
                print(f"  Done: AUROC={mr[0]['auroc']:.4f}, ECE={mr[0]['ece_quantile']:.4f} (uncalibrated)")
            except Exception as e:
                print(f"  FAILED: {e}")
                continue

    # ── Save everything ──
    pd.DataFrame(all_metrics_rows).to_csv("results/mimic_metrics.csv", index=False)
    if all_sg:
        pd.concat(all_sg).to_csv("results/mimic_subgroups.csv", index=False)
    if all_disp:
        pd.concat(all_disp).to_csv("results/mimic_disparity.csv", index=False)
    pd.DataFrame(all_rc).to_csv("results/mimic_risk_coverage.csv", index=False)
    pd.DataFrame(all_defer).to_csv("results/mimic_deferral.csv", index=False)

    # ── Print summary table ──
    mdf = pd.DataFrame(all_metrics_rows)
    summary = (mdf.groupby(["model", "recal"])[["auroc", "auprc", "ece_quantile", "brier", "aurc"]]
               .agg(["mean", "std"]).round(4))
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(summary.to_string())
    print(f"\nAll results saved to results/mimic_*.csv")


if __name__ == "__main__":
    main()
