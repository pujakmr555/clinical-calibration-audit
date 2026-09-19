"""Config-driven experiment runner.
Usage: python scripts/run_experiment.py configs/smoke.yaml
Writes: results/<name>_metrics.csv, results/<name>_subgroups.csv, results/preds/ (gitignored)."""
import sys, os, json, subprocess
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.loaders import load_synthetic, load_mimic, make_splits, xy, GROUP_COLS
from src.models.registry import build
from src.eval.metrics import all_metrics
from src.eval.calibration import recalibrate
from src.eval.selective import risk_coverage_curve, aurc, metrics_at_coverage
from src.eval.subgroups import subgroup_metrics, abstention_disparity


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    name = cfg["name"]
    os.makedirs("results/preds", exist_ok=True)
    loader = {"synthetic": load_synthetic, "mimic": load_mimic}[cfg["dataset"]["kind"]]
    df = loader(**cfg["dataset"].get("params", {}))

    metric_rows, subgroup_rows, disparity_rows = [], [], []
    for seed in cfg.get("seeds", [0, 1, 2, 3, 4]):
        train, val, test = make_splits(df, seed=seed)
        Xtr, ytr = xy(train); Xva, yva = xy(val); Xte, yte = xy(test)
        for mcfg in cfg["models"]:
            model = build(mcfg["name"], seed=seed, **mcfg.get("params", {}))
            model.fit(Xtr, ytr)
            p_va = model.predict_proba(Xva)[:, 1] if hasattr(model, "predict_proba") else model.predict(Xva)
            p_te = model.predict_proba(Xte)[:, 1]
            for recal in cfg.get("recalibration", ["none", "temperature", "platt", "isotonic"]):
                p = recalibrate(recal, yva, p_va, p_te)
                row = {"model": mcfg["name"], "recal": recal, "seed": seed,
                       "git": git_hash(), **all_metrics(yte, p),
                       "aurc": aurc(risk_coverage_curve(yte, p)),
                       **metrics_at_coverage(yte, p)}
                metric_rows.append(row)
                if recal in ("none", "temperature"):
                    for gcol in GROUP_COLS:
                        sg = subgroup_metrics(yte, p, test[gcol], n_boot=cfg.get("n_boot", 500))
                        sg["group_var"], sg["model"], sg["recal"], sg["seed"] = gcol, mcfg["name"], recal, seed
                        subgroup_rows.append(sg)
                        ad = abstention_disparity(p, test[gcol], coverage=0.9)
                        ad["group_var"], ad["model"], ad["recal"], ad["seed"] = gcol, mcfg["name"], recal, seed
                        disparity_rows.append(ad)
            np.save(f"results/preds/{name}_{mcfg['name']}_s{seed}.npy",
                    np.vstack([yte, p_te]))

    pd.DataFrame(metric_rows).to_csv(f"results/{name}_metrics.csv", index=False)
    pd.concat(subgroup_rows).to_csv(f"results/{name}_subgroups.csv", index=False)
    pd.concat(disparity_rows).to_csv(f"results/{name}_disparity.csv", index=False)

    summary = (pd.DataFrame(metric_rows)
               .groupby(["model", "recal"])[["auroc", "auprc", "ece_quantile", "brier", "aurc"]]
               .agg(["mean", "std"]).round(4))
    print(summary.to_string())
    print(f"\nWrote results/{name}_*.csv")


if __name__ == "__main__":
    main(sys.argv[1])
