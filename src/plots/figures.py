"""Paper figures — every figure regenerates from results/ CSVs and preds/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eval.metrics import reliability_bins
from src.eval.selective import risk_coverage_curve


def reliability_diagram(y, p, label, ax=None, n_bins=10):
    ax = ax or plt.gca()
    rows = reliability_bins(y, p, n_bins)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.plot([r["mean_prob"] for r in rows], [r["frac_pos"] for r in rows],
            "o-", label=label)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.legend(); ax.set_title("Reliability diagram")
    return ax


def risk_coverage_plot(series, ax=None):
    """series: list of (label, y, p)."""
    ax = ax or plt.gca()
    for label, y, p in series:
        c = risk_coverage_curve(y, p)
        ax.plot([r["coverage"] for r in c], [r["risk"] for r in c], label=label)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (error)")
    ax.legend(); ax.set_title("Risk\u2013coverage")
    return ax


if __name__ == "__main__":
    # demo: regenerate figures from saved predictions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    series = []
    for f in sorted(os.listdir("results/preds"))[:3]:
        arr = np.load(f"results/preds/{f}")
        y, p = arr[0], arr[1]
        reliability_diagram(y, p, f.split("_s")[0].replace("smoke_", ""), axes[0])
        series.append((f.split("_s")[0].replace("smoke_", ""), y, p))
    risk_coverage_plot(series, axes[1])
    os.makedirs("paper/figures", exist_ok=True)
    fig.tight_layout(); fig.savefig("paper/figures/demo.png", dpi=150)
    print("wrote paper/figures/demo.png")
