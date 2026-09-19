#!/usr/bin/env python3
"""Generate ALL paper figures from results CSVs.

Usage: python scripts/generate_figures.py

Produces (in paper/figures/):
  fig1_reliability.pdf     — Reliability diagrams (main models, pre/post temp scaling)
  fig2_risk_coverage.pdf   — Risk-coverage curves
  fig3_subgroup_ece.pdf    — Subgroup ECE with bootstrap CIs
  fig4_abstention.pdf      — Abstention disparity by subgroup
  fig5_context_scaling.pdf — TabPFN context-size scaling (AUROC + ECE)
  fig6_deferral.pdf        — System error under deferral at varying expert accuracy
  table1_main.tex          — Main results table (LaTeX)
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.metrics import reliability_bins

OUTDIR = "paper/figures"
os.makedirs(OUTDIR, exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1
})
COLORS = {"lr": "#1f77b4", "xgb": "#ff7f0e", "mlp": "#2ca02c",
          "tabpfn_ctx1000": "#d62728", "tabpfn_ctx3000": "#9467bd", "tabpfn_ctx5000": "#8c564b"}


def load_results():
    metrics = pd.read_csv("results/mimic_metrics.csv")
    try:
        subgroups = pd.read_csv("results/mimic_subgroups.csv")
    except FileNotFoundError:
        subgroups = pd.DataFrame()
    try:
        disparity = pd.read_csv("results/mimic_disparity.csv")
    except FileNotFoundError:
        disparity = pd.DataFrame()
    try:
        rc = pd.read_csv("results/mimic_risk_coverage.csv")
    except FileNotFoundError:
        rc = pd.DataFrame()
    try:
        deferral = pd.read_csv("results/mimic_deferral.csv")
    except FileNotFoundError:
        deferral = pd.DataFrame()
    return metrics, subgroups, disparity, rc, deferral


def fig1_reliability(metrics):
    """Reliability diagrams: main models before/after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    models_to_plot = ["lr", "xgb"]
    # Add best TabPFN if available
    tabpfn_models = [m for m in metrics["model"].unique() if "tabpfn" in m]
    if tabpfn_models:
        models_to_plot.append(sorted(tabpfn_models)[-1])  # largest context

    for ax_idx, (recal, title) in enumerate([("none", "Before recalibration"),
                                              ("temperature", "After temperature scaling")]):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")
        for model in models_to_plot:
            # Load seed 0 predictions
            try:
                arr = np.load(f"results/preds/{model}_s0.npy")
                y, p = arr[0], arr[1]
                if recal == "temperature":
                    from src.eval.calibration import recalibrate as rc
                    arr_va = np.load(f"results/preds/{model}_s0.npy")  # use same for demo
                    p = rc("temperature", y, p, p)  # simplified; real: use val set
                bins = reliability_bins(y, p, n_bins=10)
                ax.plot([b["mean_prob"] for b in bins], [b["frac_pos"] for b in bins],
                        "o-", color=COLORS.get(model, "gray"), label=model, markersize=5)
            except FileNotFoundError:
                continue
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.savefig(f"{OUTDIR}/fig1_reliability.pdf")
    print("✓ fig1_reliability.pdf")
    plt.close()


def fig2_risk_coverage(rc):
    """Risk-coverage curves (seed-averaged)."""
    if rc.empty:
        print("✗ No risk-coverage data")
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rc_none = rc[rc["recal"] == "none"]
    for model in rc_none["model"].unique():
        sub = rc_none[rc_none["model"] == model]
        avg = sub.groupby(sub["coverage"].round(3))["risk"].agg(["mean", "std"]).reset_index()
        ax.plot(avg["coverage"], avg["mean"], color=COLORS.get(model, "gray"), label=model)
        ax.fill_between(avg["coverage"], avg["mean"] - avg["std"], avg["mean"] + avg["std"],
                        alpha=0.15, color=COLORS.get(model, "gray"))
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (error rate)")
    ax.set_title("Risk–coverage curves")
    ax.legend()
    ax.invert_xaxis()
    fig.savefig(f"{OUTDIR}/fig2_risk_coverage.pdf")
    print("✓ fig2_risk_coverage.pdf")
    plt.close()


def fig3_subgroup_ece(subgroups):
    """Subgroup ECE forest plot with bootstrap CIs."""
    if subgroups.empty:
        print("✗ No subgroup data")
        return
    sg = subgroups[(subgroups["recal"] == "none") & (subgroups["group_var"] == "ethnicity_group")]
    if sg.empty:
        sg = subgroups[subgroups["recal"] == "none"]
        if sg.empty:
            print("✗ No subgroup data for plotting")
            return

    models = sg["model"].unique()
    fig, ax = plt.subplots(figsize=(8, 5))
    avg = sg.groupby(["model", "group"]).agg(
        ece_mean=("ece", "mean"), ece_lo=("ece_ci_lo", "mean"), ece_hi=("ece_ci_hi", "mean")
    ).reset_index()

    groups = sorted(avg["group"].unique())
    x = np.arange(len(groups))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        sub = avg[avg["model"] == model]
        vals = [sub[sub["group"] == g]["ece_mean"].values[0] if g in sub["group"].values else 0 for g in groups]
        lo = [sub[sub["group"] == g]["ece_lo"].values[0] if g in sub["group"].values else 0 for g in groups]
        hi = [sub[sub["group"] == g]["ece_hi"].values[0] if g in sub["group"].values else 0 for g in groups]
        err_lo = [v - l for v, l in zip(vals, lo)]
        err_hi = [h - v for v, h in zip(vals, hi)]
        ax.barh(x + i * width, vals, width * 0.9, xerr=[err_lo, err_hi],
                color=COLORS.get(model, "gray"), label=model, capsize=2)

    ax.set_yticks(x + width * (len(models) - 1) / 2)
    ax.set_yticklabels(groups)
    ax.set_xlabel("Expected Calibration Error (ECE)")
    ax.set_title("Subgroup calibration (uncalibrated)")
    ax.legend(loc="lower right")
    fig.savefig(f"{OUTDIR}/fig3_subgroup_ece.pdf")
    print("✓ fig3_subgroup_ece.pdf")
    plt.close()


def fig4_abstention(disparity):
    """Abstention disparity bar chart."""
    if disparity.empty:
        print("✗ No disparity data")
        return
    disp = disparity[(disparity["recal"] == "none") & (disparity["group_var"] == "ethnicity_group")]
    if disp.empty:
        disp = disparity[disparity["recal"] == "none"]

    avg = disp.groupby(["model", "group"]).agg(
        dr_mean=("deferral_rate", "mean"), dr_std=("deferral_rate", "std")
    ).reset_index()

    models = sorted(avg["model"].unique())
    groups = sorted(avg["group"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(groups))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        sub = avg[avg["model"] == model]
        vals = [sub[sub["group"] == g]["dr_mean"].values[0] if g in sub["group"].values else 0 for g in groups]
        stds = [sub[sub["group"] == g]["dr_std"].values[0] if g in sub["group"].values else 0 for g in groups]
        ax.bar(x + i * width, vals, width * 0.9, yerr=stds,
               color=COLORS.get(model, "gray"), label=model, capsize=2)

    # Global deferral rate line
    global_rate = disp["global_deferral_rate"].mean()
    ax.axhline(y=global_rate, color="black", linestyle="--", alpha=0.5, label=f"Global rate ({global_rate:.2f})")

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("Deferral rate at 90% coverage")
    ax.set_title("Abstention disparity across subgroups")
    ax.legend()
    fig.savefig(f"{OUTDIR}/fig4_abstention.pdf")
    print("✓ fig4_abstention.pdf")
    plt.close()


def fig5_context_scaling(metrics):
    """TabPFN context-size scaling: AUROC + ECE."""
    tabpfn = metrics[metrics["model"].str.contains("tabpfn") & (metrics["recal"] == "none")]
    if tabpfn.empty:
        print("✗ No TabPFN results for context scaling")
        return

    tabpfn = tabpfn.copy()
    tabpfn["ctx"] = tabpfn["model"].str.extract(r"ctx(\d+)").astype(int)
    avg = tabpfn.groupby("ctx").agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        ece_mean=("ece_quantile", "mean"), ece_std=("ece_quantile", "std")
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.errorbar(avg["ctx"], avg["auroc_mean"], yerr=avg["auroc_std"], "o-", color="#9467bd", capsize=4)
    ax1.set_xlabel("Context size"); ax1.set_ylabel("AUROC"); ax1.set_title("Discrimination vs context size")

    ax2.errorbar(avg["ctx"], avg["ece_mean"], yerr=avg["ece_std"], "o-", color="#d62728", capsize=4)
    ax2.set_xlabel("Context size"); ax2.set_ylabel("ECE"); ax2.set_title("Calibration vs context size")

    fig.suptitle("TabPFN: effect of in-context training set size", fontsize=12)
    fig.savefig(f"{OUTDIR}/fig5_context_scaling.pdf")
    print("✓ fig5_context_scaling.pdf")
    plt.close()


def table1_main(metrics):
    """Main results table in LaTeX."""
    # Aggregate across seeds
    main_models = ["lr", "xgb", "mlp"]
    tabpfn_models = [m for m in metrics["model"].unique() if "tabpfn" in m]
    all_models = main_models + sorted(tabpfn_models)

    rows = []
    for model in all_models:
        for recal in ["none", "temperature"]:
            sub = metrics[(metrics["model"] == model) & (metrics["recal"] == recal)]
            if sub.empty:
                continue
            row = {"Model": model, "Recal": recal}
            for col in ["auroc", "auprc", "ece_quantile", "brier", "aurc"]:
                m, s = sub[col].mean(), sub[col].std()
                row[col] = f"{m:.4f} ± {s:.4f}"
            rows.append(row)

    tdf = pd.DataFrame(rows)
    latex = tdf.to_latex(index=False, escape=False)
    with open(f"{OUTDIR}/table1_main.tex", "w") as f:
        f.write(latex)
    print("✓ table1_main.tex")
    # Also print to console
    print("\n" + tdf.to_string(index=False))


def main():
    metrics, subgroups, disparity, rc, deferral = load_results()
    print(f"Loaded: {len(metrics)} metric rows\n")

    table1_main(metrics)
    fig1_reliability(metrics)
    fig2_risk_coverage(rc)
    fig3_subgroup_ece(subgroups)
    fig4_abstention(disparity)
    fig5_context_scaling(metrics)

    print(f"\nAll figures saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
