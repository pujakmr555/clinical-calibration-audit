# Do Clinical Prediction Models Know What They Don't Know?

### Calibration, Selective Prediction, and Subgroup Reliability for ICU Mortality

**Paper:** Submitted to [ML4H 2026](https://ml4h.ahli.cc) (Machine Learning for Health Symposium)  
**Preprint:** [arXiv:XXXX.XXXXX](https://arxiv.org) *(forthcoming)*

---

## Summary

This repository contains the code and evaluation framework for a systematic reliability audit of clinical prediction models for ICU mortality on [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) (31,142 ICU stays, 13.3% mortality). We evaluate logistic regression, XGBoost, and multilayer perceptrons across four post-hoc recalibration strategies, analyzing not only aggregate discrimination and calibration but also **subgroup-level reliability** and **selective prediction fairness**.

### Key Findings

1. **Calibration disparities across ethnic groups:** Asian and Hispanic patients experience 3–4× higher Expected Calibration Error (ECE) than White patients across all model families.
2. **Recalibration helps globally but not equitably:** Standard post-hoc methods (temperature scaling, Platt scaling) reduce aggregate ECE but do not close subgroup gaps.
3. **Selective prediction introduces demographic bias:** At 90% coverage, elderly patients (80+) are deferred to clinician review at rates ~50% higher than younger cohorts, and deferral rates vary across ethnic groups.

## Repository Structure

```
clinical-calibration-audit/
├── src/
│   ├── data/
│   │   ├── loaders.py           # Synthetic data generator for development
│   │   └── mimic_loader.py      # MIMIC-IV parquet loader
│   ├── models/
│   │   └── registry.py          # Model zoo (LR, XGBoost, TabPFN wrapper)
│   ├── eval/
│   │   ├── metrics.py           # AUROC, AUPRC, ECE, Brier score
│   │   ├── calibration.py       # Temperature, Platt, isotonic scaling
│   │   ├── selective.py         # Risk-coverage curves, AURC, deferral simulation
│   │   └── subgroups.py         # Subgroup analysis with bootstrap CIs, abstention disparity
│   └── plots/
│       └── figures.py           # Reliability diagrams, risk-coverage plots
├── scripts/
│   ├── preprocess_mimic.py      # MIMIC-IV raw CSVs → model-ready parquet
│   ├── run_mimic_experiments.py # Full experiment suite (all models × seeds × recalibration)
│   ├── run_experiment.py        # Config-driven runner for development
│   └── generate_figures.py      # Generate all paper figures from results CSVs
├── configs/
│   └── smoke.yaml               # Smoke test configuration
├── tests/
│   └── test_metrics.py          # Unit tests for evaluation metrics
├── results/                     # Aggregate results (CSVs only — no patient data)
├── paper/figures/               # Generated figures (PDF + PNG)
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run unit tests

```bash
python -m pytest tests/ -q
```

### 3. Smoke test on synthetic data

```bash
python scripts/run_experiment.py configs/smoke.yaml
```

### 4. Reproduce paper results (requires MIMIC-IV access)

**Step 1: Obtain MIMIC-IV access**
- Create a [PhysioNet](https://physionet.org) account
- Complete [CITI training](https://about.citiprogram.org/) ("Data or Specimens Only Research" via MIT affiliation)
- Apply for [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) access

**Step 2: Preprocess**
```bash
python scripts/preprocess_mimic.py \
    --mimic_dir /path/to/mimic-iv \
    --output data/mimic_cohort.parquet
```

**Step 3: Run experiments**
```bash
python scripts/run_mimic_experiments.py --data data/mimic_cohort.parquet
```

**Step 4: Generate figures**
```bash
python scripts/generate_figures.py
```

All figures are saved to `paper/figures/`.

## Results Overview

| Model | Recal | AUROC | ECE ↓ | Brier ↓ |
|-------|-------|-------|-------|---------|
| LR | None | .867 ± .009 | .015 ± .005 | .083 ± .000 |
| XGB | Temp | .884 ± .006 | **.007 ± .001** | **.079 ± .001** |
| MLP | Platt | .869 ± .009 | .011 ± .002 | .082 ± .002 |

**Subgroup ECE (uncalibrated):**

| Ethnicity | LR | XGB | MLP |
|-----------|-----|-----|-----|
| White | .017 | .013 | .023 |
| Black | .038 | .032 | .034 |
| Hispanic | .051 | .051 | .053 |
| Asian | .057 | .052 | .054 |

## Data Governance

⚠️ **MIMIC-IV data must never be committed to this repository.** The `data/` directory is gitignored. Only aggregate metrics and results are stored in `results/`. Per the MIMIC-IV Data Use Agreement, patient-level data may not be shared or redistributed.

## Evaluation Framework

The evaluation framework in `src/eval/` is modular and reusable:

- **`metrics.py`** — ECE (equal-mass and equal-width binning), Brier score, AUROC, AUPRC
- **`calibration.py`** — Temperature scaling, Platt scaling, isotonic regression (fit on validation, apply to test)
- **`selective.py`** — Risk-coverage curves, AURC, deferral simulation with configurable expert accuracy
- **`subgroups.py`** — Per-subgroup metrics with bootstrap confidence intervals, abstention disparity analysis

All metrics are unit-tested (`tests/test_metrics.py`).

## Citation

```bibtex
@inproceedings{kumari2026calibration,
  title={Do Clinical Prediction Models Know What They Don't Know? 
         Calibration, Selective Prediction, and Subgroup Reliability 
         for ICU Mortality},
  author={Kumari, Puja},
  booktitle={Machine Learning for Health (ML4H) Symposium},
  year={2026}
}
```

## License

MIT License
