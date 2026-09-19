import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.eval.metrics import ece, brier, auroc
from src.eval.calibration import TemperatureScaler, recalibrate
from src.eval.selective import risk_coverage_curve, aurc, deferral_utility

def test_perfect_calibration_low_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 50000)
    y = (rng.uniform(size=50000) < p).astype(float)
    assert ece(y, p) < 0.01

def test_shuffled_probs_high_ece():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=20000) < 0.5).astype(float)
    p = np.where(y == 1, 0.95, 0.05)
    p_shuffled = rng.permutation(p)
    assert ece(y, p_shuffled) > 0.3

def test_brier_known_value():
    assert abs(brier([1, 0], [1.0, 0.0]) - 0.0) < 1e-9
    assert abs(brier([1, 0], [0.5, 0.5]) - 0.25) < 1e-9

def test_temperature_improves_ece():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 30000)
    y = (rng.uniform(size=30000) < 1/(1+np.exp(-z))).astype(float)
    p_over = 1/(1+np.exp(-z*3))
    p_fix = recalibrate("temperature", y[:15000], p_over[:15000], p_over[15000:])
    assert ece(y[15000:], p_fix) < ece(y[15000:], p_over[15000:])

def test_risk_coverage_confident_subset_better():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(size=5000) < p).astype(float)
    curve = risk_coverage_curve(y, p)
    full = [r for r in curve if r["coverage"] > 0.99][0]["risk"]
    low = [r for r in curve if r["coverage"] < 0.2][-1]["risk"]
    assert low < full
    assert 0 < aurc(curve) < 0.5

def test_deferral_utility_bounds():
    y = np.array([0, 1] * 500); p = np.random.default_rng(3).uniform(0, 1, 1000)
    r = deferral_utility(y, p, expert_acc=0.9, coverage=0.8)
    assert 0 <= r["system_err"] <= 1
