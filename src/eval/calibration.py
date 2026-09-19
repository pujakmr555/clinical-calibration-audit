"""Post-hoc recalibration: temperature scaling, Platt scaling, isotonic regression.
Fit on VALIDATION predictions only; apply to test predictions."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

EPS = 1e-7


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureScaler:
    """Single-parameter temperature scaling on logits (Guo et al., 2017)."""

    def __init__(self):
        self.T = 1.0

    def fit(self, y_val, p_val):
        z = _logit(np.asarray(p_val, dtype=float))
        y = np.asarray(y_val, dtype=float)

        def nll(T):
            p = np.clip(_sigmoid(z / T), EPS, 1 - EPS)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        self.T = float(res.x)
        return self

    def transform(self, p):
        return _sigmoid(_logit(np.asarray(p, dtype=float)) / self.T)


class PlattScaler:
    def __init__(self):
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, y_val, p_val):
        self.lr.fit(_logit(np.asarray(p_val)).reshape(-1, 1), np.asarray(y_val))
        return self

    def transform(self, p):
        return self.lr.predict_proba(_logit(np.asarray(p)).reshape(-1, 1))[:, 1]


class IsotonicScaler:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, y_val, p_val):
        self.iso.fit(np.asarray(p_val, dtype=float), np.asarray(y_val, dtype=float))
        return self

    def transform(self, p):
        return self.iso.predict(np.asarray(p, dtype=float))


RECALIBRATORS = {"none": None, "temperature": TemperatureScaler,
                 "platt": PlattScaler, "isotonic": IsotonicScaler}


def recalibrate(name, y_val, p_val, p_test):
    if name == "none":
        return np.asarray(p_test, dtype=float)
    scaler = RECALIBRATORS[name]().fit(y_val, p_val)
    return scaler.transform(p_test)
