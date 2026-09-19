"""Model zoo. Each model: fit(X, y) and predict_proba(X)->p(y=1).
TabPFN import is optional so the repo runs on machines without torch."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def make_lr(seed=0, **kw):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=kw.get("C", 1.0), random_state=seed))


def make_xgb(seed=0, **kw):
    return XGBClassifier(
        n_estimators=kw.get("n_estimators", 400), max_depth=kw.get("max_depth", 5),
        learning_rate=kw.get("learning_rate", 0.05), subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=seed, n_jobs=-1)


class TabPFNWrapper:
    """TabPFN v2 with context subsampling (TabPFN has a max in-context train size).
    context_size and context_strategy are the paper's experimental knobs."""

    def __init__(self, context_size=5000, context_strategy="random", seed=0, **kw):
        from tabpfn import TabPFNClassifier  # optional dep: pip install tabpfn torch
        self.clf = TabPFNClassifier(random_state=seed)
        self.context_size, self.strategy, self.seed = context_size, context_strategy, seed

    def _select_context(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        if n <= self.context_size:
            return X, y
        if self.strategy == "random":
            idx = rng.choice(n, self.context_size, replace=False)
        elif self.strategy == "balanced":
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            k = self.context_size // 2
            idx = np.concatenate([
                rng.choice(pos, min(k, len(pos)), replace=len(pos) < k),
                rng.choice(neg, self.context_size - min(k, len(pos)), replace=False)])
        else:
            raise ValueError(self.strategy)
        return X[idx], y[idx]

    def fit(self, X, y):
        Xc, yc = self._select_context(np.asarray(X), np.asarray(y))
        self.clf.fit(Xc, yc)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(np.asarray(X))


MODELS = {"lr": make_lr, "xgb": make_xgb, "tabpfn": TabPFNWrapper}


def build(name, seed=0, **kw):
    return MODELS[name](seed=seed, **kw)
