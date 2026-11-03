"""Tier-2 diagnosis: multinomial logistic regression on the evidence features.

Protocol (specification section 12): trained on *training sites* and *training
attack families*, tested on held-out sites and held-out families; kept only if
the ablation shows a gain over the rule table.  The implementation therefore
exposes only ``fit`` and ``predict_proba`` and never sees a test period: the
caller is responsible for the split, and the evaluation runner enforces
leave-one-site-out.

Reported with calibration (reliability curve and Brier score per class), because
a probabilistic diagnosis whose probabilities are not calibrated is worse than a
three-level confidence label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .evidence import CLASSES


@dataclass
class LRDiagnoser:
    model: LogisticRegression | None = None
    scaler: StandardScaler | None = None
    classes: tuple[str, ...] = CLASSES

    def fit(self, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> "LRDiagnoser":
        self.scaler = StandardScaler().fit(X)
        self.model = LogisticRegression(max_iter=2000, C=C, multi_class="multinomial",
                                        class_weight="balanced", random_state=0)
        self.model.fit(self.scaler.transform(X), y)
        self.classes = tuple(self.model.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("not fitted")
        return self.model.predict_proba(self.scaler.transform(np.atleast_2d(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[int(np.argmax(self.predict_proba(X), axis=1)[0])] if np.ndim(X) == 1 \
            else np.array([self.classes[i] for i in np.argmax(self.predict_proba(X), axis=1)])

    def prob_dict(self, x: np.ndarray) -> dict[str, float]:
        p = self.predict_proba(x)[0]
        return {c: float(v) for c, v in zip(self.classes, p)}

    def describe(self) -> dict:
        return {"type": "multinomial_logistic_regression", "classes": list(self.classes),
                "coef": self.model.coef_.tolist() if self.model is not None else None}


def reliability(y_true_binary: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    """Binned reliability curve for one class."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        out.append({"bin": [float(edges[i]), float(edges[i + 1])],
                    "n": int(m.sum()), "pred": float(p[m].mean()), "obs": float(y_true_binary[m].mean())})
    return out


def brier_multiclass(y_true: np.ndarray, P: np.ndarray, classes: tuple[str, ...]) -> float:
    Y = np.zeros_like(P)
    for i, c in enumerate(classes):
        Y[:, i] = (np.asarray(y_true) == c).astype(float)
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))
