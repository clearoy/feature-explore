"""Out-of-sample scoring of policy sets and penalized selection (on the select rows only).

    J(F) = Perf_CV(base + F) - size_penalty * |F| - redundancy_penalty * mean_{f != g in F} |corr(f, g)|

Perf_CV is the metric of out-of-fold predictions of a logistic regression, averaged over
repeated stratified K-fold with fixed splits, so every set is compared on the same folds.
`base` (the TF-IDF model's out-of-fold logit) is always in the regression, so a policy only
counts for what it adds on top of word statistics, which is how it is used in the final model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import EvalConfig


def make_model(C: float = 1.0) -> Pipeline:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def score(metric: str, y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p) if metric == "average_precision" else roc_auc_score(y, p))


class Evaluator:
    def __init__(self, y: np.ndarray, cfg: EvalConfig, seed: int = 0, base: np.ndarray | None = None):
        self.y = y
        self.base = None if base is None else np.asarray(base, dtype=float).reshape(-1, 1)
        self.cfg = cfg
        self.metric = cfg.metric
        self.frame = pd.DataFrame(index=range(len(y)))   # one column per evaluated policy
        self.splits = [list(StratifiedKFold(cfg.folds, shuffle=True, random_state=seed + r).split(np.zeros(len(y)), y))
                       for r in range(cfg.repeats)]
        self._cache: dict[tuple[str, ...], tuple[float, np.ndarray]] = {}

    def add(self, name: str, values: np.ndarray) -> None:
        self.frame[name] = values
        self._corr.cache_clear()

    # ------------------------------------------------------------ scoring
    def oof(self, names: Iterable[str]) -> tuple[float, np.ndarray]:
        """(mean metric over repeats, out-of-fold P(positive) of the first repeat)."""
        key = tuple(sorted(names))
        if key in self._cache:
            return self._cache[key]
        X = self.frame[list(key)].to_numpy(float)
        if self.base is not None:
            X = np.hstack([self.base, X])
        scores, first = [], None
        for splits in self.splits:
            pred = np.zeros(len(self.y))
            for tr, te in splits:
                pred[te] = (self.y[tr].mean() if X.shape[1] == 0
                            else make_model().fit(X[tr], self.y[tr]).predict_proba(X[te])[:, 1])
            scores.append(score(self.metric, self.y, pred))
            first = pred if first is None else first
        self._cache[key] = (float(np.mean(scores)), first)
        return self._cache[key]

    def perf(self, names: Iterable[str]) -> float:
        return self.oof(names)[0]

    def residuals(self, names: Iterable[str]) -> np.ndarray:
        return self.y - self.oof(names)[1]

    @lru_cache(maxsize=1)
    def _corr(self) -> pd.DataFrame:
        return self.frame.corr(method="spearman").abs().fillna(0.0)

    def redundancy(self, names: Iterable[str]) -> float:
        names = list(names)
        if len(names) < 2:
            return 0.0
        c = self._corr()
        return float(np.mean([c.loc[a, b] for i, a in enumerate(names) for b in names[i + 1:]]))

    def max_corr_with(self, name: str, others: Iterable[str]) -> tuple[float, str]:
        c = self._corr()
        return max(((float(c.loc[name, o]), o) for o in others if o != name), default=(0.0, ""))

    def objective(self, names: Iterable[str]) -> float:
        names = list(names)
        return (self.perf(names) - self.cfg.size_penalty * len(names)
                - self.cfg.redundancy_penalty * self.redundancy(names))

    # ------------------------------------------------------------ selection
    def select(self, pool: list[str], start: list[str], max_features: int) -> list[str]:
        """Warm-started greedy forward selection on J, then backward pruning."""
        current = [n for n in start if n in pool]
        best = self.objective(current)
        while len(current) < max_features:
            options = [(self.objective(current + [c]), c) for c in pool if c not in current]
            if not options:
                break
            val, cand = max(options)
            if val - best < self.cfg.min_gain:
                break
            current.append(cand)
            best = val
        improved = True
        while improved and current:
            improved = False
            val, drop = max((self.objective([c for c in current if c != d]), d) for d in current)
            if val >= best - 1e-9:
                current.remove(drop)
                best, improved = val, True
        return current

    def contributions(self, selected: list[str]) -> dict[str, dict[str, float | str]]:
        """Per selected policy: drop in Perf_CV when removed, and its highest |corr| with the others."""
        full = self.perf(selected)
        out = {}
        for n in selected:
            rest = [m for m in selected if m != n]
            mc, partner = self.max_corr_with(n, rest)
            out[n] = {"delta_perf": full - self.perf(rest), "max_corr": mc, "most_correlated_with": partner}
        return out
