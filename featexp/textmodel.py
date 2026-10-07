"""The TF-IDF text model: the baseline every policy has to beat, and half of the final combination."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline


def tfidf_model(C: float = 4.0) -> Pipeline:
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True),
                         LogisticRegression(C=C, max_iter=3000))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def tune_C(texts: pd.Series, y: np.ndarray, Cs: list[float], seed: int = 0) -> tuple[float, dict[float, float]]:
    """Regularization strength with the best 5-fold CV AUC."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = {C: float(roc_auc_score(y, cross_val_predict(tfidf_model(C), texts, y, cv=cv, method="predict_proba")[:, 1]))
            for C in Cs}
    return max(aucs, key=aucs.get), aucs


def oof_proba(texts: pd.Series, y: np.ndarray, C: float, seed: int = 0) -> np.ndarray:
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(tfidf_model(C), texts, y, cv=cv, method="predict_proba")[:, 1]
