"""Loading, the explore / select / holdout split, and n-gram hints for the brain.

    train file ──┬── explore rows: the only rows (and labels) the brain ever sees
                 └── select rows:  every scoring and selection decision is made here
    holdout     ──── used once, at the very end
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import Config


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in {".jsonl", ".json"}:
        return pd.read_json(p, lines=p.suffix == ".jsonl")
    return pd.read_csv(p)


@dataclass
class Dataset:
    select_text: pd.Series
    select_y: np.ndarray          # 0/1
    explore_text: pd.Series
    explore_y: np.ndarray
    test_text: pd.Series
    test_y: np.ndarray
    screen_idx: np.ndarray        # positions in the select rows used for cheap screening
    positive_label: object

    @property
    def train_text(self) -> pd.Series:
        """All training rows (select first, then explore), for the final models."""
        return pd.concat([self.select_text, self.explore_text], ignore_index=True)

    @property
    def train_y(self) -> np.ndarray:
        return np.concatenate([self.select_y, self.explore_y])


def prepare_data(cfg: Config) -> Dataset:
    d = cfg.data

    def read(path: str) -> pd.DataFrame:
        df = load_table(path)
        for col in (d.text_col, d.target_col):
            if col not in df.columns:
                raise KeyError(f"column {col!r} not in {list(df.columns)} ({path})")
        df = df[[d.text_col, d.target_col]].dropna().reset_index(drop=True)
        df[d.text_col] = df[d.text_col].astype(str).str.slice(0, d.max_chars)
        return df

    df = read(d.path)
    labels = sorted(df[d.target_col].unique())
    if len(labels) != 2:
        raise SystemExit(f"binary target required, found {len(labels)} values in {d.target_col!r}")
    positive = d.positive_label if d.positive_label is not None else labels[1]

    if d.test_path:
        dev, test = df, read(d.test_path)
    else:
        dev, test = train_test_split(df, test_size=d.test_size, random_state=d.seed, stratify=df[d.target_col])

    def cap(part: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(part) <= n:
            return part.reset_index(drop=True)
        sub, _ = train_test_split(part, train_size=n, random_state=d.seed, stratify=part[d.target_col])
        return sub.reset_index(drop=True)

    dev, test = cap(dev, d.max_dev_rows), cap(test, d.max_test_rows)
    if not 0 < d.explore_rows < len(dev):
        raise SystemExit("data.explore_rows must be between 0 and the number of dev rows")
    explore, select = train_test_split(dev, train_size=d.explore_rows, random_state=d.seed, stratify=dev[d.target_col])
    explore, select = explore.reset_index(drop=True), select.reset_index(drop=True)
    rng = np.random.default_rng(d.seed)
    screen = np.sort(rng.choice(len(select), size=min(d.screen_rows, len(select)), replace=False))

    def y01(part: pd.DataFrame) -> np.ndarray:
        return (part[d.target_col].to_numpy() == positive).astype(int)

    return Dataset(select[d.text_col], y01(select), explore[d.text_col], y01(explore),
                   test[d.text_col], y01(test), screen, positive)


def ngram_hints(texts: pd.Series, y: np.ndarray, top: int = 15) -> str:
    """Uni/bi-grams with the largest positive and negative weights in a TF-IDF logistic regression."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=max(3, len(texts) // 300), max_features=20000,
                          sublinear_tf=True, stop_words="english")
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return "  (vocabulary too small)"
    vocab = np.array(vec.get_feature_names_out())
    w = LogisticRegression(C=2.0, max_iter=3000).fit(X, y).coef_[0]
    order = np.argsort(w)
    return "\n".join(f"  {w[i]:+.2f}  {vocab[i]}" for i in list(order[::-1][:top]) + list(order[:top]))
