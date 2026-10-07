"""Run configuration: a YAML file merged over these defaults (the defaults are the best VCBench settings)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    path: str = ""                     # training file (csv / parquet / jsonl)
    test_path: str | None = None       # separate holdout file; otherwise test_size is split off `path`
    text_col: str = "text"
    target_col: str = "y"
    positive_label: Any = None         # value of target_col that counts as positive (default: the larger one)
    task_description: str = ""         # what y means; shown to the brain and put into the Jev template
    max_chars: int = 4000              # text is truncated to this many characters
    test_size: float = 0.2
    max_dev_rows: int = 4500
    max_test_rows: int = 4500
    explore_rows: int = 1000           # dev rows only the brain sees (examples, hints); never scored
    screen_rows: int = 600             # scored rows used to screen new policies cheaply
    seed: int = 42


@dataclass
class BrainConfig:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.9
    max_tokens: int = 8000
    ideas_per_iter: int = 5            # step 1: new dimensions per iteration
    policies_per_idea: int = 3         # step 2: heuristics written per idea
    n_examples: int = 14               # explore texts shown per iteration (half residual, half random)
    example_chars: int = 700


@dataclass
class JevConfig:
    model: str = "jev-latest"          # an alias is pinned to the concrete version answering the first request
    concurrency: int = 24
    questions_per_request: int = 10
    timeout: float = 60.0
    max_retries: int = 3
    # How a policy is put to Jev: one yes/no question per text. {policy} required, {task} optional.
    policy_template: str = (
        "Task: {task}\n"
        "Heuristic (guidance, not a strict rule): {policy}\n"
        "Considering this heuristic along with the full case, is the answer YES?"
    )


@dataclass
class SearchConfig:
    iterations: int = 12
    patience: int = 4                  # stop after this many iterations without objective gain
    promote_per_iter: int = 5          # policies fully extracted per iteration (main Jev cost)
    min_screen_signal: float = 0.03    # min |corr| with the residual (or half of it with y) to be promoted
    max_selected_corr: float = 0.9     # screened out if |corr| with a SELECTED policy exceeds this
    max_pool_corr: float = 0.97        # screened out if |corr| with any evaluated policy exceeds this
    max_features: int = 15


@dataclass
class EvalConfig:
    base: str = "none"                 # none: policies are scored on their own (best on VCBench)
                                       # tfidf: policies are scored on top of a TF-IDF model; gains shrink to the
                                       # noise level and selection does worse on the holdout (see README)
    tfidf_Cs: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    metric: str = "roc_auc"            # roc_auc | average_precision (selection metric)
    folds: int = 5
    repeats: int = 2
    size_penalty: float = 0.002        # objective cost per selected policy
    redundancy_penalty: float = 0.005  # cost x mean |corr| among selected policies
    min_gain: float = 0.001            # a change must improve the objective by this much
    beta: float = 0.5                  # F-beta for the decision threshold of the final models


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    jev: JevConfig = field(default_factory=JevConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    out_dir: str = "runs"
    cache_path: str = ".cache/jev.sqlite"
    mock: bool = False                 # offline: fake Jev + fake brain, for smoke tests

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(obj: Any, values: dict[str, Any]) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in (values or {}).items():
        if key not in names:
            raise ValueError(f"Unknown config key {type(obj).__name__}.{key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg
