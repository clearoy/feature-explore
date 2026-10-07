"""Holdout results of several runs side by side, with mean and standard deviation.

    .venv/bin/python scripts/summarize.py runs/vcbench_seed42_* runs/vcbench_seed1_* ...
"""

import json
import sys
from pathlib import Path

import numpy as np

MODELS = ["tfidf", "policies", "average", "stack"]
COLS = [("roc_auc", "AUC"), ("average_precision", "AP"), ("precision", "Prec"), ("recall", "Rec")]


def main() -> None:
    runs = [Path(p) for p in sys.argv[1:] if (Path(p) / "state.json").exists()]
    states = {r.name: json.loads((r / "state.json").read_text()) for r in runs}
    states = {k: v for k, v in states.items() if "summary" in v}
    if not states:
        sys.exit("no finished runs given")
    beta = next(iter(states.values()))["config"]["eval"]["beta"]
    cols = COLS + [(f"f{beta}", f"F{beta}")]
    print(f"{'run':38s} {'#pol':>4s} {'devAUC':>6s}  " + "  ".join(f"{m}:AUC/AP" for m in ["policies", "stack"]))
    for name, s in states.items():
        h = s["summary"]["holdout"]
        print(f"{name:38s} {len(s['selected']):4d} {s['summary']['dev_cv_perf']*100:6.1f}  "
              + "  ".join(f"{h[m]['roc_auc']*100:5.1f}/{h[m]['average_precision']*100:4.1f}" if m in h else "  n/a     "
                          for m in ["policies", "stack"]))
    print(f"\nmean ± sd over {len(states)} runs (holdout, %):")
    print(f"{'model':10s} " + " ".join(f"{c:>12s}" for _, c in cols))
    for m in MODELS:
        runs_m = [s["summary"]["holdout"][m] for s in states.values() if m in s["summary"]["holdout"]]
        if not runs_m:
            continue
        vals = {k: [h[k] * 100 for h in runs_m] for k, _ in cols}
        print(f"{m:10s} " + " ".join(f"{np.mean(vals[k]):6.1f} ± {np.std(vals[k]):3.1f}" for k, _ in cols))
    n_pol = [len(s["selected"]) for s in states.values()]
    print(f"\npolicies selected: {n_pol} (mean {np.mean(n_pol):.1f})")


if __name__ == "__main__":
    main()
