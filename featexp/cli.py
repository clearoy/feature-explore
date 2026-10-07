"""Command line.

    python -m featexp check                                  verify the DeepSeek and Jev keys
    python -m featexp run -c configs/vcbench.yaml [--name N] discover policies, fit final models, holdout report
    python -m featexp predict -r runs/N -i new.csv -o out.csv score new texts with a finished run
    python -m featexp demo-data                              synthetic data for an offline --mock run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .config import load_config


def cmd_run(args: argparse.Namespace) -> None:
    from .brain import Brain
    from .data import prepare_data
    from .final import finalize
    from .jev import JevScorer
    from .search import PolicySearch

    overrides: dict = {}
    if args.mock:
        overrides["mock"] = True
    if args.iterations:
        overrides["search"] = {"iterations": args.iterations}
    if args.seed is not None:
        overrides["data"] = {"seed": args.seed}
    if args.concurrency:
        overrides["jev"] = {"concurrency": args.concurrency}
    cfg = load_config(args.config, overrides)
    if not cfg.mock:
        missing = [k for k in ("DEEPSEEK_API_KEY", "TYPESAFE_API_KEY") if not os.environ.get(k, "").strip()]
        if missing:
            sys.exit(f"Missing {', '.join(missing)} in .env (or add --mock for an offline smoke test)")
    run_dir = Path(cfg.out_dir) / (args.name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(cfg)
    jev = JevScorer(cfg.jev, cfg.cache_path, mock=cfg.mock)
    search = PolicySearch(cfg, data, jev, Brain(cfg.brain, mock=cfg.mock), run_dir)
    search.run()
    finalize(search, cfg, data, jev, run_dir)


def cmd_predict(args: argparse.Namespace) -> None:
    from .data import load_table
    from .final import predict
    from .jev import JevScorer, Policy

    run_dir = Path(args.run)
    cfg = load_config(None, json.loads((run_dir / "state.json").read_text())["config"])
    bundle = joblib.load(run_dir / "model.joblib")
    policies = [Policy.from_dict(d) for d in json.loads((run_dir / "policies.json").read_text())["policies"]]
    df = load_table(args.input)
    texts = df[args.text_col or cfg.data.text_col].astype(str).str.slice(0, cfg.data.max_chars).reset_index(drop=True)
    X = JevScorer(cfg.jev, cfg.cache_path, mock=cfg.mock).score(policies, texts)[bundle["selected"]]
    probs = predict(bundle, X.to_numpy(float), texts)
    out = pd.concat([df.reset_index(drop=True), X], axis=1)
    for m, p in probs.items():
        out[f"p_{m}"] = p
    out["predicted"] = (probs["stack"] >= bundle["thresholds"]["stack"]).astype(int)
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} rows to {args.output} (p_stack = recommended score; predicted uses its "
          f"F{bundle['beta']} threshold {bundle['thresholds']['stack']:.3f})")


def cmd_check(_: argparse.Namespace) -> None:
    ok = True
    try:
        from openai import OpenAI

        c = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        r = c.chat.completions.create(model="deepseek-chat", max_tokens=5,
                                      messages=[{"role": "user", "content": "Reply with: ok"}])
        print("DeepSeek: OK ->", r.choices[0].message.content)
    except Exception as e:  # noqa: BLE001
        ok = False
        print("DeepSeek: FAILED ->", type(e).__name__, e)
    try:
        from typesafe_sdk import Noul, TypeSafeClient

        with TypeSafeClient() as c:
            r = c.system_one(state="I love this product, it works great!",
                             questions={"q": Noul(instructions="Is the review positive?")})
            print(f"Jev: OK -> P(positive) = {r.answers['q'].noul:.3f} (model {r.model})")
    except Exception as e:  # noqa: BLE001
        ok = False
        print("Jev: FAILED ->", type(e).__name__, e)
    sys.exit(0 if ok else 1)


def cmd_demo_data(args: argparse.Namespace) -> None:
    """Synthetic support tickets with a latent binary target, for an offline --mock run."""
    rng = np.random.default_rng(0)
    products = ["router", "laptop", "phone", "printer", "headset", "monitor"]
    issues = ["won't turn on", "keeps disconnecting", "is very slow", "makes a strange noise", "stopped charging"]
    rows = []
    for _ in range(args.n):
        angry, tried, premium = rng.random() < 0.4, rng.random() < 0.5, rng.random() < 0.3
        days = int(rng.integers(1, 60))
        parts = [f"My {rng.choice(products)} {rng.choice(issues)}.",
                 rng.choice(["This is unacceptable!!", "I am furious.", "Worst purchase ever!"] if angry else
                            ["Could you help please?", "Thanks in advance.", "Appreciate any tips."]),
                 f"I bought it {days} days ago."]
        if tried:
            parts.append("I already restarted it and reinstalled the drivers, nothing worked.")
        if premium:
            parts.append("I'm on the premium support plan.")
        rng.shuffle(parts)
        logit = -1.2 + 1.6 * angry + 1.0 * tried + 1.3 * premium * angry - 0.02 * days + rng.normal(0, 0.5)
        rows.append({"text": " ".join(parts), "y": int(rng.random() < 1 / (1 + np.exp(-logit)))})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.n} rows to {args.output}")


def main() -> None:
    load_dotenv(Path.cwd() / ".env")
    p = argparse.ArgumentParser(prog="python -m featexp", description="Idea -> policy feature discovery")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="discover policies, fit the final models, evaluate on the holdout")
    r.add_argument("-c", "--config", required=True)
    r.add_argument("--name", help="run directory name under out_dir (default: timestamp)")
    r.add_argument("--iterations", type=int)
    r.add_argument("--seed", type=int, help="override data.seed (explore/select split, screen sample, CV folds)")
    r.add_argument("--concurrency", type=int, help="override jev.concurrency (lower it when runs share the API)")
    r.add_argument("--mock", action="store_true", help="offline: fake Jev + fake brain")
    r.set_defaults(fn=cmd_run)

    pr = sub.add_parser("predict", help="score new texts with a finished run")
    pr.add_argument("-r", "--run", required=True)
    pr.add_argument("-i", "--input", required=True)
    pr.add_argument("-o", "--output", required=True)
    pr.add_argument("--text-col")
    pr.set_defaults(fn=cmd_predict)

    sub.add_parser("check", help="verify the DeepSeek and Jev API keys").set_defaults(fn=cmd_check)

    d = sub.add_parser("demo-data", help="write a synthetic dataset for an offline --mock run")
    d.add_argument("-n", type=int, default=1500)
    d.add_argument("-o", "--output", default="data/demo.csv")
    d.set_defaults(fn=cmd_demo_data)

    args = p.parse_args()
    args.fn(args)
