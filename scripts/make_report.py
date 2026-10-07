"""Build a one-page HTML overview of finished runs: the pipeline, each experiment group, and the
policies of every run (click a run to see them).

    .venv/bin/python scripts/make_report.py -o report/index.html \\
        --group "Default" "runs/v1_baseline/vcbench_seed*" refit \\
        --group "Original (TF-IDF C = 4)" "runs/v1_baseline/vcbench_seed*" \\
        --group "Policies scored on top of TF-IDF" "runs/vcbench_seed*"

A group is a label, a glob of run directories and an optional "refit" flag. "refit" refits the final
models with the current code (TF-IDF C tuned by CV) from cached Jev answers instead of reading the
run's saved results; nothing is sent to Jev when the answers are cached.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from featexp.config import load_config  # noqa: E402
from featexp.data import prepare_data  # noqa: E402
from featexp.final import fit_final, metrics, predict  # noqa: E402
from featexp.jev import JevScorer, Policy  # noqa: E402

def load_run(run_dir: Path, refit: bool, jev_model: str | None) -> dict:
    st = json.loads((run_dir / "state.json").read_text())
    s = st["summary"]
    pols = {p["name"]: p for p in st["policies"]}
    sel = st["selected"]
    holdout, stack_w = s["holdout"], s.get("stack_weights", {})
    if refit and sel:
        cfg = load_config(ROOT / "configs/vcbench.yaml", {"data": {"seed": st["config"]["data"]["seed"]}})
        if jev_model:
            cfg.jev.model = jev_model
        d = prepare_data(cfg)
        jev = JevScorer(cfg.jev, ROOT / cfg.cache_path)
        objs = [Policy.from_dict(pols[n]) for n in sel]
        X = jev.score(objs, d.train_text)[sel].to_numpy(float)
        Xt = jev.score(objs, d.test_text)[sel].to_numpy(float)
        b = fit_final(X, d.train_text, d.train_y, cfg.eval.beta, cfg.eval.tfidf_Cs)
        p = predict(b, Xt, d.test_text)
        holdout = {m: metrics(d.test_y, p[m], b["thresholds"][m], cfg.eval.beta) for m in p}
        stack_w = dict(zip(["tfidf_logit"] + sel, b["stack"][-1].coef_[0]))
    contrib = s.get("contributions", {})
    return {
        "name": run_dir.name, "seed": st["config"]["data"]["seed"], "holdout": holdout,
        "dev": s["dev_cv_perf"], "jev_requests": s["usage"]["jev"]["requests"],
        "tfidf_weight": stack_w.get("tfidf_logit"),
        "policies": [{"idea": pols[n]["idea"], "policy": pols[n]["policy"], "weight": stack_w.get(n),
                      "delta": contrib.get(n, {}).get("delta_perf")} for n in sel],
        "n_candidates": len(st["policies"]), "n_pool": len(st["pool"]),
    }


def pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.1f}"


def weight(p: dict) -> str:
    return "" if p["weight"] is None else f"{p['weight']:+.2f}"


def mean_sd(vals: list[float]) -> str:
    return f"{np.mean(vals) * 100:.1f}<span class=sd> ± {np.std(vals) * 100:.1f}</span>"


def group_html(label: str, note: str, runs: list[dict]) -> str:
    """One experiment group: per-run rows (click for policies), mean ± sd, and the TF-IDF baseline.
    Each row shows the selected policies alone and the policies combined with TF-IDF (stack)."""
    beta_key = next(k for k in runs[0]["holdout"]["tfidf"] if k.startswith("f"))
    cols = [("roc_auc", "AUC"), ("average_precision", "AP"), (beta_key, beta_key.upper())]
    blocks = [("policies", "Policies alone"), ("stack", "+ TF-IDF (stack)")]

    def cells(h: dict, block: str) -> str:
        cls = "num alone" if block == "policies" else "num"
        return "".join(f"<span class='{cls}'>{pct(h.get(block, {}).get(k))}</span>" for k, _ in cols)

    def mean_cells(block: str) -> str:
        hs = [r["holdout"][block] for r in runs if block in r["holdout"]]
        cls = "num alone" if block == "policies" else "num"
        return "".join(f"<span class='{cls}'>{mean_sd([h[k] for h in hs]) if hs else '–'}</span>" for k, _ in cols)

    tf = {k: float(np.mean([r["holdout"]["tfidf"][k] for r in runs])) for k, _ in cols}   # F-beta varies by seed
    rows = []
    for r in sorted(runs, key=lambda r: r["seed"]):
        h = r["holdout"]
        pol_rows = "".join(
            f"<tr><td class=num>{weight(p)}</td>"
            f"<td class=idea>{html.escape(p['idea'])}</td><td>{html.escape(p['policy'])}</td></tr>"
            for p in sorted(r["policies"], key=lambda p: -abs(p["weight"] or 0)))
        detail = (f"<table class=pol><thead><tr><th>weight</th><th>idea</th><th>policy</th></tr></thead>"
                  f"<tbody>{pol_rows}</tbody></table>" if r["policies"] else "<p class=muted>No policy selected.</p>")
        tfw = "" if r["tfidf_weight"] is None else f"TF-IDF weight in stack {r['tfidf_weight']:+.2f} · "
        n = len(r["policies"])
        rows.append(
            f"<details><summary><span class=seed>seed {r['seed']}</span>"
            f"<span class=npol>{n} {'policy' if n == 1 else 'policies'}</span>"
            f"{cells(h, 'policies')}{cells(h, 'stack')}</summary>"
            f"<div class=body><p class=muted>{tfw}{r['n_candidates']} candidates, {r['n_pool']} fully scored, "
            f"{r['jev_requests']:,} Jev requests</p>{detail}</div></details>")
    blank = "<span class='num alone'>–</span>" * len(cols)
    return f"""
<section>
  <h2>{html.escape(label)}</h2>
  <p class=note>{note}</p>
  <div class=grid>
    <div class=band><span></span><span></span><span class='blk alone'>{blocks[0][1]}</span><span class=blk>{blocks[1][1]}</span></div>
    <div class=hdr><span>run</span><span></span>{''.join(f"<span class='num alone'>{c}</span>" for _, c in cols)}{''.join(f'<span class=num>{c}</span>' for _, c in cols)}</div>
    {''.join(rows)}
    <div class=mean><span>mean ± sd</span><span></span>{mean_cells('policies')}{mean_cells('stack')}</div>
    <div class=ref><span>TF-IDF alone</span><span></span>{blank}{''.join(f'<span class=num>{pct(tf[k])}</span>' for k, _ in cols)}</div>
  </div>
</section>"""


PAGE = """<!doctype html>
<html lang=en>
<head>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Policy Discovery Report</title>
<style>
:root {{ --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --line:#e4e3df; --accent:#2f5d50; --soft:#f1f0ec; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#161615; --fg:#ecebe7; --muted:#9d9c96; --line:#2c2c2a; --accent:#8fc4b3; --soft:#1f1f1d; }} }}
:root[data-theme="dark"] {{ --bg:#161615; --fg:#ecebe7; --muted:#9d9c96; --line:#2c2c2a; --accent:#8fc4b3; --soft:#1f1f1d; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:48px 16px 80px; }}
h1 {{ font-size:22px; font-weight:600; margin:0 0 4px; }}
h2 {{ font-size:16px; font-weight:600; margin:0 0 2px; }}
.sub, .note, .muted {{ color:var(--muted); }}
.sub {{ margin:0 0 32px; }}
.note {{ margin:0 0 12px; font-size:14px; }}
section {{ margin-top:40px; }}
ol.flow {{ padding-left:20px; margin:12px 0 0; }}
ol.flow li {{ margin:4px 0; }}
ol.flow b {{ font-weight:600; }}
.grid > div, details > summary {{ display:grid; grid-template-columns: 1fr .9fr repeat(6, 74px); gap:6px; align-items:baseline; padding:9px 4px; border-bottom:1px solid var(--line); }}
.grid > div.band {{ border-bottom:none; padding-bottom:0; font-size:12px; color:var(--muted); }}
.band .blk {{ grid-column: span 3; text-align:center; border-bottom:1px solid var(--line); padding-bottom:4px; }}
.alone {{ color:var(--muted); }}
.mean .alone {{ font-weight:500; }}
.hdr {{ color:var(--muted); font-size:13px; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.sd {{ color:var(--muted); font-size:12px; }}
.mean {{ font-weight:600; }}
.ref {{ color:var(--muted); }}
details > summary {{ cursor:pointer; list-style:none; }}
details > summary::-webkit-details-marker {{ display:none; }}
details > summary .seed::before {{ content:"▸"; display:inline-block; width:14px; color:var(--muted); transition:transform .15s; }}
details[open] > summary .seed::before {{ transform:rotate(90deg); }}
details > summary:hover {{ background:var(--soft); }}
.npol {{ color:var(--muted); }}
.body {{ padding:4px 4px 16px 18px; border-bottom:1px solid var(--line); }}
.body p {{ font-size:13px; margin:8px 0 10px; }}
table.pol {{ width:100%; border-collapse:collapse; font-size:14px; }}
table.pol th {{ text-align:left; color:var(--muted); font-weight:500; font-size:12px; padding:4px 8px 4px 0; }}
table.pol td {{ padding:6px 8px 6px 0; border-top:1px solid var(--line); vertical-align:top; }}
table.pol td.num {{ width:56px; text-align:left; }}
td.idea {{ color:var(--accent); width:28%; }}
.key {{ margin-top:12px; font-size:13px; color:var(--muted); }}
@media (max-width:600px) {{
  .grid > div, details > summary {{ grid-template-columns: 1fr repeat(6, 42px); gap:4px; font-size:13px; }}
  .grid > div > span:nth-child(2), details > summary .npol {{ display:none; }}
  .sd {{ display:none; }}
  td.idea {{ display:none; }}
}}
</style>
</head>
<body>
<main>
  <h1>Policy discovery on VCBench</h1>
  <p class=sub>Predict founder success (9% positives) from anonymised profiles. 4,500 public rows for discovery,
  4,500 private rows scored once. Generated {today}.</p>

  <section style="margin-top:0">
    <h2>Pipeline</h2>
    <ol class=flow>
      <li><b>Split</b> — 1,000 public rows are shown to the brain; the other 3,500 are only used for scoring.</li>
      <li><b>Ideas</b> — DeepSeek proposes 5 new dimensions, guided by the cases the current model gets wrong.</li>
      <li><b>Policies</b> — each idea becomes 3 investor heuristics.</li>
      <li><b>Judge</b> — Jev answers "considering this heuristic, will this founder succeed?" for every profile.</li>
      <li><b>Select</b> — cross-validated forward/backward selection keeps policies that raise AUC, penalizing count and redundancy.
          Repeat up to 12 rounds.</li>
      <li><b>Combine</b> — selected policies + TF-IDF (stacked logistic regression), evaluated once on the private set.</li>
    </ol>
  </section>
  {groups}
  <p class=key>Private holdout, in %. "Policies alone" is a logistic regression on the selected policies only;
  "+ TF-IDF (stack)" combines them with the TF-IDF text model. Click a run to see its policies.
  Weights are on standardized features in the stacked model; because every policy asks about success, they are
  conditional contrasts rather than the direction stated in the text. F0.5 depends on a fitted threshold and is the
  noisiest of the three.</p>
</main>
</body>
</html>
"""


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="report/index.html")
    ap.add_argument("--group", nargs="+", action="append", required=True, metavar=("LABEL", "GLOB"),
                    help='label, glob of run dirs, optional "refit"; a fourth item is shown as a note')
    ap.add_argument("--jev-model", help="Jev version for refits (default: pin jev-latest)")
    args = ap.parse_args()
    parts = []
    for g in args.group:
        label, pattern, rest = g[0], g[1], g[2:]
        refit = "refit" in rest
        note = next((x for x in rest if x != "refit"), "")
        dirs = [Path(p) for p in sorted(glob.glob(pattern)) if (Path(p) / "state.json").exists()]
        runs = [load_run(d, refit, args.jev_model) for d in dirs]
        print(f"{label}: {len(runs)} runs", file=sys.stderr)
        parts.append(group_html(label, note, runs))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.format(groups="".join(parts), today=date.today().isoformat()), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
