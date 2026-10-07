"""Final models, one holdout evaluation, report, and the prediction bundle.

All final models are fit on every training row (select + explore). Decision thresholds maximize
F-beta on out-of-fold predictions of the training rows; the holdout is only scored.

  policies   logistic regression on the selected policies
  tfidf      TF-IDF (1-2 grams) logistic regression on the raw text, C tuned by CV AUC
  stack      logistic regression on [logit of TF-IDF's out-of-fold probability, selected policies]
  average    mean of the policies and TF-IDF probabilities
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import Dataset
from .evaluator import Evaluator, make_model
from .jev import JevScorer, Policy
from .textmodel import logit, tfidf_model, tune_C

MODELS = ["tfidf", "policies", "average", "stack"]


def best_threshold(y: np.ndarray, p: np.ndarray, beta: float) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.0, 0.999, 400)))
    return float(max(grid, key=lambda t: fbeta_score(y, p >= t, beta=beta, zero_division=0)))


def metrics(y: np.ndarray, p: np.ndarray, threshold: float, beta: float) -> dict[str, float]:
    hat = (p >= threshold).astype(int)
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "precision": float(precision_score(y, hat, zero_division=0)),
            "recall": float(recall_score(y, hat, zero_division=0)),
            f"f{beta}": float(fbeta_score(y, hat, beta=beta, zero_division=0)), "threshold": threshold}


def fit_final(policy_X: np.ndarray, texts: pd.Series, y: np.ndarray, beta: float,
              tfidf_Cs: list[float]) -> dict[str, Any]:
    """Fit every final model on the training rows; thresholds from 5-fold OOF predictions.
    With no selected policy (none beat TF-IDF) only the TF-IDF model is fit."""
    C, _ = tune_C(texts, y, tfidf_Cs)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    tf_oof = cross_val_predict(tfidf_model(C), texts, y, cv=cv, method="predict_proba")[:, 1]
    if policy_X.shape[1] == 0:
        return {"tfidf_C": C, "tfidf": tfidf_model(C).fit(texts, y), "policies": None, "stack": None,
                "thresholds": {"tfidf": best_threshold(y, tf_oof, beta)},
                "oof_metrics": {"tfidf": {"roc_auc": float(roc_auc_score(y, tf_oof)),
                                          "average_precision": float(average_precision_score(y, tf_oof))}}}
    pol_oof = cross_val_predict(make_model(), policy_X, y, cv=cv, method="predict_proba")[:, 1]
    S = np.column_stack([logit(tf_oof), policy_X])
    stack_oof = cross_val_predict(make_model(), S, y, cv=cv, method="predict_proba")[:, 1]
    oof = {"tfidf": tf_oof, "policies": pol_oof, "average": (tf_oof + pol_oof) / 2, "stack": stack_oof}
    return {
        "tfidf_C": C,
        "tfidf": tfidf_model(C).fit(texts, y),
        "policies": make_model().fit(policy_X, y),
        "stack": make_model().fit(S, y),
        "thresholds": {m: best_threshold(y, oof[m], beta) for m in MODELS},
        "oof_metrics": {m: {"roc_auc": float(roc_auc_score(y, oof[m])),
                            "average_precision": float(average_precision_score(y, oof[m]))} for m in MODELS},
    }


def predict(bundle: dict[str, Any], policy_X: np.ndarray, texts: pd.Series) -> dict[str, np.ndarray]:
    tf = bundle["tfidf"].predict_proba(texts)[:, 1]
    if bundle["policies"] is None:
        return {"tfidf": tf}
    pol = bundle["policies"].predict_proba(policy_X)[:, 1]
    stack = bundle["stack"].predict_proba(np.column_stack([logit(tf), policy_X]))[:, 1]
    return {"tfidf": tf, "policies": pol, "average": (tf + pol) / 2, "stack": stack}


def finalize(search, cfg, data: Dataset, jev: JevScorer, run_dir: Path) -> dict[str, Any]:
    ev: Evaluator = search.ev
    sel = search.selected
    policies: list[Policy] = [search.policies[n] for n in sel]
    if not policies:
        print("\nno policy improved on the baseline; reporting the TF-IDF model only", flush=True)
    beta = cfg.eval.beta
    print(f"\nfinal models on {len(data.train_y)} training rows; holdout {len(data.test_y)} rows ...", flush=True)

    train_X = np.vstack([ev.frame[sel].to_numpy(float), search.explore_frame[sel].to_numpy(float)])
    bundle = fit_final(train_X, data.train_text, data.train_y, beta, cfg.eval.tfidf_Cs)
    test_X = jev.score(policies, data.test_text).reindex(columns=sel).to_numpy(float)
    probs = predict(bundle, test_X, data.test_text)
    holdout = {m: metrics(data.test_y, probs[m], bundle["thresholds"][m], beta) for m in probs}

    weights = dict(zip(["tfidf_logit"] + sel, bundle["stack"][-1].coef_[0])) if sel else {}
    pol_weights = dict(zip(sel, bundle["policies"][-1].coef_[0])) if sel else {}
    contrib = ev.contributions(sel) if sel else {}
    summary = {
        "n_select": len(data.select_y), "n_explore": len(data.explore_y), "n_test": len(data.test_y),
        "metric": ev.metric, "dev_cv_perf": ev.perf(sel), "dev_objective": ev.objective(sel),
        "redundancy": ev.redundancy(sel), "holdout": holdout, "oof": bundle["oof_metrics"],
        "policy_weights": pol_weights, "stack_weights": weights, "contributions": contrib,
        "n_candidates": len(search.policies), "n_pool": len(search.pool),
        "tfidf_C": bundle["tfidf_C"], "selection_base": cfg.eval.base, "jev_model": jev.version,
        "usage": {"jev": jev.usage(), "brain": search.brain.usage()},
    }
    joblib.dump({**bundle, "selected": sel, "beta": beta, "jev_model": jev.version}, run_dir / "model.joblib")
    (run_dir / "policies.json").write_text(json.dumps(
        {"selected": sel, "policies": [p.to_dict() for p in policies]}, indent=2, ensure_ascii=False))
    pd.DataFrame({"y": data.test_y, **{f"p_{m}": p for m, p in probs.items()}}).to_csv(
        run_dir / "holdout_predictions.csv", index=False)
    search.save({"summary": summary})
    write_report(run_dir, cfg, search, summary)

    print(f"\n{'model':10s} {'AUC':>6s} {'AP':>6s} {'Prec':>6s} {'Rec':>6s} {'F' + str(beta):>6s}")
    for m, h in holdout.items():
        print(f"{m:10s} {h['roc_auc']*100:6.1f} {h['average_precision']*100:6.1f} {h['precision']*100:6.1f} "
              f"{h['recall']*100:6.1f} {h[f'f{beta}']*100:6.1f}")
    print(f"\nreport: {run_dir / 'report.md'}")
    return summary


def write_report(run_dir: Path, cfg, search, s: dict[str, Any]) -> None:
    beta = cfg.eval.beta
    names = {"tfidf": "TF-IDF alone", "policies": "Selected policies alone", "average": "Average of the two",
             "stack": "**Stack: TF-IDF + policies**"}
    lines = [
        "# Idea -> Policy feature discovery", "",
        f"- Data: `{cfg.data.path}`; {s['n_select']} select rows (scoring), {s['n_explore']} explore rows "
        f"(seen by the brain), holdout {s['n_test']} rows used once",
        f"- {s['n_candidates']} candidate policies, {s['n_pool']} fully evaluated, **{len(search.selected)} selected**; "
        f"dev CV {s['metric']} {s['dev_cv_perf']:.4f}, mean |corr| {s['redundancy']:.2f}",
        f"- Selection scored policies on top of: {s['selection_base']}; final TF-IDF C={s['tfidf_C']}; "
        f"Jev pinned to `{s['jev_model']}`",
        f"- Jev {s['usage']['jev']}; brain {s['usage']['brain']}", "",
        "## Holdout", "",
        f"| model | AUC | AP | precision | recall | F{beta} |", "|---|---|---|---|---|---|",
        *[f"| {names[m]} | {h['roc_auc']*100:.1f} | {h['average_precision']*100:.1f} | {h['precision']*100:.1f} | "
          f"{h['recall']*100:.1f} | {h[f'f{beta}']*100:.1f} |" for m, h in s["holdout"].items()],
        "",
        f"F{beta} depends on a threshold fitted on training OOF predictions and moves by a few points between "
        "runs; AUC and AP are the steadier comparison.", "",
        "## Selected policies", "",
        "Weights are on standardized features. Each feature is Jev's P(success | this heuristic), so all policies "
        "share an 'overall promise' component and the weights are conditional contrasts, not the direction stated "
        "in the text.", "",
        "| weight (policies model) | weight (stack) | dev delta | idea | policy |", "|---|---|---|---|---|",
    ]
    for n in sorted(search.selected, key=lambda n: -abs(s["policy_weights"][n])):
        p = search.policies[n]
        lines.append(f"| {s['policy_weights'][n]:+.3f} | {s['stack_weights'][n]:+.3f} | "
                     f"{s['contributions'][n]['delta_perf']:+.4f} | {p.idea} | {p.policy} |")
    if s["stack_weights"]:
        lines.append(f"| | {s['stack_weights']['tfidf_logit']:+.3f} | | TF-IDF (stack only) | |")
    lines += ["",
              "## Search trajectory", "", "| iter | dev perf | objective | #selected | promoted |", "|---|---|---|---|---|"]
    for h in search.history:
        lines.append(f"| {h['iteration']} | {h['perf']:.4f} | {h['objective']:.4f} | {len(h['selected'])} | "
                     f"{', '.join(h['promoted'])} |")
    lines += ["", "## All candidate policies", "", "| iter | status | gain | idea | policy | note |",
              "|---|---|---|---|---|---|"]
    for p in search.policies.values():
        g = p.stats.get("gain")
        lines.append(f"| {p.iteration} | {p.status} | {'' if g is None else f'{g:+.4f}'} | {p.idea} | "
                     f"{p.policy} | {'; '.join(p.notes)[:100]} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
