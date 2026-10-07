"""The idea -> policy search loop.

Each iteration:
  1. ideas     the brain names new dimensions, seeing only explore rows (residual + random examples),
               the selected policies with their contributions, and how earlier ideas fared
  2. policies  each idea becomes a few investor heuristics (in parallel)
  3. screen    Jev scores the new policies on `screen_rows` select rows; keep those that correlate with
               the current model's residual and are not near-duplicates of what is already there
  4. promote   the best `promote_per_iter` are scored on every train row
  5. select    warm-started forward/backward selection on J over every policy evaluated so far
Stops after `patience` iterations without objective gain, or after `iterations`.

With eval.base = tfidf (default) a TF-IDF model is part of every regression, residual and
example shown to the brain, so the search looks for what word statistics miss.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .brain import Brain
from .config import Config
from .data import Dataset, ngram_hints
from .evaluator import Evaluator, make_model
from .jev import JevScorer, Policy
from .textmodel import logit, oof_proba, tfidf_model, tune_C


def max_abs_corr(m: np.ndarray, t: np.ndarray) -> float:
    m = np.where(np.isnan(m), np.nanmedian(m), m)
    if m.std() < 1e-9 or t.std() < 1e-9:
        return 0.0
    v = abs(float(np.corrcoef(m, t)[0, 1]))
    return 0.0 if np.isnan(v) else v


class PolicySearch:
    def __init__(self, cfg: Config, data: Dataset, jev: JevScorer, brain: Brain, run_dir: Path):
        self.cfg, self.data, self.jev, self.brain, self.run_dir = cfg, data, jev, brain, run_dir
        self.base_sel: np.ndarray | None = None      # TF-IDF logit: out-of-fold on select rows
        self.base_explore: np.ndarray | None = None  # TF-IDF logit on explore rows (model fit on select rows)
        self.tfidf_C: float | None = None
        if cfg.eval.base == "tfidf":
            self.tfidf_C, aucs = tune_C(data.select_text, data.select_y, cfg.eval.tfidf_Cs, cfg.data.seed)
            print(f"tfidf base: C={self.tfidf_C} (CV AUC by C: "
                  + ", ".join(f"{c}:{a:.4f}" for c, a in aucs.items()) + ")", flush=True)
            self.base_sel = logit(oof_proba(data.select_text, data.select_y, self.tfidf_C, cfg.data.seed))
            m = tfidf_model(self.tfidf_C).fit(data.select_text, data.select_y)
            self.base_explore = logit(m.predict_proba(data.explore_text)[:, 1])
        self.ev = Evaluator(data.select_y, cfg.eval, seed=cfg.data.seed, base=self.base_sel)
        self.explore_frame = pd.DataFrame(index=range(len(data.explore_y)))   # policy values on explore rows
        self.policies: dict[str, Policy] = {}
        self.pool: list[str] = []          # scored on every train row
        self.selected: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.baseline = self.ev.perf([])
        self.hints = ngram_hints(data.explore_text, data.explore_y)
        print(f"select={len(data.select_y)} explore={len(data.explore_y)} test={len(data.test_y)} "
              f"positives={data.select_y.mean():.1%} metric={self.ev.metric} "
              f"baseline ({'tfidf' if self.base_sel is not None else 'no features'})={self.baseline:.4f}", flush=True)

    # ------------------------------------------------------------ context for the brain
    def explore_predictions(self) -> np.ndarray:
        """P(positive) on the explore rows from a model (base + selected policies) fit on the select rows."""
        X_sel = self.ev.frame[self.selected].to_numpy(float)
        X_ex = self.explore_frame[self.selected].to_numpy(float)
        if self.base_sel is not None:
            X_sel = np.column_stack([self.base_sel, X_sel])
            X_ex = np.column_stack([self.base_explore, X_ex])
        if X_sel.shape[1] == 0:
            return np.full(len(self.data.explore_y), self.data.select_y.mean())
        return make_model().fit(X_sel, self.data.select_y).predict_proba(X_ex)[:, 1]

    def context(self, iteration: int) -> dict[str, Any]:
        sel, ev, d = self.selected, self.ev, self.data
        contrib = ev.contributions(sel) if sel else {}
        sel_lines = [
            f"  {n} | jev_policy | delta_perf={contrib[n]['delta_perf']:+.4f} | max_corr={contrib[n]['max_corr']:.2f} | "
            f"{self.policies[n].policy} | spec={json.dumps(self._spec(self.policies[n]), ensure_ascii=False)[:300]}"
            for n in sel]
        p = self.explore_predictions()
        err = np.abs(d.explore_y - p)
        top = np.argsort(-err)[: self.cfg.brain.n_examples // 2]
        rest = np.setdiff1d(np.arange(len(d.explore_y)), top)
        rand = np.random.default_rng(iteration).choice(
            rest, size=min(len(rest), self.cfg.brain.n_examples - len(top)), replace=False)

        def show(i: int) -> str:
            t = d.explore_text.iloc[i].replace("\n", " ")[: self.cfg.brain.example_chars]
            return f"  [y={d.explore_y[i]} | pred=0:{1 - p[i]:.2f}, 1:{p[i]:.2f}] {t}"

        return {
            "iteration": iteration, "task_description": self.cfg.data.task_description,
            "metric": ev.metric, "baseline": self.baseline,
            "current_perf": ev.perf(sel), "current_obj": ev.objective(sel),
            "selected_table": "\n".join(sel_lines), "ideas_table": self.ideas_table(),
            "ngram_hints": self.hints,
            "residual_examples": "\n".join(show(i) for i in top),
            "random_examples": "\n".join(show(i) for i in rand),
            "history": [round(h["objective"], 4) for h in self.history],
            "base_note": (
                "All scores are measured ON TOP OF a TF-IDF word model (the baseline is TF-IDF alone), and the "
                "residual examples are cases that TF-IDF plus the selected policies get wrong. A policy only helps "
                "if it captures something word statistics miss: judgement, context, combinations, or reading "
                "between the lines, not the presence of particular words, titles or names."
                if self.base_sel is not None else ""),
        }

    @staticmethod
    def _spec(p: Policy) -> dict[str, str]:
        return {"idea": p.idea, "policy": p.policy, "question": p.question}

    def ideas_table(self) -> str:
        """Per idea: policies written, promoted to full evaluation, selected now, best objective gain."""
        rows: dict[str, dict[str, Any]] = {}
        for p in self.policies.values():
            r = rows.setdefault(p.idea, {"n": 0, "promoted": 0, "selected": 0, "gain": None, "it": p.iteration})
            r["n"] += 1
            r["promoted"] += p.name in self.pool
            r["selected"] += p.name in self.selected
            g = p.stats.get("gain")
            if g is not None and (r["gain"] is None or g > r["gain"]):
                r["gain"] = g
        return "\n".join(
            f"  [iter {r['it']}] {idea} | {r['n']} | {r['promoted']} | {r['selected']} | "
            + ("n/a" if r["gain"] is None else f"{r['gain']:+.4f}")
            for idea, r in list(rows.items())[-40:])

    # ------------------------------------------------------------ one iteration
    def new_policies(self, proposals: list[tuple[dict[str, str], str]], iteration: int) -> list[Policy]:
        known = {p.key() for p in self.policies.values()}
        out = []
        for j, (idea, text) in enumerate(proposals):
            name_idea = str(idea["idea"]).strip()
            stem = re.sub(r"[^a-z0-9]+", "_", name_idea.lower()).strip("_")[:32] or "idea"
            name = f"pol_{stem}_{iteration}_{j}"
            while name in self.policies:
                name += "x"
            p = Policy(name=name, idea=name_idea, policy=text, rationale=str(idea.get("rationale", "")),
                       iteration=iteration,
                       question=self.cfg.jev.policy_template.format(policy=text, task=self.cfg.data.task_description))
            if not 10 <= len(text) <= 500 or p.key() in known:
                continue
            known.add(p.key())
            self.policies[name] = p
            out.append(p)
        return out

    def screen(self, p: Policy, values: np.ndarray, resid: np.ndarray, y: np.ndarray) -> float | None:
        """Cheap gate before full extraction. Returns a priority, or None if screened out."""
        rows = self.data.screen_idx
        sig_r, sig_y = max_abs_corr(values, resid[rows]), max_abs_corr(values, y[rows])
        pool_corr = max(((max_abs_corr(values, self.ev.frame[n].to_numpy(float)[rows]), n) for n in self.pool),
                        default=(0.0, ""))
        sel_corr = max(((max_abs_corr(values, self.ev.frame[n].to_numpy(float)[rows]), n) for n in self.selected),
                       default=(0.0, ""))
        p.stats.update(screen_resid_corr=sig_r, screen_target_corr=sig_y, screen_max_pool_corr=pool_corr[0])
        sc = self.cfg.search
        if np.nanstd(values) < 1e-6:
            p.status, p.notes = "screened_out", ["constant on screen sample"]
        elif pool_corr[0] > sc.max_pool_corr:
            p.status, p.notes = "screened_out", [f"near-duplicate of {pool_corr[1]} (|corr|={pool_corr[0]:.2f})"]
        elif sel_corr[0] > sc.max_selected_corr:
            p.status, p.notes = "screened_out", [f"redundant with selected {sel_corr[1]} (|corr|={sel_corr[0]:.2f})"]
        elif max(sig_r, 0.5 * sig_y) < sc.min_screen_signal:
            p.status, p.notes = "screened_out", [f"weak signal (resid corr {sig_r:.3f})"]
        print(f"  {p.name:46s} resid_corr={sig_r:.3f} y_corr={sig_y:.3f} pool_corr={pool_corr[0]:.2f} "
              f"{'' if p.status == 'candidate' else '-> ' + p.notes[0]}")
        return sig_r + 0.25 * sig_y if p.status == "candidate" else None

    def step(self, iteration: int) -> dict[str, Any]:
        t0, ev, d, sc = time.time(), self.ev, self.data, self.cfg.search
        thinking, proposals = self.brain.propose(self.context(iteration))
        print(f"\n[iter {iteration}] brain: {thinking[:400]}", flush=True)
        cands = self.new_policies(proposals, iteration)

        # screen on a small sample of select rows
        resid = ev.residuals(self.selected)
        screen_vals = self.jev.score(cands, d.select_text.iloc[d.screen_idx])
        scored = []
        for p in cands:
            s = self.screen(p, screen_vals[p.name].to_numpy(float), resid, d.select_y)
            if s is not None:
                scored.append((s, p))
        scored.sort(key=lambda x: -x[0])
        promoted = [p for _, p in scored[: sc.promote_per_iter]]
        for _, p in scored[sc.promote_per_iter:]:
            p.status, p.notes = "screened_out", ["not in top promoted this iteration"]

        # promote: score on every train row (select rows for scoring, explore rows for the brain's examples)
        obj_before = ev.objective(self.selected)
        if promoted:
            full = self.jev.score(promoted, d.train_text)
            n_sel = len(d.select_y)
            for p in promoted:
                ev.add(p.name, full[p.name].to_numpy(float)[:n_sel])
                self.explore_frame[p.name] = full[p.name].to_numpy(float)[n_sel:]
                self.pool.append(p.name)
                p.status = "pool"
            for p in promoted:
                p.stats["solo_perf"] = ev.perf([p.name])
                p.stats["gain"] = ev.objective(self.selected + [p.name]) - obj_before

        # select over everything evaluated so far
        new_sel = ev.select(self.pool, self.selected, sc.max_features)
        for n in self.pool:
            p = self.policies[n]
            if n in new_sel:
                p.status = "selected"
            elif p.status == "selected" or p.iteration == iteration:
                p.status = "rejected"
                p.notes.append("dropped by selection" if n in self.selected
                               else f"no objective gain (gain={p.stats.get('gain', 0):+.4f})")
        self.selected = new_sel
        rec = {"iteration": iteration, "proposed": len(proposals), "promoted": [p.name for p in promoted],
               "selected": list(new_sel), "perf": ev.perf(new_sel), "objective": ev.objective(new_sel),
               "redundancy": ev.redundancy(new_sel), "objective_before": obj_before,
               "seconds": round(time.time() - t0, 1), "brain_thinking": thinking}
        self.history.append(rec)
        print(f"  => selected {len(new_sel)}: {new_sel}\n  => perf={rec['perf']:.4f} "
              f"objective={rec['objective']:.4f} (was {obj_before:.4f})", flush=True)
        return rec

    # ------------------------------------------------------------ run
    def run(self) -> list[str]:
        sc = self.cfg.search
        best_obj, stale = self.ev.objective([]), 0
        for it in range(1, sc.iterations + 1):
            try:
                rec = self.step(it)
            except Exception as e:  # noqa: BLE001 - log and continue with the next iteration
                print(f"  !! iteration {it} failed: {type(e).__name__}: {e}", flush=True)
                stale += 1
                continue
            finally:
                self.save()
            if rec["objective"] > best_obj + self.cfg.eval.min_gain:
                best_obj, stale = rec["objective"], 0
            else:
                stale += 1
            if stale >= sc.patience:
                print(f"\nno improvement for {stale} iterations, stopping.")
                break
        return self.selected

    def save(self, extra: dict[str, Any] | None = None) -> None:
        state = {"config": self.cfg.to_dict(), "selected": self.selected, "pool": self.pool,
                 "policies": [p.to_dict() for p in self.policies.values()], "history": self.history,
                 "usage": {"jev": self.jev.usage(), "brain": self.brain.usage()}, **(extra or {})}
        (self.run_dir / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False, default=float))
