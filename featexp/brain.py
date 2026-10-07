"""The brain (DeepSeek, OpenAI-compatible API): step 1 proposes ideas, step 2 writes policies per idea."""

from __future__ import annotations

import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import BrainConfig

IDEA_SYSTEM = """You are the research lead of a feature-discovery project. Your job is to decide WHAT to look at next,
not how to measure it: name new semantic dimensions (ideas) along which the cases differ and which should
predict the target beyond what the current features already capture.

OBJECTIVE
  maximize  Perf_OOS(F)  with preference for a SMALLER and more DIVERSE (non-redundant) feature set F.

Each idea will be handed to a colleague who turns it into a few investor-style heuristics ("policies");
a judge model applies each policy to every case and a logistic regression weighs them.

Good ideas:
- are one coherent dimension (e.g. "depth of prior founding experience", "fit between career and startup sector"),
  not a bundle of unrelated signals
- are NEW: clearly different from the dimensions already covered by selected features and from ideas that failed
- are motivated by the cases the current model gets wrong (residual examples) and by domain knowledge
- can be judged from the text alone

Return ONLY JSON:
{"thinking": "what the current set misses", "ideas": [{"idea": "short name", "rationale": "why it should add signal beyond current features", "evidence": "what in the examples suggests it"}, ...]}"""


POLICY_SYSTEM = """You write investor heuristics for predicting an outcome. Each heuristic is one short sentence an
experienced investor would use to judge a case, e.g.
"Founders who previously built and sold a company are more likely to succeed."
A separate model applies each heuristic to every case, and a logistic regression learns how much to trust each one.

You are given ONE idea (a dimension to explore). Write heuristics that express that idea from different angles
(positive and negative direction, different strengths or facets), so the regression can find the useful one.

Good heuristics are:
- general: drawn from domain knowledge and true of many cases, not one sample
- focused: one signal each, all within the given idea
- clean: no quotes, names or exact numbers taken from the samples
- new: not a rewording of the existing selected heuristics

Return ONLY JSON: {"policies": ["...", ...]}"""


def _common_context(ctx: dict[str, Any]) -> list[str]:
    return [
        f"TASK: {ctx['task_description'] or '(no description given)'}",
        f"Target type: binary; metric: {ctx['metric']} (higher is better). "
        f"Baseline {ctx['baseline']:.4f}; current selected set {ctx['current_perf']:.4f}; objective J {ctx['current_obj']:.4f}.",
        "",
        "CURRENTLY SELECTED FEATURES (delta_perf = drop in OOS metric when removed):",
        ctx["selected_table"] or "  (none yet)",
    ]


def build_idea_prompt(ctx: dict[str, Any], n: int) -> str:
    return "\n".join(_common_context(ctx) + [
        "",
        "IDEAS TRIED SO FAR (idea | policies written | promoted | selected | best gain):",
        ctx["ideas_table"] or "  (none)",
        "",
        "STATISTICAL HINTS - n-grams most associated with the target (positive weight = higher y):",
        ctx["ngram_hints"],
        "",
        "RESIDUAL EXAMPLES - cases where the current model is most wrong:",
        ctx["residual_examples"],
        "",
        "RANDOM EXAMPLES:",
        ctx["random_examples"],
        "",
        f"History of objective by iteration: {ctx['history']}",
        "",
        f"Propose exactly {n} NEW ideas, each a different dimension, complementary to the selected features.",
    ])


def build_policy_prompt(ctx: dict[str, Any], idea: dict[str, str], k: int) -> str:
    return "\n".join(_common_context(ctx) + [
        "",
        f"IDEA TO EXPRESS: {idea.get('idea')}",
        f"Rationale: {idea.get('rationale', '')}",
        f"Evidence: {idea.get('evidence', '')}",
        "",
        "EXAMPLES (for grounding only; do not quote them):",
        ctx["residual_examples"],
        "",
        f"Write {k} heuristics expressing this idea.",
    ])


def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


class Brain:
    def __init__(self, cfg: BrainConfig, mock: bool = False):
        self.cfg = cfg
        self.mock = mock
        self.input_tokens = self.output_tokens = 0
        if not mock:
            from openai import OpenAI

            key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if not key:
                raise RuntimeError("DEEPSEEK_API_KEY is not set (put it in .env)")
            self.client = OpenAI(api_key=key, base_url=cfg.base_url)

    def _chat(self, system: str, user: str) -> dict[str, Any]:
        """One JSON-mode chat call, up to 3 attempts at getting parseable JSON."""
        kwargs: dict[str, Any] = {}
        if "reasoner" not in self.cfg.model:
            kwargs = {"temperature": self.cfg.temperature, "response_format": {"type": "json_object"}}
        last_err: Exception | None = None
        for _ in range(3):
            resp = self.client.chat.completions.create(
                model=self.cfg.model, max_tokens=self.cfg.max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
            if resp.usage:
                self.input_tokens += resp.usage.prompt_tokens
                self.output_tokens += resp.usage.completion_tokens
            try:
                return parse_json(resp.choices[0].message.content or "")
            except (json.JSONDecodeError, AttributeError) as e:
                last_err = e
        raise RuntimeError(f"brain returned unparseable JSON 3 times: {last_err}")

    def propose_ideas(self, ctx: dict[str, Any], n: int) -> tuple[str, list[dict[str, str]]]:
        """Step 1: name new semantic dimensions worth exploring."""
        if self.mock:
            return mock_ideas(ctx, n)
        data = self._chat(IDEA_SYSTEM, build_idea_prompt(ctx, n))
        ideas = [i for i in data.get("ideas", []) if isinstance(i, dict) and str(i.get("idea", "")).strip()]
        return str(data.get("thinking", "")), ideas[:n]

    def write_policies(self, ctx: dict[str, Any], idea: dict[str, str], k: int) -> list[str]:
        """Step 2: turn one idea into k investor-style policies."""
        if self.mock:
            return mock_policies(idea, k)
        data = self._chat(POLICY_SYSTEM, build_policy_prompt(ctx, idea, k))
        return [str(p).strip() for p in data.get("policies", []) if str(p).strip()][:k]

    def propose(self, ctx: dict[str, Any]) -> tuple[str, list[tuple[dict[str, str], str]]]:
        """Both steps (ideas in parallel for step 2). Returns (thinking, [(idea, policy text), ...])."""
        thinking, ideas = self.propose_ideas(ctx, self.cfg.ideas_per_iter)

        def safe(idea: dict[str, str]) -> list[str]:
            try:
                return self.write_policies(ctx, idea, self.cfg.policies_per_idea)
            except Exception as e:  # noqa: BLE001 - one failed idea must not sink the iteration
                print(f"  ! policy generation failed for idea {idea.get('idea')!r}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=max(1, len(ideas))) as pool:
            results = list(pool.map(safe, ideas))
        summary = "; ".join(f"{i['idea']} ({len(p)} policies)" for i, p in zip(ideas, results))
        return f"{thinking}\n  ideas: {summary}", [(i, p) for i, ps in zip(ideas, results) for p in ps]

    def usage(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def mock_ideas(ctx: dict[str, Any], n: int) -> tuple[str, list[dict[str, str]]]:
    words = re.findall(r"^\s*[+-][\d.]+\s+(.+)$", ctx["ngram_hints"], re.M)
    rng = random.Random(ctx["iteration"])
    picks = rng.sample(words, min(n, len(words))) if words else ["tone"] * n
    return "mock ideas", [{"idea": f"signal around {w}", "rationale": "mock", "evidence": "mock"} for w in picks]


def mock_policies(idea: dict[str, str], k: int) -> list[str]:
    topic = idea["idea"].replace("signal around ", "")
    angles = ["Cases that mention {t} are more likely to be positive.",
              "Cases without any sign of {t} are less likely to be positive.",
              "Strong, repeated emphasis on {t} signals a positive outcome.",
              "{t} only matters when combined with urgency."]
    return [a.format(t=topic) for a in angles[:k]]
