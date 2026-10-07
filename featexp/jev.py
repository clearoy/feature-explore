"""Policies and their Jev (TypeSafe System One) scores, with a persistent SQLite cache.

A policy is one investor-style heuristic. Rendered into `jev.policy_template`, it becomes one
yes/no question; Jev's P(yes) for a text is that policy's feature value. Several pending
questions for the same text go into one request, answers are cached by
(policy hash, text hash, Jev version), and answers lost to errors get one more pass.

The Jev version is pinned: an alias such as `jev-latest` is resolved to the concrete version
(e.g. `jev-1.13.0`) before the first lookup, every request names that version, and an answer
from any other version is an error. A model trained on one version's scores is then never fed
another version's scores, and `predict` reuses the version saved with the run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FULL_VERSION = re.compile(r"jev-\d+\.\d+\.\d+")

import numpy as np
import pandas as pd

from .config import JevConfig


@dataclass
class Policy:
    name: str
    idea: str
    policy: str
    question: str                      # the policy rendered into the Jev template
    rationale: str = ""
    iteration: int = 0
    status: str = "candidate"          # candidate | screened_out | pool | selected | rejected
    notes: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    def key(self) -> str:
        """Cache key; identical to the original feature_discover repo, so its cache can be reused."""
        spec = {"idea": self.idea, "policy": self.policy, "question": self.question}
        payload = json.dumps({"kind": "jev_policy", "spec": spec}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Policy":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


class JevCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=120)   # several runs may share one cache
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS answers (fkey TEXT, thash TEXT, model TEXT, value TEXT,"
            " PRIMARY KEY (fkey, thash, model))"
        )

    def get_many(self, fkey: str, hashes: list[str], model: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for i in range(0, len(hashes), 900):
            chunk = hashes[i:i + 900]
            q = f"SELECT thash, value FROM answers WHERE fkey=? AND model=? AND thash IN ({','.join('?' * len(chunk))})"
            for th, v in self.db.execute(q, [fkey, model, *chunk]):
                out[th] = json.loads(v)[0]
        return out

    def put_many(self, rows: list[tuple[str, str, str, float]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)",
                            [(fk, th, m, json.dumps([v])) for fk, th, m, v in rows])
        self.db.commit()


def mock_answer(p: Policy, text: str) -> float:
    """Deterministic offline stand-in: word overlap between the policy and the text, plus noise."""
    q_words = set(re.findall(r"[a-z]{4,}", (p.idea + " " + p.policy).lower()))
    t_words = set(re.findall(r"[a-z]{4,}", text.lower()))
    overlap = len(q_words & t_words) / (len(q_words) ** 0.5 + 1)
    noise = int(hashlib.md5((p.key() + text).encode()).hexdigest()[:6], 16) / 0xFFFFFF
    return float(1 / (1 + np.exp(-(2.5 * overlap - 1.5 + (noise - 0.5)))))


class VersionMismatch(RuntimeError):
    pass


class JevScorer:
    def __init__(self, cfg: JevConfig, cache_path: str | Path, mock: bool = False):
        self.cfg = cfg
        self.cache = JevCache(cache_path)
        self.mock = mock
        self.version: str | None = "mock" if mock else (cfg.model if FULL_VERSION.fullmatch(cfg.model) else None)
        self.requests = self.input_tokens = self.output_tokens = self.failures = 0

    def resolve_version(self) -> str:
        """Pin an alias to the concrete Jev version with one tiny request."""
        if self.version is None:
            from typesafe_sdk import Noul, TypeSafeClient

            with TypeSafeClient(model=self.cfg.model, timeout=self.cfg.timeout) as client:
                resp = client.system_one(state="ping", questions={"q": Noul(instructions="Is this text non-empty?")})
            self.version = resp.model
            print(f"    jev: {self.cfg.model} pinned to {self.version}", flush=True)
        return self.version

    def score(self, policies: list[Policy], texts: pd.Series) -> pd.DataFrame:
        """P(yes) for every (text, policy), as a DataFrame aligned to texts.index with one column per policy."""
        if not policies:
            return pd.DataFrame(index=texts.index)
        version = self.resolve_version()
        hashes = [text_hash(t) for t in texts]
        cached = {p.name: self.cache.get_many(p.key(), list(set(hashes)), version) for p in policies}

        pending: dict[str, tuple[str, list[Policy]]] = {}
        for th, t in zip(hashes, texts):
            missing = [p for p in policies if th not in cached[p.name]]
            if missing and th not in pending:
                pending[th] = (t, missing)
        for attempt in range(2):                      # second pass only for answers lost to errors
            if not pending:
                break
            n = sum(len(m) for _, m in pending.values())
            print(f"    jev: {'retrying ' if attempt else ''}{n} answers for {len(pending)} texts "
                  f"({len(policies)} policies)", flush=True)
            for (name, th), v in asyncio.run(self._fetch(pending)).items():
                cached[name][th] = v
            pending = {th: (t, [p for p in ps if th not in cached[p.name]]) for th, (t, ps) in pending.items()}
            pending = {th: v for th, v in pending.items() if v[1]}

        return pd.DataFrame({p.name: [cached[p.name].get(th, np.nan) for th in hashes] for p in policies},
                            index=texts.index)

    async def _fetch(self, pending: dict[str, tuple[str, list[Policy]]]) -> dict[tuple[str, str], float]:
        per = max(1, self.cfg.questions_per_request)
        jobs = [(th, t, ps[i:i + per]) for th, (t, ps) in pending.items() for i in range(0, len(ps), per)]
        results: dict[tuple[str, str], float] = {}
        buffer: list[tuple[str, str, str, float]] = []
        done = 0

        def record(th: str, p: Policy, v: float) -> None:
            results[(p.name, th)] = v
            buffer.append((p.key(), th, self.version, v))

        def progress() -> None:
            nonlocal done
            done += 1
            if len(buffer) >= 200:
                self.cache.put_many(buffer)
                buffer.clear()
            if done % 50 == 0 or done == len(jobs):
                print(f"\r    jev: {done}/{len(jobs)} requests", end="", file=sys.stderr, flush=True)

        if self.mock:
            for th, t, ps in jobs:
                for p in ps:
                    record(th, p, mock_answer(p, t))
                progress()
        else:
            from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy

            sem = asyncio.Semaphore(self.cfg.concurrency)
            retry = RetryPolicy(max_retries=self.cfg.max_retries, timeout=self.cfg.timeout)
            async with AsyncTypeSafeClient(model=self.version, retry=retry, timeout=self.cfg.timeout) as client:
                async def worker(th: str, t: str, ps: list[Policy]) -> None:
                    questions = {f"q{i}": Noul(instructions=p.question) for i, p in enumerate(ps)}
                    try:
                        async with sem:
                            resp = await client.system_one(state=t, questions=questions)
                        if resp.model != self.version:
                            raise VersionMismatch(f"Jev answered with {resp.model}, pinned {self.version}")
                        self.requests += 1
                        self.input_tokens += resp.usage.input_tokens or 0
                        self.output_tokens += resp.usage.output_tokens or 0
                        for i, p in enumerate(ps):
                            ans = resp.answers.get(f"q{i}")
                            if ans is not None:
                                record(th, p, float(ans.noul))
                    except VersionMismatch:
                        raise
                    except Exception as e:  # noqa: BLE001 - one bad request must not kill the run
                        self.failures += 1
                        if self.failures <= 5:
                            print(f"\n    jev request failed: {type(e).__name__}: {e}", file=sys.stderr)
                    progress()

                await asyncio.gather(*(worker(*job) for job in jobs))

        if buffer:
            self.cache.put_many(buffer)
        print(file=sys.stderr)
        return results

    def usage(self) -> dict[str, int]:
        return {"requests": self.requests, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "failures": self.failures}
