# featexp — Idea → Policy Feature Discovery

Given raw text `X_text` and a binary target `y`, find a **small, complementary, interpretable** set of features (investor-style heuristics, "policies") and combine them with a TF-IDF text model:

$$\max_F \; Perf_{OOS}(F) \quad \text{with preference for smaller and more diverse } F$$

- **Brain: DeepSeek.** It proposes new dimensions to explore (*ideas*), then writes a few policies for each idea.
- **Judge: Jev (TypeSafe).** For every text it answers "considering this policy, will this case be positive?". That probability is the feature value.
- **Referee: cross-validation.** A policy is kept only if it improves out-of-sample performance without duplicating the policies already selected.
- **Final model:** the selected policies combined with TF-IDF (stacking, or a plain average of the two probabilities).

This is the best-performing approach from a series of experiments (direct feature generation with regex/python/combo features, explain → rules, explain → policies, and others). Only the code it needs is kept here.

## Results on VCBench

Founder success prediction: 4,500 public rows for discovery and 4,500 private rows used once as the holdout, with 9% positives. Mean ± sd over three seeds, holdout, in %:

| model | features | AUC | AP | F0.5 |
|---|---|---|---|---|
| TF-IDF alone | ~50k n-grams | 73.7 | 28.7 | 32.5 ± 1.1 |
| Selected policies alone | 1–8 policies | 71.6 ± 0.4 | 24.7 ± 1.5 | 28.3 ± 0.9 |
| **Average of TF-IDF and policies** | | **75.2 ± 0.2** | **31.2 ± 0.7** | **34.9 ± 0.6** |
| Stack: TF-IDF + policies | | 75.0 ± 0.2 | 31.0 ± 0.7 | 34.3 ± 1.2 |
| Reference: Jev + 201 policies (policy-induction) | 201 policies | 74.2 | – | 36.7 |

- The gain from adding policies to TF-IDF is consistent across seeds: about +1.5 AUC and +2.5 AP.
- The best single run reached stack AUC 75.7, AP 33.1 and F0.5 36.4, but that was a favourable draw. Use the means above.
- The number of selected policies varies a lot between seeds (1, 6 and 8), yet the combined result barely changes. Each policy feature carries Jev's holistic judgement of the whole profile, and that shared component is most of what TF-IDF lacks.
- F0.5 depends on a threshold fitted on training predictions and moves by a few points between runs. AUC and AP are the steadier comparison.

## Quick start

```bash
cd feature_explore
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in DEEPSEEK_API_KEY and TYPESAFE_API_KEY
python -m featexp check       # verify both keys
```

To exercise the whole pipeline offline at no cost (fake Jev and fake brain):

```bash
python -m featexp demo-data
python -m featexp run -c configs/demo.yaml --mock --name smoke
```

For VCBench, put `vcbench_final_public.csv` and `vcbench_final_private.csv` in `data/`, then:

```bash
scripts/run_vcbench.sh                          # one run, seed from the config (42)
scripts/run_vcbench.sh --seed 1 --concurrency 12
.venv/bin/python scripts/summarize.py runs/vcbench_seed*    # mean ± sd over finished runs
```

For your own data, copy `configs/vcbench.yaml` and edit two things:
- the `data` section: `path`, `test_path`, `text_col`, `target_col`, `positive_label`, `task_description`;
- `jev.policy_template`.

Then run:

```bash
python -m featexp run -c configs/my_data.yaml --name v1
```

To score new texts with a finished run:

```bash
python -m featexp predict -r runs/v1 -i data/new.csv -o runs/v1/scored.csv
```

In the output, `p_stack` is the recommended score, `p_average` is the simple average of the two models, and `predicted` is the 0/1 decision at the F0.5 threshold.

> Run it as `python -m featexp` rather than installing it with `pip install -e .`. Under `~/Documents`, macOS can mark the venv's `.pth` files as hidden, and Python 3.13+ skips hidden `.pth` files.

## How it works

```
train file ──┬── explore rows (default 1000): the only rows (and labels) the brain ever sees
             └── select rows (the rest):       every scoring and selection decision is made here
holdout     ──── used once, at the very end

each iteration (up to 12):
 ① ideas     DeepSeek is shown four things:
               - the selected policies and their contributions
               - how earlier ideas fared
               - n-gram hints
               - the explore rows the current model gets most wrong
             → it proposes 5 new dimensions
 ② policies  each idea becomes 3 investor heuristics (positive, negative, different strengths), in parallel
             → 15 candidates
 ③ screen    Jev scores the candidates on 600 select rows only.
             It keeps those that correlate with the current model's residual
             and drops those correlating > 0.9 with a selected policy
 ④ promote   the best 5 are scored on every training row
 ⑤ select    warm-started forward/backward selection over every policy evaluated so far, maximizing
             J = AUC_CV − 0.002·|F| − 0.005·mean|corr|   (5-fold × 2 repeats, select rows only)
 stop when J has not improved for 4 iterations, or after 12

final: train TF-IDF, a policy model, their average and their stack on all training rows.
       Thresholds come from out-of-fold predictions. The holdout is scored once.
```

## Outputs (`runs/<name>/`)

| file | contents |
|---|---|
| `report.md` | holdout table, selected policies with weights, search trajectory, every candidate with the reason it was dropped |
| `policies.json` | the selected policies, including the full Jev question |
| `model.joblib` | TF-IDF, policy model, stack and their thresholds (used by `predict`) |
| `state.json` | every candidate, per-iteration records, the brain's reasoning, usage |
| `holdout_predictions.csv` | holdout probabilities of all four models |
| `run.log` | full log when started through `scripts/run_vcbench.sh` |

Jev answers are cached in `.cache/jev.sqlite`, keyed by (policy, text, model), so a rerun never pays twice for the same answer. Runs started in parallel can share the cache.

## Key parameters

| parameter | default | effect |
|---|---|---|
| `search.promote_per_iter` | 5 | policies scored on every row per iteration; the **main cost knob** |
| `brain.ideas_per_iter` / `policies_per_idea` | 5 / 3 | candidates per iteration = their product |
| `brain.temperature` | 0.9 | DeepSeek sampling. The same seed gives different ideas each run; set 0 for repeatable runs |
| `eval.size_penalty` | 0.002 | cost per extra policy; raise it for fewer policies |
| `eval.redundancy_penalty` | 0.005 | policies share a ~0.5 baseline correlation, and 0.03 stalls the search at one policy |
| `search.max_selected_corr` | 0.9 | candidates correlating more than this with a selected policy are dropped at screening |
| `data.explore_rows` | 1000 | rows only the brain sees; they are never scored, which prevents leakage |
| `data.seed` / `--seed` | 42 | explore/select split, screening sample, CV folds |
| `jev.policy_template` | generic | change it to fit your task |

**Cost on VCBench with the default config:** about 35k–115k Jev requests per run (fewer when the search stops early) and about 200k DeepSeek tokens.

## Reading the weights

Every policy feature is "the probability of success given this heuristic", so all policies share an "overall promise" component. The regression cancels that component with weights of opposite sign, which makes **the weights conditional contrasts, not the direction stated in the heuristic**. For example, "founders with a PhD are more likely to succeed" can end up with a negative weight.

If you need weights that read literally, change the template so Jev only asks whether the statement is true for the founder. On VCBench that version predicts noticeably worse.

## Layout

```
featexp/
  config.py     configuration (defaults = the best VCBench settings)
  data.py       loading; explore / select / holdout split; n-gram hints
  jev.py        Policy; batched Jev scoring + SQLite cache + retry pass + mock
  brain.py      the two DeepSeek prompts (ideas, policies) + mock
  evaluator.py  repeated K-fold CV, objective J, forward/backward selection
  search.py     the iteration loop: ideas → policies → screen → promote → select
  final.py      final models (TF-IDF, policies, average, stack), holdout evaluation, report
  cli.py        run / predict / check / demo-data
configs/
  vcbench.yaml  the best VCBench configuration
  demo.yaml     offline smoke test
scripts/
  run_vcbench.sh   one full VCBench run (accepts --seed, --concurrency)
  summarize.py     holdout results of several runs, with mean ± sd
```
