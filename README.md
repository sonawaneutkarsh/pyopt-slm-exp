# PyOpt: LoRA fine-tuning Qwen3-4B to optimize Python functions

[![CI](https://github.com/sonawaneutkarsh/pyopt-slm-exp/actions/workflows/ci.yml/badge.svg)](https://github.com/sonawaneutkarsh/pyopt-slm-exp/actions/workflows/ci.yml)

A small, controlled experiment (September 2026): does a 4B instruct model,
LoRA-tuned on 160 verified "make this function faster" pairs, write faster and
still-correct Python than the base model? Outputs are scored by **running** them
against the original function, not by string matching.

## Results

Setup: `unsloth/Qwen3-4B-Instruct-2507` in 4-bit, LoRA r=32 / alpha=32 on all
attention and MLP projections (66.1M of 4.09B parameters, 1.62%), 2 epochs
(40 steps), effective batch 8, lr 2e-4, loss on the answer only, Kaggle 2× T4,
greedy decoding. Training loss went from 0.758 (step 1) to 0.027 (step 40).

**Published run, 40-prompt test split (v1.1)**. Numbers from the saved notebook
outputs, timed on the Kaggle host:

| Metric | Base Qwen3-4B | + LoRA |
|---|---|---|
| Valid, correct code | 40/40 | 40/40 |
| Faster than the original (speedup > 1.05×) | 29/40 | 36/40 |
| Median speedup | 3.57× | 5.65× |
| Already-optimal code left unchanged (`no_change`) | 0/5 | 5/5 |
| Identical to the reference optimization¹ | 7/40 | 31/40 |

Most of the speedup gain comes from one family, `membership` (list lookup → set
lookup): median 1.01× → 11.58×. The other families barely moved.

**Out-of-distribution probe** (10 hand-written problems from 5 families not in the
training data): base **9/10** correct, LoRA **8/10**. The tuned model reused
training idioms (for example an `allowed_lookup` set) where they did not fit and
broke two functions. Narrow synthetic data taught the template, not general
optimization skill.

> ⚠️ **Train/test leakage in the published run.** 13 of the 40 test prompts in
> the v1.1 split are renamed copies of training prompts (identical code apart
> from the function name). The in-distribution numbers above are therefore **not
> a clean held-out evaluation**. The table below re-scores the same saved outputs
> on the 27 prompts that have no copy in the training set.

**Re-scored on the 27 clean test prompts** (`python evaluate.py`, offline):

| Metric | Base | + LoRA |
|---|---|---|
| Correct | 27/27 | 27/27 |
| Identical to the reference¹ | 2/27 | 20/27 |
| `no_change` left unchanged | 0/3 | 3/3 |
| Faster than the original² | 19/27 | 26/27 |

The pattern holds on the clean prompts, so leakage does not explain it. But the
"clean" prompts still come from the same six templates with different variable
names, so this measures how well the model learned the templates. The
out-of-distribution probe is the better test of generalization, and there the
tuned model was slightly worse.

¹ From `evaluate.py`; the notebook did not compute it.
² Re-measured with the notebook's timing method on a different machine (aarch64
Linux, Python 3.12). Timings vary by machine and run: the same method gave 28/40
and 38/40 on all 40 prompts here versus 29/40 and 36/40 on Kaggle. Speedup
magnitudes are not comparable across machines.

**Fixed for the next run.** The dataset generator now builds a leak-free v1.2
split (`data/`): examples whose code is identical apart from the function name
always land in the same split (0 test or validation prompts duplicate a training
prompt; CI checks this). The model has **not** been retrained on v1.2 yet, because
that needs a GPU. Until then, treat the published in-distribution numbers as
affected by leakage.

## Dataset

230 Python functions in six families. Each reference optimization is executed
against the original on edge cases before anything is written, and generation is
seeded, so a rebuild is byte-identical.

| Family | Optimization | Train / val / test |
|---|---|---|
| `membership` | repeated list membership → set lookup | 28 / 5 / 7 |
| `repeated_sum` | hoist an invariant `sum()` out of the loop | 28 / 5 / 7 |
| `repeated_sort` | sort once instead of every iteration | 28 / 5 / 7 |
| `list_concat` | `result = result + [x]` → `append` | 28 / 5 / 7 |
| `string_concat` | `+=` in a loop → `"".join` | 28 / 5 / 7 |
| `no_change` | already optimal; the right answer is to leave it alone | 20 / 5 / 5 |

```bash
python build_pyopt_dataset.py                          # v1.2 split -> data/
python build_pyopt_dataset.py --split v1.1 --out data/v1.1   # reproduce the published split
```

## Reproduce

```bash
pip install -r requirements.txt     # pytest only; the scripts use the standard library
pytest                              # rebuild check, leakage check, scorer tests
python evaluate.py --test data/v1.1/pyopt_test.jsonl --train data/v1.1/pyopt_train.jsonl \
    results/v1.1/base_outputs.jsonl results/v1.1/finetuned_outputs.jsonl [--timing notebook]
```

Training needs a CUDA GPU: install `requirements-train.txt` (the versions of the
recorded run) and run `pyopt.ipynb`. The notebook reads `PYOPT_DATA_DIR` (default
`data`, the v1.2 split) and writes to `PYOPT_OUT_DIR` (default `results/v1.2`).
Its saved outputs are from the original v1.1 run; it was cleaned afterwards but
not re-executed.

## Layout

```
build_pyopt_dataset.py   dataset generator (v1.2 group-aware split, v1.1 for reproduction)
evaluate.py              offline scorer: correctness, exact match, leakage flag, optional timing
pyopt.ipynb              training + evaluation notebook (saved outputs from the v1.1 run)
data/                    v1.2 split (leak-free); data/v1.1/ = split used by the published run
results/v1.1/            base and fine-tuned outputs, training loss, local re-score
tests/                   pytest suite run in CI (CPU only)
```

## Limitations

- 230 synthetic examples from six templates; the test set is tiny.
- The published run used the leaky v1.1 split (see above). No v1.2 run yet.
- Single training run, one seed, no validation loss logged.
- Speedups are single-machine `timeit` measurements and include argument copying.
- The out-of-distribution probe checks correctness only; its speedup was never computed.
- The LoRA adapter is not published.
