"""Offline scorer for PyOpt model outputs. No GPU, standard library only.

Scores a JSONL file of model outputs (one row per test example, with ``id`` and
``model_output``) against the test split it was generated from:

- valid:     the extracted code defines the expected function
- correct:   same output as the original on the notebook's benchmark inputs
             and on the dataset generator's edge cases
- exact:     extracted code equals the reference optimization
- unchanged: for ``no_change`` rows, the model returned the original unchanged
- leaked:    the test prompt is a renamed copy of a training prompt
             (only when --train is given)
- speedup:   original time / candidate time on the largest benchmark input
             (only with --timing; machine-dependent). ``--timing notebook``
             reproduces the notebook method (timeit, 100 calls, deepcopy of the
             arguments inside the timed call); ``--timing clean`` times only the
             function call (min of 5 repeats of 50 calls).

Usage:
  python evaluate.py --test data/v1.1/pyopt_test.jsonl --train data/v1.1/pyopt_train.jsonl \
      results/v1.1/base_outputs.jsonl results/v1.1/finetuned_outputs.jsonl [--timing notebook]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import timeit
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from build_pyopt_dataset import cases_for, normalized_original

FASTER_THRESHOLD = 1.05  # same threshold as the notebook ("meaningfully faster")


def extract_code(text: str) -> str:
    """First fenced code block, or the whole text if there is none (as in the notebook)."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def benchmark_cases(category: str) -> List[tuple]:
    """The notebook's benchmark inputs; the last one is the timed case."""
    if category == "membership":
        return [(list(range(100)), list(range(50, 150))), (list(range(1000)), list(range(500, 1500)))]
    if category == "repeated_sum":
        return [(list(range(1, 101)),), (list(range(1, 1001)),)]
    if category == "repeated_sort":
        return [(list(range(100, 0, -1)), 20), (list(range(1000, 0, -1)), 100)]
    if category in {"list_concat", "no_change"}:
        return [(list(range(100)),), (list(range(1000)),)]
    if category == "string_concat":
        return [(["abc"] * 100,), (["abc"] * 1000,)]
    raise ValueError(category)


def _load_function(code: str, name: str) -> Optional[Callable[..., Any]]:
    # Model output is executed here. Only run this on outputs you trust to be
    # harmless (these are short pure functions from our own experiment).
    namespace: Dict[str, Any] = {}
    exec(code, namespace)
    fn = namespace.get(name)
    return fn if callable(fn) else None


def _time(fn: Callable[..., Any], args: tuple, mode: str) -> float:
    if mode == "notebook":
        return timeit.timeit(lambda: fn(*copy.deepcopy(args)), number=100)
    # No family mutates its inputs, so the arguments are built once, outside the timer.
    return min(timeit.repeat(lambda: fn(*args), number=50, repeat=5))


def score_row(
    example: Dict[str, Any],
    output_text: str,
    *,
    timing: Optional[str] = None,
) -> Dict[str, Any]:
    code = extract_code(output_text)
    row: Dict[str, Any] = {
        "id": example["id"],
        "category": example["category"],
        "valid": False,
        "correct": False,
        "exact": code == example["optimized"].strip(),
        "unchanged": code == example["original"].strip(),
        "speedup": None,
        "error": None,
    }
    try:
        candidate = _load_function(code, example["fn"])
        if candidate is None:
            row["error"] = "function missing"
            return row
        row["valid"] = True
        original = _load_function(example["original"], example["fn"])
        assert original is not None
        cases = benchmark_cases(example["category"]) + list(cases_for(example["category"]))
        for args in cases:
            if candidate(*copy.deepcopy(args)) != original(*copy.deepcopy(args)):
                row["error"] = f"output mismatch on {args!r:.60}"
                return row
        row["correct"] = True
        if timing:
            args = benchmark_cases(example["category"])[-1]
            row["speedup"] = _time(original, args, timing) / _time(candidate, args, timing)
    except Exception as exc:  # broken model code counts as invalid, not a crash
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    no_change = [r for r in rows if r["category"] == "no_change"]
    speedups = [r["speedup"] for r in rows if r["correct"] and r["speedup"] is not None]
    summary: Dict[str, Any] = {
        "n": len(rows),
        "valid": sum(r["valid"] for r in rows),
        "correct": sum(r["correct"] for r in rows),
        "exact": sum(r["exact"] for r in rows),
        "no_change_n": len(no_change),
        "no_change_kept": sum(r["unchanged"] for r in no_change),
    }
    if speedups:
        summary["faster"] = sum(s > FASTER_THRESHOLD for s in speedups)
        summary["median_speedup"] = round(statistics.median(speedups), 2)
    return summary


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    test_path: Path,
    output_paths: List[Path],
    train_path: Optional[Path] = None,
    *,
    timing: Optional[str] = None,
) -> Dict[str, Any]:
    test = {row["id"]: row for row in load_jsonl(test_path)}
    leaked_ids = set()
    if train_path is not None:
        train_keys = {normalized_original(row) for row in load_jsonl(train_path)}
        leaked_ids = {i for i, row in test.items() if normalized_original(row) in train_keys}

    report: Dict[str, Any] = {"test": str(test_path), "leaked_ids": sorted(leaked_ids), "models": {}}
    for path in output_paths:
        outputs = load_jsonl(path)
        missing = set(test) - {o["id"] for o in outputs}
        if missing:
            raise ValueError(f"{path}: no output for {sorted(missing)}")
        rows = [score_row(test[o["id"]], o["model_output"], timing=timing) for o in outputs]
        entry = {"all": summarize(rows), "rows": rows}
        if train_path is not None:
            entry["clean"] = summarize(r for r in rows if r["id"] not in leaked_ids)
            entry["leaked"] = summarize(r for r in rows if r["id"] in leaked_ids)
        report["models"][Path(path).stem] = entry
    return report


def _fmt(summary: Dict[str, Any]) -> str:
    text = (
        f"correct {summary['correct']}/{summary['n']}  exact {summary['exact']}/{summary['n']}  "
        f"no_change kept {summary['no_change_kept']}/{summary['no_change_n']}"
    )
    if "median_speedup" in summary:
        text += f"  faster {summary['faster']}/{summary['n']}  median {summary['median_speedup']}x"
    return text


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score PyOpt model outputs offline.")
    parser.add_argument("outputs", nargs="+", type=Path, help="model output JSONL files")
    parser.add_argument("--test", type=Path, required=True, help="test split the outputs answer")
    parser.add_argument("--train", type=Path, help="training split, to flag leaked test prompts")
    parser.add_argument(
        "--timing",
        choices=["notebook", "clean"],
        help="also measure speedups (machine-dependent): notebook method or call-only",
    )
    parser.add_argument("--json", type=Path, help="write the full report (with per-row scores)")
    args = parser.parse_args(argv)

    report = evaluate(args.test, args.outputs, args.train, timing=args.timing)
    if args.train is not None:
        print(f"leaked test prompts: {len(report['leaked_ids'])}/{len(load_jsonl(args.test))}")
    for name, entry in report["models"].items():
        print(f"\n{name}")
        for part in ("all", "clean", "leaked"):
            if part in entry:
                print(f"  {part:<7} {_fmt(entry[part])}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
