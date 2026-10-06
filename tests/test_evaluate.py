"""Offline scorer: unit checks plus the published v1.1 numbers that do not depend on timing."""

import evaluate as ev
from build_pyopt_dataset import REPO_DIR

V11 = REPO_DIR / "data" / "v1.1"
RESULTS = REPO_DIR / "results" / "v1.1"

EXAMPLE = {
    "id": "membership-x",
    "category": "membership",
    "fn": "keep",
    "original": "def keep(xs, ok):\n    out = []\n    for x in xs:\n        if x in ok:\n            out.append(x)\n    return out\n",
    "optimized": "def keep(xs, ok):\n    s = set(ok)\n    return [x for x in xs if x in s]",
}


def test_extract_code_takes_first_fenced_block():
    assert ev.extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1"
    assert ev.extract_code("  y = 2  ") == "y = 2"


def test_correct_candidate_scores_correct_and_exact():
    row = ev.score_row(EXAMPLE, "```python\n" + EXAMPLE["optimized"] + "\n```")
    assert row["valid"] and row["correct"] and row["exact"]


def test_wrong_candidate_is_valid_but_not_correct():
    wrong = "def keep(xs, ok):\n    return list(ok)"
    row = ev.score_row(EXAMPLE, wrong)
    assert row["valid"] and not row["correct"]


def test_broken_or_missing_function_is_invalid():
    assert not ev.score_row(EXAMPLE, "def other():\n    pass")["valid"]
    assert not ev.score_row(EXAMPLE, "def keep(:")["valid"]


def test_published_v11_scores():
    report = ev.evaluate(
        V11 / "pyopt_test.jsonl",
        [RESULTS / "base_outputs.jsonl", RESULTS / "finetuned_outputs.jsonl"],
        V11 / "pyopt_train.jsonl",
    )
    assert len(report["leaked_ids"]) == 13
    base, tuned = report["models"]["base_outputs"], report["models"]["finetuned_outputs"]

    assert (base["all"]["correct"], tuned["all"]["correct"]) == (40, 40)
    assert (base["all"]["exact"], tuned["all"]["exact"]) == (7, 31)
    assert (base["all"]["no_change_kept"], tuned["all"]["no_change_kept"]) == (0, 5)

    assert (base["clean"]["n"], base["clean"]["correct"], tuned["clean"]["correct"]) == (27, 27, 27)
    assert (base["clean"]["exact"], tuned["clean"]["exact"]) == (2, 20)
    assert (base["clean"]["no_change_kept"], tuned["clean"]["no_change_kept"]) == (0, 3)
