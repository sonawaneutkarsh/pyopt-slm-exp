"""Build the PyOpt dataset: verified Python optimization pairs in six families.

Every reference optimization is executed against the original on edge cases
before anything is written. Generation is seeded, so a rebuild is byte-identical.

Splits:
  v1.2 (default) - group-aware split. Examples whose ``original`` code is identical
                   once the function name is ignored stay in the same split, so no
                   test prompt is a renamed copy of a training prompt.
  v1.1           - the original per-category random split, used for the published
                   fine-tuning run. 13 of its 40 test prompts are renamed copies of
                   training prompts. Kept only to reproduce data/v1.1/.

Usage:
  python build_pyopt_dataset.py                       # v1.2 -> data/
  python build_pyopt_dataset.py --split v1.1 --out data/v1.1
"""

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 3407
REPO_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = REPO_DIR / "data"

# Per-category split sizes: (train, val, test).
SPLIT_SIZES = {"no_change": (20, 5, 5)}
DEFAULT_SPLIT_SIZES = (28, 5, 7)

NAME_POOLS = {
    "item": ["item", "x", "value", "entry", "number"],
    "items": ["items", "values", "numbers", "entries", "data"],
    "allowed": ["allowed", "valid", "choices", "accepted", "pool"],
    "result": ["result", "out", "output", "answer", "acc"],
    "values": ["values", "numbers", "data", "items", "xs"],
    "total": ["total", "s", "sum_value", "denom"],
    "k": ["k", "count", "limit", "n"],
    "i": ["i", "idx", "pos", "j"],
    "ordered": ["ordered", "sorted_values", "ranked", "arranged"],
    "word": ["word", "part", "piece", "text"],
    "words": ["words", "parts", "pieces", "chunks"],
}

def make_code(fn, category, vars_):
    if category == "membership":
        item, items, allowed, result = vars_
        original = f"""def {fn}({items}, {allowed}):
    {result} = []
    for {item} in {items}:
        if {item} in {allowed}:
            {result}.append({item})
    return {result}
"""
        optimized = f"""def {fn}({items}, {allowed}):
    allowed_lookup = set({allowed})
    {result} = []
    for {item} in {items}:
        if {item} in allowed_lookup:
            {result}.append({item})
    return {result}
"""
        constraints = f"{items} and {allowed} are lists of integers."

    elif category == "repeated_sum":
        x, values, result, total = vars_
        original = f"""def {fn}({values}):
    {result} = []
    for {x} in {values}:
        if sum({values}) == 0:
            {result}.append(0.0)
        else:
            {result}.append({x} / sum({values}))
    return {result}
"""
        optimized = f"""def {fn}({values}):
    {total} = sum({values})
    {result} = []
    for {x} in {values}:
        if {total} == 0:
            {result}.append(0.0)
        else:
            {result}.append({x} / {total})
    return {result}
"""
        constraints = f"{values} is a list of integers."

    elif category == "repeated_sort":
        values, k, result, i, ordered = vars_
        original = f"""def {fn}({values}, {k}):
    if {k} <= 0:
        return []
    {result} = []
    for {i} in range(min({k}, len({values}))):
        {result}.append(sorted({values})[{i}])
    return {result}
"""
        optimized = f"""def {fn}({values}, {k}):
    if {k} <= 0:
        return []
    {ordered} = sorted({values})
    return {ordered}[:min({k}, len({values}))]
"""
        constraints = f"{values} is a list of integers and {k} is an integer."

    elif category == "list_concat":
        x, values, result = vars_
        original = f"""def {fn}({values}):
    {result} = []
    for {x} in {values}:
        {result} = {result} + [{x} * {x}]
    return {result}
"""
        optimized = f"""def {fn}({values}):
    {result} = []
    for {x} in {values}:
        {result}.append({x} * {x})
    return {result}
"""
        constraints = f"{values} is a list of integers."

    elif category == "string_concat":
        word, words, result = vars_
        original = f"""def {fn}({words}):
    {result} = ""
    for {word} in {words}:
        {result} += {word}
    return {result}
"""
        optimized = f"""def {fn}({words}):
    return "".join({words})
"""
        constraints = f"{words} is a list of strings."

    elif category == "no_change":
        x, values, result = vars_
        original = f"""def {fn}({values}):
    return [{x} * {x} for {x} in {values}]
"""
        optimized = original
        constraints = (
            f"{values} is a list of integers. "
            "The function is already efficient and concise for this task; do not rewrite it unnecessarily."
        )

    else:
        raise ValueError(category)

    return original, optimized, constraints


def cases_for(category):
    if category == "membership":
        return [
            ([], []),
            ([1, 2, 2, 3], [2, 3]),
            ([1, -1, 5], [5]),
            ([4, 4, 4], [1, 2]),
            ([0, 1, 2, 3], []),
        ]
    if category == "repeated_sum":
        return [
            ([],),
            ([1, 2, 3],),
            ([1, -1],),
            ([0, 0],),
            ([5],),
            ([-2, -3, 5],),
        ]
    if category == "repeated_sort":
        return [
            ([], 3),
            ([3, 1, 2], 2),
            ([5, 5, 1], 10),
            ([3, 2, 1], 0),
            ([7, -1, 4, 4, 2], 3),
        ]
    if category in {"list_concat", "no_change"}:
        return [
            ([],),
            ([1, 2, -3],),
            ([0],),
            (list(range(10)),),
        ]
    if category == "string_concat":
        return [
            ([],),
            (["a", "b"],),
            (["", "x", ""],),
            (["hello", " ", "world"],),
            (["a"] * 10,),
        ]
    raise ValueError(category)


def run_function(code, fn_name, args):
    # exec() only ever runs code produced by make_code() in this file.
    namespace = {}
    exec(code, namespace)
    return namespace[fn_name](*copy.deepcopy(args))


def verify_example(example):
    for args in cases_for(example["category"]):
        original_out = run_function(example["original"], example["fn"], args)
        optimized_out = run_function(example["optimized"], example["fn"], args)
        if original_out != optimized_out:
            raise AssertionError(
                f"{example['id']} failed for args={args}: "
                f"{original_out!r} != {optimized_out!r}"
            )


def make_record(example):
    prompt = f"""Optimize this Python function while preserving its behavior.

Input constraints:
- {example['constraints']}

Return only the optimized Python function.

```python
{example['original'].strip()}
```"""

    answer = f"""```python
{example['optimized'].strip()}
```"""

    return {
        "id": example["id"],
        "category": example["category"],
        "fn": example["fn"],
        "original": example["original"],
        "optimized": example["optimized"],
        "constraints": example["constraints"],
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def generate_examples():
    rng = random.Random(SEED)
    examples = []

    for category in ["membership", "repeated_sum", "repeated_sort", "list_concat", "string_concat"]:
        for j in range(40):
            if category == "membership":
                fn = f"{rng.choice(['filter_allowed','keep_matches','select_members','retain_valid','common_items'])}_{j}"
                vars_ = [
                    rng.choice(NAME_POOLS["item"]),
                    rng.choice(NAME_POOLS["items"]),
                    rng.choice(NAME_POOLS["allowed"]),
                    rng.choice(NAME_POOLS["result"]),
                ]
            elif category == "repeated_sum":
                fn = f"{rng.choice(['normalize_values','scale_values','weights'])}_{j}"
                vars_ = [
                    rng.choice(NAME_POOLS["item"]),
                    rng.choice(NAME_POOLS["values"]),
                    rng.choice(NAME_POOLS["result"]),
                    rng.choice(NAME_POOLS["total"]),
                ]
            elif category == "repeated_sort":
                fn = f"{rng.choice(['smallest_values','take_smallest','sorted_prefix'])}_{j}"
                vars_ = [
                    rng.choice(NAME_POOLS["values"]),
                    rng.choice(NAME_POOLS["k"]),
                    rng.choice(NAME_POOLS["result"]),
                    rng.choice(NAME_POOLS["i"]),
                    rng.choice(NAME_POOLS["ordered"]),
                ]
            elif category == "list_concat":
                fn = f"{rng.choice(['square_values','squares','build_squares'])}_{j}"
                vars_ = [
                    rng.choice(NAME_POOLS["item"]),
                    rng.choice(NAME_POOLS["values"]),
                    rng.choice(NAME_POOLS["result"]),
                ]
            else:
                fn = f"{rng.choice(['concat_words','merge_words','join_parts'])}_{j}"
                vars_ = [
                    rng.choice(NAME_POOLS["word"]),
                    rng.choice(NAME_POOLS["words"]),
                    rng.choice(NAME_POOLS["result"]),
                ]

            original, optimized, constraints = make_code(fn, category, vars_)
            example = {
                "id": f"{category}-{j:03d}",
                "category": category,
                "fn": fn,
                "original": original,
                "optimized": optimized,
                "constraints": constraints,
            }
            verify_example(example)
            examples.append(make_record(example))

    for j in range(30):
        category = "no_change"
        fn = f"{rng.choice(['efficient_squares','square_values_fast','map_squares'])}_{j}"
        vars_ = [
            rng.choice(NAME_POOLS["item"]),
            rng.choice(NAME_POOLS["values"]),
            rng.choice(NAME_POOLS["result"]),
        ]
        original, optimized, constraints = make_code(fn, category, vars_)
        example = {
            "id": f"{category}-{j:03d}",
            "category": category,
            "fn": fn,
            "original": original,
            "optimized": optimized,
            "constraints": constraints,
        }
        verify_example(example)
        examples.append(make_record(example))

    return examples


def normalized_original(example):
    """The original code with the function name removed (``def f(``).

    Two examples with the same normalized original present the model with the
    same prompt apart from the function name.
    """
    return re.sub(r"def \w+\(", "def f(", example["original"], count=1)


def stratified_split(examples):
    """v1.1 split (published run): per-category shuffle, no duplicate check."""
    by_cat = {}
    for ex in examples:
        by_cat.setdefault(ex["category"], []).append(ex)

    train, val, test = [], [], []
    rng = random.Random(SEED)

    for category, rows in by_cat.items():
        rng.shuffle(rows)
        if category == "no_change":
            train.extend(rows[:20])
            val.extend(rows[20:25])
            test.extend(rows[25:30])
        else:
            train.extend(rows[:28])
            val.extend(rows[28:33])
            test.extend(rows[33:40])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _take_groups(groups, target):
    """Remove whole groups from ``groups`` until exactly ``target`` rows are taken."""
    taken = []
    for group in list(groups):
        if len(taken) == target:
            break
        if len(taken) + len(group) <= target:
            taken.extend(group)
            groups.remove(group)
    if len(taken) != target:
        raise RuntimeError(f"could not fill a split of {target} with whole groups")
    return taken


def group_split(examples):
    """v1.2 split: same per-category sizes, but duplicate groups never cross splits."""
    by_cat = {}
    for ex in examples:
        by_cat.setdefault(ex["category"], []).append(ex)

    train, val, test = [], [], []
    rng = random.Random(SEED)

    for category, rows in by_cat.items():
        groups = {}
        for ex in rows:
            groups.setdefault(normalized_original(ex), []).append(ex)
        ordered = list(groups.values())
        rng.shuffle(ordered)

        _, n_val, n_test = SPLIT_SIZES.get(category, DEFAULT_SPLIT_SIZES)
        test.extend(_take_groups(ordered, n_test))
        val.extend(_take_groups(ordered, n_val))
        for group in ordered:
            train.extend(group)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def cross_split_duplicates(train, val, test):
    """Count prompts in a later split whose normalized original is in an earlier one."""
    train_keys = {normalized_original(x) for x in train}
    val_keys = {normalized_original(x) for x in val}
    return {
        "test_in_train": sum(normalized_original(x) in train_keys for x in test),
        "val_in_train": sum(normalized_original(x) in train_keys for x in val),
        "test_in_val": sum(normalized_original(x) in val_keys for x in test),
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(split="v1.2", out_dir=DEFAULT_OUT_DIR):
    examples = generate_examples()
    if split == "v1.1":
        train, val, test = stratified_split(examples)
    elif split == "v1.2":
        train, val, test = group_split(examples)
    else:
        raise ValueError(f"unknown split {split!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "pyopt_train.jsonl", train)
    write_jsonl(out_dir / "pyopt_val.jsonl", val)
    write_jsonl(out_dir / "pyopt_test.jsonl", test)
    return train, val, test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["v1.2", "v1.1"], default="v1.2")
    parser.add_argument("--out", default=None, help="output directory (default: data/)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR
    train, val, test = build(args.split, out_dir)

    print(f"Dataset built ({args.split} split) and reference pairs verified -> {out_dir}")
    print(f"train={len(train)}, val={len(val)}, test={len(test)}")
    print("train categories:", Counter(x["category"] for x in train))
    print("val categories:", Counter(x["category"] for x in val))
    print("test categories:", Counter(x["category"] for x in test))
    print("cross-split duplicates:", cross_split_duplicates(train, val, test))


if __name__ == "__main__":
    main()
