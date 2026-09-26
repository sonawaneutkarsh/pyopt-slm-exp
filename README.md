# PyOpt v1.1 starter

This project is a small python function optimizer using SLMs that I was messing with .
It consists of a small, controlled dataset for my Qwen 4B + LoRA experiment.

## What it contains
- 160 training examples
- 30 validation examples
- 40 held-out test examples
- 6 task types:
  - repeated list membership -> set lookup
  - repeated sum -> hoist invariant work
  - repeated sorting -> sort once
  - list concatenation -> append
  - string concatenation -> join
  - already-efficient code -> no unnecessary rewrite

Every reference optimization is automatically checked against several test cases before the dataset is written.

## Build/rebuild the dataset

```bash
python build_pyopt_dataset.py
```

Outputs appear in `data/`.

## Important
This is PyOpt v1: a pipeline-validation dataset, not a serious benchmark yet.
The examples are intentionally controlled so I can prove that:
1. the dataset loads,
2. Qwen 4B can be fine-tuned,
3. the adapter saves,
4. you can compare base vs fine-tuned behavior.

After this works, expand the dataset with harder optimization families and stronger tests.


## v1.1 change
`no_change` examples are now genuinely already-optimized list comprehensions, so the expected behavior is to leave them unchanged.
