# PyOpt — Small Language Model Fine-Tuning Experiment

A small experiment testing whether LoRA fine-tuning can specialize a 4B language model for Python performance optimization.

## Model

Base model: Qwen3-4B-Instruct-2507  
Fine-tuning: LoRA with Unsloth  
GPU: Kaggle T4  
Training examples: 160  
Validation examples: 30  
Test examples: 40  

## Task

Given an inefficient Python function, generate a behavior-preserving optimized version.

The training dataset included several simple optimization patterns:

- repeated membership checks
- repeated calculations inside loops
- repeated sorting
- inefficient list concatenation
- inefficient string concatenation
- already-efficient code that should remain unchanged

## Results

### In-distribution test set

| Metric | Base Qwen 4B | PyOpt |
|---|---:|---:|
| Valid code | 40/40 | 40/40 |
| Correct behavior | 40/40 | 40/40 |
| Meaningfully faster | 29/40 | 36/40 |
| Median speedup | 3.57x | 5.65x |

The largest improvement appeared in repeated membership optimization:

- Base median speedup: 1.01x
- PyOpt median speedup: 11.58x

### Unseen optimization patterns

A small 10-example generalization test used optimization patterns that were not directly represented in training.

- Base Qwen: 9/10 correct
- PyOpt: 8/10 correct

PyOpt generalized successfully to several loop-invariant computations such as repeated `min()`, `max()`, and dictionary creation, but it also overgeneralized some membership transformations and produced two incorrect optimizations.

## Takeaway

Fine-tuning successfully taught the small model specific optimization behavior, particularly repeated membership optimization.

However, the experiment also showed that better performance on the training distribution does not necessarily translate to better generalization.

This is a proof-of-concept experiment, not a production Python optimizer or comprehensive benchmark.

## Future Work

Potential improvements include:

- more diverse optimization categories
- larger and independently constructed evaluation sets
- stronger semantic-equivalence testing
- property-based testing
- better benchmarking methodology
- training examples targeting incorrect overgeneralization
