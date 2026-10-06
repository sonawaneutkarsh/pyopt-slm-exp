"""Dataset generator: determinism, split sizes, verified references, no leakage."""

import filecmp
from collections import Counter

import pytest

import build_pyopt_dataset as gen

DATA = gen.REPO_DIR / "data"
FILES = ["pyopt_train.jsonl", "pyopt_val.jsonl", "pyopt_test.jsonl"]


@pytest.mark.parametrize("split,committed", [("v1.2", DATA), ("v1.1", DATA / "v1.1")])
def test_rebuild_is_byte_identical_to_committed_files(tmp_path, split, committed):
    gen.build(split, tmp_path)
    for name in FILES:
        assert filecmp.cmp(tmp_path / name, committed / name, shallow=False), name


def test_split_sizes_per_category():
    train, val, test = gen.group_split(gen.generate_examples())
    assert (len(train), len(val), len(test)) == (160, 30, 40)
    for rows, index in ((train, 0), (val, 1), (test, 2)):
        counts = Counter(r["category"] for r in rows)
        for category, count in counts.items():
            assert count == gen.SPLIT_SIZES.get(category, gen.DEFAULT_SPLIT_SIZES)[index]


def test_v12_split_has_no_cross_split_duplicates():
    train, val, test = gen.group_split(gen.generate_examples())
    assert gen.cross_split_duplicates(train, val, test) == {
        "test_in_train": 0,
        "val_in_train": 0,
        "test_in_val": 0,
    }


def test_v11_split_leakage_is_what_the_readme_reports():
    train, val, test = gen.stratified_split(gen.generate_examples())
    assert gen.cross_split_duplicates(train, val, test)["test_in_train"] == 13


def test_every_reference_matches_its_original():
    for example in gen.generate_examples():
        gen.verify_example(example)  # raises on any mismatch


def test_import_has_no_side_effects(tmp_path, monkeypatch):
    import importlib

    monkeypatch.chdir(tmp_path)
    importlib.reload(gen)
    assert not (tmp_path / "data").exists()
