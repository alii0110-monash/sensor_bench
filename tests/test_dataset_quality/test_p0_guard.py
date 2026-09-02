"""P0 guard: test split must never enter probe evaluation.

Two layers:
1. argparse `choices` rejects 'test' at parse time.
2. `validate_splits` enforces again at runtime (defense in depth).
"""
import argparse

import pytest

from scripts.run_dataset_quality import parse_args, validate_splits


def test_argparse_rejects_test_split():
    """argparse choices must reject 'test' before reaching validate_splits."""
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "datasets/mmfi/v4",
                    "--eval-split", "test", "--out", "/tmp/q.json"])


def test_validate_splits_rejects_test_namespace():
    """If a caller bypasses argparse, validate_splits must still catch it."""
    args = argparse.Namespace(eval_split="test")
    with pytest.raises(AssertionError, match="test split"):
        validate_splits(args)


def test_validate_splits_accepts_val():
    args = argparse.Namespace(eval_split="val")
    validate_splits(args)


def test_validate_splits_accepts_train():
    args = argparse.Namespace(eval_split="train")
    validate_splits(args)