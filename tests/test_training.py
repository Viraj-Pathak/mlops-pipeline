"""Unit tests for the training data utilities."""
import pandas as pd
import pytest
from src.training.data_utils import preprocess, FEATURE_COLS


def _make_df(n=10):
    import numpy as np
    rng = np.random.default_rng(0)
    data = {col: rng.integers(0, 5, n) for col in FEATURE_COLS}
    data["age"] = rng.integers(30, 80, n)
    return pd.DataFrame(data)


def test_preprocess_adds_age_bin():
    df = _make_df()
    out = preprocess(df)
    assert "age_bin" in out.columns


def test_preprocess_does_not_drop_rows():
    df = _make_df(20)
    out = preprocess(df)
    assert len(out) == 20


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 13
