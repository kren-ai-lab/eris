from __future__ import annotations

import numpy as np
import pytest

from eris.features.feature_builders import build_from_registry
from eris.features.feature_registry import FeatureSpec


def _spec(token, order, fn, keys_fn, requires):
    return FeatureSpec(
        token=token,
        order=order,
        requires=set(requires),
        fn=fn,
        keys_fn=keys_fn,
    )


def test_builder_rejects_one_dimensional_feature_output():
    spec = _spec(
        "bad",
        10,
        lambda data, ctx, params: np.zeros((data["n"],), dtype=np.float32),
        lambda data, ctx, params: ["bad"],
        {"n"},
    )

    with pytest.raises(ValueError, match="Feature 'bad'.*2D array"):
        build_from_registry(
            {"n": 2},
            mode="bad",
            registry={"bad": spec},
        )


@pytest.mark.parametrize("count_key", ["n", "E"])
def test_builder_rejects_row_mismatch_against_authoritative_count(count_key):
    spec = _spec(
        "short",
        10,
        lambda data, ctx, params: np.zeros((1, 1), dtype=np.float32),
        lambda data, ctx, params: ["short"],
        {count_key},
    )

    with pytest.raises(ValueError, match="expected 2, got 1"):
        build_from_registry(
            {count_key: 2},
            mode="short",
            registry={"short": spec},
        )


def test_builder_rejects_duplicate_tokens():
    spec = _spec(
        "value",
        10,
        lambda data, ctx, params: np.ones((data["n"], 1), dtype=np.float32),
        lambda data, ctx, params: ["value"],
        {"n"},
    )

    with pytest.raises(ValueError, match="Duplicate feature tokens.*value"):
        build_from_registry(
            {"n": 2},
            mode="value+value",
            registry={"value": spec},
        )


def test_builder_rejects_unsupported_feature_parameters():
    spec = _spec(
        "value",
        10,
        lambda data, ctx, params: np.ones((data["n"], 1), dtype=np.float32),
        lambda data, ctx, params: ["value"],
        {"n"},
    )

    with pytest.raises(ValueError, match="does not support parameters.*bogus"):
        build_from_registry(
            {"n": 2},
            mode=[{"value": {"bogus": 1}}],
            registry={"value": spec},
        )


def test_builder_preserves_order_dtype_and_key_count():
    first = _spec(
        "first",
        10,
        lambda data, ctx, params: np.ones((data["n"], 1), dtype=np.int16),
        lambda data, ctx, params: ["first"],
        {"n"},
    )
    second = _spec(
        "second",
        20,
        lambda data, ctx, params: np.full((data["n"], 2), 2.0, dtype=np.float64),
        lambda data, ctx, params: ["second.0", "second.1"],
        {"n"},
    )

    result, keys = build_from_registry(
        {"n": 2},
        mode="second+first",
        registry={"first": first, "second": second},
        return_keys=True,
    )

    assert result.shape == (2, 3)
    assert result.dtype == np.float32
    assert keys == ["first", "second.0", "second.1"]
    assert len(keys) == result.shape[1]
