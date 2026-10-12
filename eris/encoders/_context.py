"""Shared narrowing of the prepared encoding context."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from eris.core.base_encoder import EncodingContext
    from eris.core.io import LoadedStructureData


def prepared_arrays(ctx: EncodingContext) -> tuple[LoadedStructureData, np.ndarray, np.ndarray]:
    """Return the raw structure, centroid and normalized positions of a prepared context."""
    strc, centroid, pos = ctx.raw_structure, ctx.centroid, ctx.pos
    # Guaranteed by BaseEncoder._prepare_data, which always runs before _create_data.
    assert strc is not None  # noqa: S101
    assert centroid is not None  # noqa: S101
    assert pos is not None  # noqa: S101
    return strc, centroid, pos
