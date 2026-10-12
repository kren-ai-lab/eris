"""Dataset utilities."""

from importlib import import_module
from typing import Any

from eris.datasets.torch_dataset import DatasetIndexError, ErisFolderDataset

__all__ = [
    "DatasetIndexError",
    "ErisFolderDataset",
    "collate_cloud",
    "collate_graph",
    "collate_grid",
    "collate_learned",
    "collate_matrix",
]


def __getattr__(name: str) -> Any:
    if name in {
        "collate_cloud",
        "collate_graph",
        "collate_grid",
        "collate_learned",
        "collate_matrix",
    }:
        collate = import_module("eris.datasets.collate")
        return getattr(collate, name)
    raise AttributeError(f"module 'eris.datasets' has no attribute {name!r}")
