from eris.datasets.torch_dataset import DatasetIndexError, ErisFolderDataset

__all__ = [
    "DatasetIndexError",
    "ErisFolderDataset",
    "collate_cloud",
    "collate_cloud_pad",
    "collate_graph",
    "collate_graph_strict",
    "collate_grid",
    "collate_grid_pad",
    "collate_learned",
    "collate_matrix",
]


def __getattr__(name: str):
    if name in {
        "collate_cloud",
        "collate_cloud_pad",
        "collate_graph",
        "collate_graph_strict",
        "collate_grid",
        "collate_grid_pad",
        "collate_learned",
        "collate_matrix",
    }:
        from importlib import import_module

        collate = import_module("eris.datasets.collate")
        return getattr(collate, name)
    raise AttributeError(f"module 'eris.datasets' has no attribute {name!r}")
