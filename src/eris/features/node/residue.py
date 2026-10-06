from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from eris.features.utils import ensure_1d
from eris.features.feature_builders import build_from_registry, validate_feature_mode
from eris.features.feature_registry import RESIDUE_NODE_REGISTRY


def build_node_attr(
    aa_index: np.ndarray,
    *,
    # new names (contract)
    plddt_ca: Optional[np.ndarray] = None,
    bfactor_ca: Optional[np.ndarray] = None,
    has_ca: Optional[np.ndarray] = None,
    residue_ids: Optional[np.ndarray] = None,
    chain_id: Optional[np.ndarray] = None,
    chain_ids: Optional[tuple[str, ...]] = None,
    model_index: Optional[int] = None,
    structural_residue_index: Optional[np.ndarray] = None,
    backbone_n: Optional[np.ndarray] = None,
    backbone_ca: Optional[np.ndarray] = None,
    backbone_c: Optional[np.ndarray] = None,
    backbone_n_mask: Optional[np.ndarray] = None,
    backbone_ca_mask: Optional[np.ndarray] = None,
    backbone_c_mask: Optional[np.ndarray] = None,
    resname: Optional[np.ndarray] = None,
    polymer_heavy_atom_coordinates: Optional[np.ndarray] = None,
    polymer_heavy_atom_residue_index: Optional[np.ndarray] = None,
    polymer_heavy_atom_name: Optional[np.ndarray] = None,
    polymer_heavy_atom_element: Optional[np.ndarray] = None,
    interface_flags: Optional[np.ndarray] = None,
    # contract
    mode: str = "onehot+physchem+bfactor_ca",
    missing: str = "zero",
    return_keys: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:

    aa_index = np.asarray(aa_index, dtype=np.int64).reshape(-1)
    n = int(aa_index.shape[0])
    if n == 0:
        validate_feature_mode(mode, RESIDUE_NODE_REGISTRY)
        x = np.zeros((0, 0), dtype=np.float32)
        return (x, []) if return_keys else x

    if has_ca is None:
        has_ca = np.ones((n,), dtype=bool)
    else:
        has_ca = ensure_1d(has_ca, n, "has_ca").astype(bool)

    data = {
        "aa_index": aa_index,
        "plddt_ca": plddt_ca,
        "bfactor_ca": bfactor_ca,
        "has_ca": has_ca,
        "n": n,
    }
    if residue_ids is not None:
        data["residue_ids"] = ensure_1d(residue_ids, n, "residue_ids")
    if chain_id is not None:
        data["chain_id"] = ensure_1d(chain_id, n, "chain_id")
    if structural_residue_index is not None:
        data["structural_residue_index"] = ensure_1d(
            structural_residue_index,
            n,
            "structural_residue_index",
        ).astype(np.int64)
    if resname is not None:
        data["resname"] = ensure_1d(resname, n, "resname").astype(np.str_)
    for name, value in (
        ("backbone_n", backbone_n),
        ("backbone_ca", backbone_ca),
        ("backbone_c", backbone_c),
    ):
        if value is not None:
            coordinate = np.asarray(value, dtype=np.float32)
            if coordinate.shape != (n, 3):
                raise ValueError(f"{name} must have shape ({n}, 3), got {coordinate.shape}")
            data[name] = coordinate
    for name, value in (
        ("backbone_n_mask", backbone_n_mask),
        ("backbone_ca_mask", backbone_ca_mask),
        ("backbone_c_mask", backbone_c_mask),
    ):
        if value is not None:
            data[name] = ensure_1d(value, n, name).astype(bool)
    if interface_flags is not None:
        data["interface_flags"] = ensure_1d(
            interface_flags,
            n,
            "interface_flags",
        ).astype(bool)
    if polymer_heavy_atom_coordinates is not None:
        coordinates = np.asarray(polymer_heavy_atom_coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                f"polymer_heavy_atom_coordinates must have shape (M, 3), got {coordinates.shape}"
            )
        atom_count = coordinates.shape[0]
        data["polymer_heavy_atom_coordinates"] = coordinates
        for name, value, dtype in (
            ("polymer_heavy_atom_residue_index", polymer_heavy_atom_residue_index, np.int64),
            ("polymer_heavy_atom_name", polymer_heavy_atom_name, np.str_),
            ("polymer_heavy_atom_element", polymer_heavy_atom_element, np.str_),
        ):
            if value is not None:
                data[name] = ensure_1d(value, atom_count, name).astype(dtype)

    feat_ctx = {
        "missing": missing,
    }
    if chain_ids is not None:
        feat_ctx["chain_ids"] = chain_ids
    if model_index is not None:
        feat_ctx["model_index"] = model_index

    return build_from_registry(
        data=data,
        mode=mode,
        registry=RESIDUE_NODE_REGISTRY,
        feat_ctx=feat_ctx,
        return_keys=return_keys,
    )
