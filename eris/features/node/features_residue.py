from __future__ import annotations

from importlib import metadata
from numbers import Real
from typing import Any

import numpy as np

from eris.features.feature_registry import FeatureSpec, register_residue_node_feature
from eris.features.geometry import (
    SIDECHAIN_CHI_DEFINITION_TABLE,
    backbone_torsions,
    sidechain_torsions,
)
from eris.features.interface import normalize_interface_params
from eris.features.node.constants import (
    AA_VOCAB,
    PHYSCHEM_LOOKUPS,
    UNKNOWN_INDEX,
    PHYSchem_KEYS,
)
from eris.features.node.external import (
    external_feature_keys,
    external_feature_values,
    normalize_external_params,
    project_external_fingerprint_params,
)
from eris.features.structure_annotations import (
    BIOTITE_PACKAGE,
    BIOTITE_VERSION,
    SECONDARY_STRUCTURE_KEYS,
    SECONDARY_STRUCTURE_STATE_MAPPING,
    TIEN_MAXIMUM_ASA,
    secondary_structure_onehot,
    solvent_accessibility,
)
from eris.features.utils import normalize_choice, normalize_scalar_params, onehot, safe_float

# --- canonical ordering (NO CAMBIAR en releases)
_AA_ONEHOT_KEYS = [f"aa_onehot.{aa}" for aa in AA_VOCAB]
_AA_INDEX_KEY = ["aa_index"]


def aa_index_to_letters(aa_index: np.ndarray) -> np.ndarray:
    aa_index = np.asarray(aa_index, dtype=np.int64)
    invalid = (aa_index < 0) | (aa_index >= len(AA_VOCAB))
    if np.any(invalid):
        value = int(aa_index.flat[np.flatnonzero(invalid)[0]])
        raise ValueError(f"aa_index values must be within [0, {len(AA_VOCAB) - 1}], got {value}")
    return np.array(AA_VOCAB)[aa_index]


# -------------------------
# aa_onehot
# -------------------------
def aa_onehot_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    aa_index = np.asarray(data["aa_index"], dtype=np.int64).reshape(-1)

    oh = onehot(
        np.clip(aa_index, 0, len(AA_VOCAB) - 1),
        num_classes=len(AA_VOCAB),
    )

    return oh.astype(np.float32)


def aa_onehot_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    return _AA_ONEHOT_KEYS


register_residue_node_feature(
    FeatureSpec(
        token="onehot",
        order=10,
        requires={"aa_index"},
        fn=aa_onehot_fn,
        keys_fn=aa_onehot_keys,
    )
)


# -------------------------
# secondary structure
# -------------------------

_SECONDARY_STRUCTURE_ALGORITHM = "P-SEA"
_SECONDARY_STRUCTURE_SEGMENTS = "selected_chain_and_peptide_continuity"


def normalize_secondary_structure_params(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "provider",
        "provider_version",
        "algorithm",
        "coordinate_basis",
        "state_order",
        "state_mapping",
        "segment_handling",
        "max_peptide_bond_distance",
        "missing_ca_policy",
    }
    unknown_params = sorted(set(params) - allowed)
    if unknown_params:
        raise ValueError(f"Unknown secondary_structure parameters: {unknown_params}")

    distance = params.get("max_peptide_bond_distance", 2.0)
    if isinstance(distance, bool) or not isinstance(distance, Real):
        raise TypeError("secondary_structure parameter 'max_peptide_bond_distance' must be a real number")
    distance = float(distance)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError(
            "secondary_structure parameter 'max_peptide_bond_distance' must be a finite positive number"
        )

    fixed = {
        "provider": BIOTITE_PACKAGE,
        "provider_version": BIOTITE_VERSION,
        "algorithm": _SECONDARY_STRUCTURE_ALGORITHM,
        "coordinate_basis": "CA geometry",
        "state_order": ["helix", "sheet", "coil", "unassigned"],
        "state_mapping": dict(SECONDARY_STRUCTURE_STATE_MAPPING),
        "segment_handling": _SECONDARY_STRUCTURE_SEGMENTS,
        "missing_ca_policy": "unassigned",
    }
    for name, expected in fixed.items():
        if name in params and params[name] != expected:
            raise ValueError(f"secondary_structure parameter '{name}' must be {expected!r}")
    return {**fixed, "max_peptide_bond_distance": distance}


def secondary_structure_fn(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    return secondary_structure_onehot(
        ca_coord=data["backbone_ca"],
        ca_valid=data["backbone_ca_mask"],
        c_coord=data["backbone_c"],
        n_coord=data["backbone_n"],
        c_valid=data["backbone_c_mask"],
        n_valid=data["backbone_n_mask"],
        chain_id=data["chain_id"],
        structural_residue_index=data["structural_residue_index"],
        resname=data["resname"],
        max_peptide_bond_distance=params["max_peptide_bond_distance"],
    )


def secondary_structure_keys(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    return list(SECONDARY_STRUCTURE_KEYS)


register_residue_node_feature(
    FeatureSpec(
        token="secondary_structure",
        order=100,
        requires={
            "backbone_n",
            "backbone_ca",
            "backbone_c",
            "backbone_n_mask",
            "backbone_ca_mask",
            "backbone_c_mask",
            "chain_id",
            "structural_residue_index",
            "resname",
        },
        fn=secondary_structure_fn,
        keys_fn=secondary_structure_keys,
        params_fn=normalize_secondary_structure_params,
    )
)


# -------------------------
# SASA and RSA
# -------------------------

_ACCESSIBILITY_BASE_PARAMS = {
    "provider": BIOTITE_PACKAGE,
    "provider_version": BIOTITE_VERSION,
    "algorithm": "Shrake-Rupley",
    "probe_radius_angstrom": 1.4,
    "sphere_point_count": 1000,
    "sphere_point_distribution": "Fibonacci",
    "vdw_radii": "ProtOr",
    "atom_selection": "selected_chain_polymer_heavy_atoms",
    "hydrogen_handling": "excluded",
    "ion_handling": "excluded",
    "calculation_context": "selected_complex",
    "unsupported_radius_policy": "error",
    "incomplete_canonical_residue_policy": "nan",
    "units": "square_angstroms",
}
_RSA_PARAMS = {
    **_ACCESSIBILITY_BASE_PARAMS,
    "normalization": "Tien et al. theoretical maximum ASA",
    "maximum_asa_table": dict(TIEN_MAXIMUM_ASA),
    "noncanonical_normalization_policy": "nan",
    "clipping": "none",
}


def _normalize_fixed_accessibility_params(
    token: str,
    params: dict[str, Any],
    fixed: dict[str, Any],
) -> dict[str, Any]:
    unknown_params = sorted(set(params) - set(fixed))
    if unknown_params:
        raise ValueError(f"Unknown {token} parameters: {unknown_params}")
    for name, expected in fixed.items():
        if name in params and params[name] != expected:
            raise ValueError(f"{token} parameter '{name}' must be {expected!r}")
    return dict(fixed)


def normalize_sasa_params(params: dict[str, Any]) -> dict[str, Any]:
    return _normalize_fixed_accessibility_params("sasa", params, _ACCESSIBILITY_BASE_PARAMS)


def normalize_rsa_params(params: dict[str, Any]) -> dict[str, Any]:
    return _normalize_fixed_accessibility_params("rsa", params, _RSA_PARAMS)


def _accessibility_values(data: dict[str, Any], ctx: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cache_key = "_eris_solvent_accessibility"
    cached = ctx.get(cache_key)
    if cached is None:
        cached = solvent_accessibility(
            coordinates=data["polymer_heavy_atom_coordinates"],
            residue_index=data["polymer_heavy_atom_residue_index"],
            atom_name=data["polymer_heavy_atom_name"],
            element=data["polymer_heavy_atom_element"],
            resname=data["resname"],
            chain_id=data["chain_id"],
        )
        ctx[cache_key] = cached
    return cached


def sasa_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    sasa, _ = _accessibility_values(data, ctx)
    return sasa.reshape(-1, 1)


def rsa_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    _, rsa = _accessibility_values(data, ctx)
    return rsa.reshape(-1, 1)


def sasa_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    return ["sasa"]


def rsa_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    return ["rsa"]


_ACCESSIBILITY_REQUIREMENTS = {
    "polymer_heavy_atom_coordinates",
    "polymer_heavy_atom_residue_index",
    "polymer_heavy_atom_name",
    "polymer_heavy_atom_element",
    "resname",
    "chain_id",
}

register_residue_node_feature(
    FeatureSpec(
        token="sasa",
        order=110,
        requires=_ACCESSIBILITY_REQUIREMENTS,
        fn=sasa_fn,
        keys_fn=sasa_keys,
        params_fn=normalize_sasa_params,
    )
)

register_residue_node_feature(
    FeatureSpec(
        token="rsa",
        order=120,
        requires=_ACCESSIBILITY_REQUIREMENTS,
        fn=rsa_fn,
        keys_fn=rsa_keys,
        params_fn=normalize_rsa_params,
    )
)


# -------------------------
# aa_index
# -------------------------
def aa_index_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    aa_index = np.asarray(data["aa_index"], dtype=np.int64).reshape(-1, 1)

    return aa_index.astype(np.float32)


def aa_index_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    return _AA_INDEX_KEY


register_residue_node_feature(
    FeatureSpec(
        token="aa_index",
        order=20,
        requires={"aa_index"},
        fn=aa_index_fn,
        keys_fn=aa_index_keys,
    )
)


# -------------------------
# physchem
# -------------------------
def physchem_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    aa_index = np.asarray(data["aa_index"], dtype=np.int64)

    feats = [lookup[aa_index] for _, lookup in PHYSCHEM_LOOKUPS]

    return np.stack(feats, axis=1).astype(np.float32)


def physchem_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    return PHYSchem_KEYS


register_residue_node_feature(
    FeatureSpec(
        token="physchem",
        order=30,
        requires={"aa_index"},
        fn=physchem_fn,
        keys_fn=physchem_keys,
    )
)


# -------------------------
# aaindex
# -------------------------

_AAINDEX_PROVIDER = "aaindex"
_AAINDEX_SOURCE = "AAIndex1"


def _choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    return normalize_choice(f"aaindex parameter '{name}'", value, choices)


def normalize_aaindex_params(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "ids",
        "normalization",
        "unknown",
        "missing",
        "provider",
        "source",
        "provider_version",
    }
    unknown_params = sorted(set(params) - allowed)
    if unknown_params:
        raise ValueError(f"Unknown aaindex parameters: {unknown_params}")

    if "ids" not in params:
        raise ValueError("aaindex parameter 'ids' is required")
    ids_value = params["ids"]
    if not isinstance(ids_value, list):
        raise TypeError("aaindex parameter 'ids' must be a list of accession strings")
    if not ids_value:
        raise ValueError("aaindex parameter 'ids' must not be empty")

    accessions: list[str] = []
    for accession in ids_value:
        if not isinstance(accession, str):
            raise TypeError("aaindex parameter 'ids' must contain only strings")
        canonical = accession.strip().upper()
        if not canonical:
            raise ValueError("aaindex accession identifiers must not be empty")
        accessions.append(canonical)
    duplicates = sorted({item for item in accessions if accessions.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate AAIndex1 accession identifiers: {duplicates}")

    try:
        provider_version = metadata.version(_AAINDEX_PROVIDER)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("AAIndex features require the 'aaindex>=1.3.2,<2' package") from exc

    supplied_identity = {
        "provider": _AAINDEX_PROVIDER,
        "source": _AAINDEX_SOURCE,
        "provider_version": provider_version,
    }
    for name, expected in supplied_identity.items():
        if name in params and params[name] != expected:
            raise ValueError(f"aaindex parameter '{name}' must match the active provider value {expected!r}")

    from aaindex import aaindex1

    invalid = [accession for accession in accessions if accession not in aaindex1]
    if invalid:
        raise ValueError(f"Unknown AAIndex1 accession identifiers: {invalid}")

    return {
        "ids": accessions,
        "normalization": _choice("normalization", params.get("normalization", "none"), ("none",)),
        "unknown": _choice("unknown", params.get("unknown", "nan"), ("nan", "zero", "error")),
        "missing": _choice("missing", params.get("missing", "nan"), ("nan", "zero", "error")),
        **supplied_identity,
    }


def _policy_value(policy: str, *, message: str) -> float:
    if policy == "nan":
        return float("nan")
    if policy == "zero":
        return 0.0
    raise ValueError(message)


def aaindex_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    from aaindex import aaindex1

    letters = aa_index_to_letters(data["aa_index"])
    columns: list[np.ndarray] = []
    for accession in params["ids"]:
        record_values = aaindex1.values(accession)
        column: list[float] = []
        for row, letter in enumerate(letters.tolist()):
            if letter == "X":
                column.append(
                    _policy_value(
                        params["unknown"],
                        message=(
                            f"AAIndex1 feature '{accession}' cannot encode unknown residue X "
                            f"at row {row} with unknown='error'"
                        ),
                    )
                )
                continue

            value = record_values.get(letter)
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
            ):
                column.append(
                    _policy_value(
                        params["missing"],
                        message=(
                            f"AAIndex1 record '{accession}' has no finite value for residue "
                            f"{letter} at row {row} with missing='error'"
                        ),
                    )
                )
            else:
                column.append(float(value))
        columns.append(np.asarray(column, dtype=np.float32))

    return np.stack(columns, axis=1).astype(np.float32)


def aaindex_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    return [f"aaindex.{accession}" for accession in params["ids"]]


register_residue_node_feature(
    FeatureSpec(
        token="aaindex",
        order=40,
        requires={"aa_index"},
        fn=aaindex_fn,
        keys_fn=aaindex_keys,
        params_fn=normalize_aaindex_params,
    )
)

# -------------------------
# plddt_ca
# -------------------------


def normalize_plddt_params(params: dict[str, Any]) -> dict[str, Any]:
    return normalize_scalar_params("plddt_ca", params)


def plddt_ca_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    n = data["n"]
    missing = ctx["missing"]

    if data["plddt_ca"] is None:
        v = np.full((n, 1), np.nan if missing == "nan" else 0.0, dtype=np.float32)
    else:
        v = safe_float(data["plddt_ca"], missing=missing).reshape(-1, 1)
        finite = np.isfinite(v)
        invalid = finite & ((v < 0.0) | (v > 100.0))
        if np.any(invalid):
            value = float(v[np.flatnonzero(invalid)[0], 0])
            raise ValueError(f"plddt_ca values must be within [0, 100], got {value}")
        if params["normalization"] == "unit_interval":
            v = v / 100.0
    return v.astype(np.float32)


def plddt_ca_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    return ["plddt_ca"]


register_residue_node_feature(
    FeatureSpec(
        token="plddt_ca",
        order=50,
        requires={"n", "plddt_ca", "missing"},
        fn=plddt_ca_fn,
        keys_fn=plddt_ca_keys,
        params_fn=normalize_plddt_params,
    )
)


# -------------------------
# bfactor_ca
# -------------------------


def bfactor_ca_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    n = data["n"]
    missing = ctx["missing"]

    if data["bfactor_ca"] is None:
        v = np.full((n, 1), np.nan if missing == "nan" else 0.0, dtype=np.float32)
    else:
        v = safe_float(data["bfactor_ca"], missing=missing).reshape(-1, 1)
    return v.astype(np.float32)


def bfactor_ca_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    return ["bfactor_ca"]


register_residue_node_feature(
    FeatureSpec(
        token="bfactor_ca",
        order=60,
        requires={"n", "bfactor_ca", "missing"},
        fn=bfactor_ca_fn,
        keys_fn=bfactor_ca_keys,
    )
)


# -------------------------
# flags
# -------------------------


def flags_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    aa_index = np.asarray(data["aa_index"], dtype=np.int64).reshape(-1)
    has_ca = data["has_ca"]

    is_gly = (aa_index == int(list(AA_VOCAB).index("G"))).astype(np.float32).reshape(-1, 1)
    is_pro = (aa_index == int(list(AA_VOCAB).index("P"))).astype(np.float32).reshape(-1, 1)
    is_unk = (aa_index == UNKNOWN_INDEX).astype(np.float32).reshape(-1, 1)
    has_ca_f = has_ca.astype(np.float32).reshape(-1, 1)

    return np.concatenate([is_gly, is_pro, is_unk, has_ca_f], axis=1).astype(np.float32)


def flags_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    return ["flag.is_gly", "flag.is_pro", "flag.is_unk", "flag.has_ca"]


register_residue_node_feature(
    FeatureSpec(
        token="flags",
        order=70,
        requires={"aa_index", "has_ca"},
        fn=flags_fn,
        keys_fn=flags_keys,
    )
)


# -------------------------
# backbone_geometry
# -------------------------

_BACKBONE_TORSION_CONVENTION = "standard_signed_atan2[-pi,pi)"
_BACKBONE_TORSION_UNITS = "radians"
_BACKBONE_MISSING_VALUE_POLICY = "nan"


def normalize_backbone_geometry_params(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "mode",
        "max_peptide_bond_distance",
        "torsion_convention",
        "units",
        "missing_value_policy",
    }
    unknown_params = sorted(set(params) - allowed)
    if unknown_params:
        raise ValueError(f"Unknown backbone_geometry parameters: {unknown_params}")

    mode = params.get("mode", "sincos")
    if not isinstance(mode, str):
        raise TypeError("backbone_geometry parameter 'mode' must be a string")
    mode = mode.strip().lower()
    if mode not in {"sincos", "raw", "both"}:
        raise ValueError("backbone_geometry parameter 'mode' must be one of: sincos, raw, both")

    distance = params.get("max_peptide_bond_distance", 2.0)
    if isinstance(distance, bool) or not isinstance(distance, Real):
        raise TypeError("backbone_geometry parameter 'max_peptide_bond_distance' must be a real number")
    distance = float(distance)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError(
            "backbone_geometry parameter 'max_peptide_bond_distance' must be a finite positive number"
        )

    fixed_parameters = {
        "torsion_convention": _BACKBONE_TORSION_CONVENTION,
        "units": _BACKBONE_TORSION_UNITS,
        "missing_value_policy": _BACKBONE_MISSING_VALUE_POLICY,
    }
    for name, expected in fixed_parameters.items():
        if name in params and params[name] != expected:
            raise ValueError(f"backbone_geometry parameter '{name}' must be {expected!r}")

    return {
        "mode": mode,
        **fixed_parameters,
        "max_peptide_bond_distance": distance,
    }


def backbone_geometry_fn(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    angles = backbone_torsions(
        data["backbone_n"],
        data["backbone_ca"],
        data["backbone_c"],
        data["backbone_n_mask"],
        data["backbone_ca_mask"],
        data["backbone_c_mask"],
        data["chain_id"],
        data["structural_residue_index"],
        max_peptide_bond_distance=params["max_peptide_bond_distance"],
    )
    mode = params["mode"]
    if mode == "raw":
        return angles

    periodic = np.column_stack(
        [
            np.sin(angles[:, 0]),
            np.cos(angles[:, 0]),
            np.sin(angles[:, 1]),
            np.cos(angles[:, 1]),
            np.sin(angles[:, 2]),
            np.cos(angles[:, 2]),
        ]
    ).astype(np.float32)
    if mode == "both":
        return np.concatenate([periodic, angles], axis=1).astype(np.float32)
    return periodic


def backbone_geometry_keys(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    periodic = [
        "backbone.phi_sin",
        "backbone.phi_cos",
        "backbone.psi_sin",
        "backbone.psi_cos",
        "backbone.omega_sin",
        "backbone.omega_cos",
    ]
    raw = ["backbone.phi", "backbone.psi", "backbone.omega"]
    if params["mode"] == "sincos":
        return periodic
    if params["mode"] == "raw":
        return raw
    return [*periodic, *raw]


register_residue_node_feature(
    FeatureSpec(
        token="backbone_geometry",
        order=75,
        requires={
            "backbone_n",
            "backbone_ca",
            "backbone_c",
            "backbone_n_mask",
            "backbone_ca_mask",
            "backbone_c_mask",
            "chain_id",
            "structural_residue_index",
        },
        fn=backbone_geometry_fn,
        keys_fn=backbone_geometry_keys,
        params_fn=normalize_backbone_geometry_params,
    )
)


# -------------------------
# sidechain_geometry
# -------------------------

_SIDECHAIN_ATOM_DEFINITION_CONVENTION = "standard_residue_specific_chi1_through_chi4"


def normalize_sidechain_geometry_params(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "mode",
        "torsion_convention",
        "units",
        "atom_definition_convention",
        "atom_definition_table",
        "missing_value_policy",
    }
    unknown_params = sorted(set(params) - allowed)
    if unknown_params:
        raise ValueError(f"Unknown sidechain_geometry parameters: {unknown_params}")

    mode = params.get("mode", "sincos")
    if not isinstance(mode, str):
        raise TypeError("sidechain_geometry parameter 'mode' must be a string")
    mode = mode.strip().lower()
    if mode not in {"sincos", "raw", "both"}:
        raise ValueError("sidechain_geometry parameter 'mode' must be one of: sincos, raw, both")

    fixed_parameters = {
        "torsion_convention": _BACKBONE_TORSION_CONVENTION,
        "units": _BACKBONE_TORSION_UNITS,
        "atom_definition_convention": _SIDECHAIN_ATOM_DEFINITION_CONVENTION,
        "atom_definition_table": SIDECHAIN_CHI_DEFINITION_TABLE,
        "missing_value_policy": _BACKBONE_MISSING_VALUE_POLICY,
    }
    for name, expected in fixed_parameters.items():
        if name in params and params[name] != expected:
            raise ValueError(f"sidechain_geometry parameter '{name}' must be {expected!r}")
    return {"mode": mode, **fixed_parameters}


def sidechain_geometry_fn(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    angles = sidechain_torsions(
        data["polymer_heavy_atom_coordinates"],
        data["polymer_heavy_atom_residue_index"],
        data["polymer_heavy_atom_name"],
        data["resname"],
    )
    if params["mode"] == "raw":
        return angles

    periodic = np.column_stack(
        [component(angles[:, chi]) for chi in range(4) for component in (np.sin, np.cos)]
    ).astype(np.float32)
    if params["mode"] == "both":
        return np.concatenate([periodic, angles], axis=1).astype(np.float32)
    return periodic


def sidechain_geometry_keys(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    periodic = [f"sidechain.chi{chi}_{component}" for chi in range(1, 5) for component in ("sin", "cos")]
    raw = [f"sidechain.chi{chi}" for chi in range(1, 5)]
    if params["mode"] == "sincos":
        return periodic
    if params["mode"] == "raw":
        return raw
    return [*periodic, *raw]


register_residue_node_feature(
    FeatureSpec(
        token="sidechain_geometry",
        order=76,
        requires={
            "polymer_heavy_atom_coordinates",
            "polymer_heavy_atom_residue_index",
            "polymer_heavy_atom_name",
            "resname",
        },
        fn=sidechain_geometry_fn,
        keys_fn=sidechain_geometry_keys,
        params_fn=normalize_sidechain_geometry_params,
    )
)


# -------------------------
# external
# -------------------------


register_residue_node_feature(
    FeatureSpec(
        token="external",
        order=80,
        requires={"residue_ids", "chain_ids", "model_index"},
        fn=external_feature_values,
        keys_fn=external_feature_keys,
        params_fn=normalize_external_params,
        fingerprint_params_fn=project_external_fingerprint_params,
    )
)


# -------------------------
# inter-chain interface
# -------------------------
def interface_fn(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    return np.asarray(data["interface_flags"], dtype=np.float32).reshape(-1, 1)


def interface_keys(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    return ["interface.is_interface"]


register_residue_node_feature(
    FeatureSpec(
        token="interface",
        order=90,
        requires={"interface_flags"},
        fn=interface_fn,
        keys_fn=interface_keys,
        params_fn=normalize_interface_params,
    )
)
