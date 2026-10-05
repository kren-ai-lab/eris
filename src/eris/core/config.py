from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from numbers import Real
from types import MappingProxyType
from typing import Any, ClassVar, Literal

from eris.features.feature_registry import (
    ATOM_NODE_REGISTRY,
    GRAPH_EDGE_REGISTRY,
    GRID_VOXEL_REGISTRY,
    RESIDUE_NODE_REGISTRY,
)
from eris.features.utils import parse_mode


def _metadata(
    description: str,
    *,
    unit: str | None = None,
    choices: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"description": description}
    if unit is not None:
        metadata["unit"] = unit
    if choices is not None:
        metadata["choices"] = choices
    return metadata


def _require_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _require_int(
    name: str,
    value: Any,
    *,
    minimum: int,
    inclusive: bool = True,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    valid = value >= minimum if inclusive else value > minimum
    if not valid:
        operator = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be {operator} {minimum}")
    return value


def _require_float(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    inclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        valid = result >= minimum if inclusive else result > minimum
        if not valid:
            operator = ">=" if inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}")
    return result


def _require_optional_positive_float(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _require_float(name, value, minimum=0.0)


def _normalize_choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower()
    if normalized not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {allowed}")
    return normalized


def _normalize_altloc(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("altloc must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("altloc must not be empty")
    lowered = normalized.lower()
    if lowered in {"first", "max_occupancy"}:
        return lowered
    return normalized.upper()


@dataclass(frozen=True, kw_only=True)
class _FrozenMapping:
    items: tuple[tuple[str, Any], ...] = ()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, _FrozenMapping):
        return value
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("feature parameters must contain only finite numbers")
        return normalized
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("feature parameter mapping keys must be strings")
        frozen_items: list[tuple[str, Any]] = []
        for key in sorted(keys):
            frozen_items.append((key, _freeze_json(value[key])))
        return _FrozenMapping(items=tuple(frozen_items))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"feature parameters must be JSON-compatible, got {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, _FrozenMapping):
        return {key: _thaw_json(item) for key, item in value.items}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, kw_only=True)
class FeatureToken:
    """One requested feature token with recursively immutable parameters."""

    name: str
    params: _FrozenMapping = field(default_factory=_FrozenMapping)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("feature token name must be a string")
        name = self.name.strip()
        if not name:
            raise ValueError("feature token name must not be empty")
        params = _freeze_json(self.params)
        if not isinstance(params, _FrozenMapping):
            raise TypeError("feature token params must be a mapping")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "params", params)

    def to_value(self) -> str | dict[str, Any]:
        if not self.params.items:
            return self.name
        return {self.name: _thaw_json(self.params)}


_FeatureKind = Literal["atom_node", "residue_node", "edge", "grid"]

_FEATURE_REGISTRIES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "atom_node": ATOM_NODE_REGISTRY,
        "residue_node": RESIDUE_NODE_REGISTRY,
        "edge": GRAPH_EDGE_REGISTRY,
        "grid": GRID_VOXEL_REGISTRY,
    }
)


@dataclass(frozen=True, kw_only=True)
class FeatureMode:
    """Canonical requested feature order, independent of physical output order."""

    tokens: tuple[FeatureToken, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, (list, tuple)):
            raise TypeError("feature mode tokens must be a list or tuple")
        if any(not isinstance(token, FeatureToken) for token in self.tokens):
            raise TypeError("feature mode tokens must be FeatureToken instances")
        object.__setattr__(self, "tokens", tuple(self.tokens))

    @classmethod
    def from_value(cls, value: Any, *, kind: _FeatureKind) -> FeatureMode:
        if kind not in _FEATURE_REGISTRIES:
            raise ValueError(f"unknown feature mode kind: {kind}")

        if isinstance(value, cls):
            source: Any = value.to_list()
        elif isinstance(value, tuple):
            source = list(value)
        else:
            source = value

        parsed = parse_mode(source)
        registry = _FEATURE_REGISTRIES[kind]
        names = [name for name, _ in parsed]

        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate {kind} feature tokens: {duplicates}")

        unknown = [name for name in names if name not in registry]
        if unknown:
            raise ValueError(f"unknown {kind} feature tokens: {unknown}")

        tokens: list[FeatureToken] = []
        for name, params in parsed:
            spec = registry[name]
            if kind == "grid" and getattr(spec, "params_fn", None) is None:
                normalized_params = spec.normalize_params(params, allow_unvalidated=True)
            else:
                normalized_params = spec.normalize_params(params)
            frozen_params = _freeze_json(normalized_params)
            if not isinstance(frozen_params, _FrozenMapping):
                raise TypeError(f"parameters for feature '{name}' must be a mapping")
            tokens.append(FeatureToken(name=name, params=frozen_params))

        return cls(tokens=tuple(tokens))

    def to_list(self) -> list[Any]:
        return [token.to_value() for token in self.tokens]

    def to_fingerprint_list(self, *, kind: _FeatureKind) -> list[Any]:
        registry = _FEATURE_REGISTRIES[kind]
        projected: list[Any] = []
        for token in self.tokens:
            params = registry[token.name].project_fingerprint_params(_thaw_json(token.params))
            projected.append(token.name if not params else {token.name: params})
        return projected


def _atom_node_mode_default() -> FeatureMode:
    return FeatureMode.from_value("elem+role+bfactor+occ", kind="atom_node")


def _residue_node_mode_default() -> FeatureMode:
    return FeatureMode.from_value(
        "onehot+physchem+bfactor_ca",
        kind="residue_node",
    )


def _edge_mode_default() -> FeatureMode:
    return FeatureMode.from_value("dist+rbf+seq+local", kind="edge")


def _grid_mode_default() -> FeatureMode:
    return FeatureMode.from_value(
        "occupancy+density+aa_onehot+bfactor",
        kind="grid",
    )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, FeatureMode):
        return value.to_list()
    if isinstance(value, _FrozenMapping):
        return _thaw_json(value)
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    return value


@dataclass(frozen=True, kw_only=True)
class CanonicalRepresentationConfig:
    """Internal canonical parameter configuration for one representation."""

    representation_name: ClassVar[str]
    granularity: ClassVar[Literal["atom", "residue"]]

    @property
    def coordinate_source(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            config_field.name: _json_compatible(getattr(self, config_field.name))
            for config_field in fields(self)
        }

    def to_fingerprint_dict(self) -> dict[str, Any]:
        """Return canonical scientific parameters used for config/cache identity."""
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


@dataclass(frozen=True, kw_only=True)
class _CommonGeometryConfig(CanonicalRepresentationConfig):
    center: bool = field(
        default=True,
        metadata=_metadata("Center coordinates before representation construction."),
    )
    scale: float | None = field(
        default=None,
        metadata=_metadata("Optional coordinate divisor.", unit="angstrom"),
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _require_bool("center", self.center))
        object.__setattr__(
            self,
            "scale",
            _require_optional_positive_float("scale", self.scale),
        )


@dataclass(frozen=True, kw_only=True)
class _AtomSelectionConfig(_CommonGeometryConfig):
    include_hetero: bool = field(default=False, metadata=_metadata("Include hetero residues."))
    include_h: bool = field(default=False, metadata=_metadata("Include hydrogen atoms."))
    drop_water: bool = field(default=True, metadata=_metadata("Remove water residues."))
    keep_ions: bool = field(
        default=True,
        metadata=_metadata("Keep ion residues when otherwise selected."),
    )
    altloc: str = field(
        default="first",
        metadata=_metadata("Alternate-location selection policy."),
    )
    amino_acids_only: bool = field(
        default=False,
        metadata=_metadata("Restrict extraction to amino-acid residues."),
    )

    @property
    def coordinate_source(self) -> str:
        return "atom"

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "include_hetero",
            "include_h",
            "drop_water",
            "keep_ions",
            "amino_acids_only",
        ):
            object.__setattr__(self, name, _require_bool(name, getattr(self, name)))
        object.__setattr__(self, "altloc", _normalize_altloc(self.altloc))


@dataclass(frozen=True, kw_only=True)
class _ResidueSelectionConfig(_CommonGeometryConfig):
    residue_center: Literal["ca", "mean", "com"] = field(
        default="ca",
        metadata=_metadata(
            "Coordinate source used for each residue.",
            choices=("ca", "mean", "com"),
        ),
    )
    amino_acids_only: Literal[True] = field(
        default=True,
        metadata=_metadata("Frozen residue extraction policy for v0.1."),
    )
    altloc: Literal["max_occupancy"] = field(
        default="max_occupancy",
        metadata=_metadata("Frozen residue alternate-location policy for v0.1."),
    )
    missing_residue_center: Literal["skip", "error"] = field(
        default="skip",
        metadata=_metadata(
            "Missing residue-center policy for explicit residue representations.",
            choices=("skip", "error"),
        ),
    )
    plddt_source: Literal["none", "bfactor"] = field(
        default="none",
        metadata=_metadata(
            "Explicit source used to interpret residue pLDDT values.",
            choices=("none", "bfactor"),
        ),
    )

    @property
    def coordinate_source(self) -> str:
        return {"ca": "ca", "mean": "residue_mean", "com": "residue_com"}[self.residue_center]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "residue_center",
            _normalize_choice("residue_center", self.residue_center, ("ca", "mean", "com")),
        )
        amino_acids_only = _require_bool("amino_acids_only", self.amino_acids_only)
        if not amino_acids_only:
            raise ValueError("amino_acids_only must remain True in M2.2")
        object.__setattr__(self, "amino_acids_only", amino_acids_only)
        object.__setattr__(
            self,
            "altloc",
            _normalize_choice("altloc", self.altloc, ("max_occupancy",)),
        )
        object.__setattr__(
            self,
            "missing_residue_center",
            _normalize_choice(
                "missing_residue_center",
                self.missing_residue_center,
                ("skip", "error"),
            ),
        )
        object.__setattr__(
            self,
            "plddt_source",
            _normalize_choice("plddt_source", self.plddt_source, ("none", "bfactor")),
        )


@dataclass(frozen=True, kw_only=True)
class _AtomNodeFeatureConfig(_AtomSelectionConfig):
    node_attr_mode: FeatureMode = field(
        default_factory=_atom_node_mode_default,
        metadata=_metadata("Requested atom node-feature mode."),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "node_attr_mode",
            FeatureMode.from_value(self.node_attr_mode, kind="atom_node"),
        )


@dataclass(frozen=True, kw_only=True)
class _ResidueNodeFeatureConfig(_ResidueSelectionConfig):
    node_attr_mode: FeatureMode = field(
        default_factory=_residue_node_mode_default,
        metadata=_metadata("Requested residue node-feature mode."),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "node_attr_mode",
            FeatureMode.from_value(self.node_attr_mode, kind="residue_node"),
        )

    def to_fingerprint_dict(self) -> dict[str, Any]:
        parameters = super().to_fingerprint_dict()
        parameters["node_attr_mode"] = self.node_attr_mode.to_fingerprint_list(kind="residue_node")
        return parameters


def _validate_edge_feature_fields(config: Any) -> None:
    object.__setattr__(
        config,
        "edge_attr_mode",
        FeatureMode.from_value(config.edge_attr_mode, kind="edge"),
    )
    object.__setattr__(
        config,
        "rbf_k",
        _require_int("rbf_k", config.rbf_k, minimum=0, inclusive=False),
    )
    object.__setattr__(
        config,
        "rbf_dmax",
        _require_float("rbf_dmax", config.rbf_dmax, minimum=0.0),
    )
    object.__setattr__(
        config,
        "local_cutoff",
        _require_int("local_cutoff", config.local_cutoff, minimum=0),
    )


@dataclass(frozen=True, kw_only=True)
class _AtomGraphFeatureConfig(_AtomNodeFeatureConfig):
    edge_attr_mode: FeatureMode = field(
        default_factory=_edge_mode_default,
        metadata=_metadata("Requested edge-feature mode."),
    )
    rbf_k: int = field(default=16, metadata=_metadata("Number of edge RBF kernels."))
    rbf_dmax: float = field(
        default=20.0,
        metadata=_metadata("Maximum edge RBF distance.", unit="angstrom"),
    )
    local_cutoff: int = field(
        default=4,
        metadata=_metadata("Local edge threshold in sequence positions."),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_edge_feature_fields(self)


@dataclass(frozen=True, kw_only=True)
class _ResidueGraphFeatureConfig(_ResidueNodeFeatureConfig):
    edge_attr_mode: FeatureMode = field(
        default_factory=_edge_mode_default,
        metadata=_metadata("Requested edge-feature mode."),
    )
    rbf_k: int = field(default=16, metadata=_metadata("Number of edge RBF kernels."))
    rbf_dmax: float = field(
        default=20.0,
        metadata=_metadata("Maximum edge RBF distance.", unit="angstrom"),
    )
    local_cutoff: int = field(
        default=4,
        metadata=_metadata("Local edge threshold in sequence positions."),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_edge_feature_fields(self)


@dataclass(frozen=True, kw_only=True)
class _ResidueBinningConfig(_ResidueSelectionConfig):
    edges: tuple[float, ...] | None = field(
        default=None,
        metadata=_metadata("Optional explicit bin boundaries.", unit="angstrom"),
    )
    dmin: float | None = field(
        default=2.0,
        metadata=_metadata("Minimum generated bin boundary.", unit="angstrom"),
    )
    dmax: float | None = field(
        default=20.0,
        metadata=_metadata("Maximum generated bin boundary.", unit="angstrom"),
    )
    num_bins: int | None = field(
        default=36,
        metadata=_metadata("Number of uniformly generated bins."),
    )
    add_no_contact_class: bool = field(
        default=True,
        metadata=_metadata("Reserve class zero for no contact."),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "add_no_contact_class",
            _require_bool("add_no_contact_class", self.add_no_contact_class),
        )

        if self.edges is not None:
            if not isinstance(self.edges, (list, tuple)):
                raise TypeError("edges must be a list, tuple, or None")
            normalized = tuple(
                _require_float(f"edges[{index}]", value) for index, value in enumerate(self.edges)
            )
            if len(normalized) < 2:
                raise ValueError("edges must contain at least two values")
            if any(right <= left for left, right in zip(normalized, normalized[1:], strict=False)):
                raise ValueError("edges must be strictly increasing")
            object.__setattr__(self, "edges", normalized)
            object.__setattr__(self, "dmin", None)
            object.__setattr__(self, "dmax", None)
            object.__setattr__(self, "num_bins", None)
            return

        if self.dmin is None or self.dmax is None or self.num_bins is None:
            raise ValueError("dmin, dmax, and num_bins are required when edges is None")

        dmin = _require_float("dmin", self.dmin)
        dmax = _require_float("dmax", self.dmax)
        num_bins = _require_int("num_bins", self.num_bins, minimum=0, inclusive=False)
        if dmax <= dmin:
            raise ValueError("dmax must be greater than dmin")
        object.__setattr__(self, "edges", None)
        object.__setattr__(self, "dmin", dmin)
        object.__setattr__(self, "dmax", dmax)
        object.__setattr__(self, "num_bins", num_bins)


@dataclass(frozen=True, kw_only=True)
class AtomPointCloudConfig(_AtomNodeFeatureConfig):
    representation_name: ClassVar[str] = "atom_point_cloud"
    granularity: ClassVar[Literal["atom"]] = "atom"


@dataclass(frozen=True, kw_only=True)
class ResiduePointCloudConfig(_ResidueNodeFeatureConfig):
    representation_name: ClassVar[str] = "residue_point_cloud"
    granularity: ClassVar[Literal["residue"]] = "residue"


@dataclass(frozen=True, kw_only=True)
class AtomKNNGraphConfig(_AtomGraphFeatureConfig):
    representation_name: ClassVar[str] = "atom_knn_graph"
    granularity: ClassVar[Literal["atom"]] = "atom"

    k: int = field(default=16, metadata=_metadata("Number of nearest neighbors."))
    cutoff: float | None = field(
        default=12.0,
        metadata=_metadata("Optional neighbor-search radius.", unit="angstrom"),
    )
    symmetric: bool = field(
        default=True,
        metadata=_metadata("Emit unique reverse directed edges."),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "k",
            _require_int("k", self.k, minimum=0, inclusive=False),
        )
        object.__setattr__(
            self,
            "cutoff",
            _require_optional_positive_float("cutoff", self.cutoff),
        )
        object.__setattr__(self, "symmetric", _require_bool("symmetric", self.symmetric))


@dataclass(frozen=True, kw_only=True)
class ResidueKNNGraphConfig(_ResidueGraphFeatureConfig):
    representation_name: ClassVar[str] = "residue_knn_graph"
    granularity: ClassVar[Literal["residue"]] = "residue"

    knn_mode: Literal["local", "global"] = field(
        default="local",
        metadata=_metadata("Neighbor candidate-search mode.", choices=("local", "global")),
    )
    k: int = field(default=16, metadata=_metadata("Number of nearest neighbors."))
    cutoff: float | None = field(
        default=12.0,
        metadata=_metadata("Optional neighbor-search radius.", unit="angstrom"),
    )
    symmetric: bool = field(
        default=True,
        metadata=_metadata("Emit unique reverse directed edges."),
    )
    include_self_loops: bool = field(
        default=False,
        metadata=_metadata("Include one self-loop per residue."),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        knn_mode = _normalize_choice("knn_mode", self.knn_mode, ("local", "global"))
        object.__setattr__(self, "knn_mode", knn_mode)
        object.__setattr__(
            self,
            "k",
            _require_int("k", self.k, minimum=0, inclusive=False),
        )
        cutoff = _require_optional_positive_float("cutoff", self.cutoff)
        object.__setattr__(self, "cutoff", None if knn_mode == "global" else cutoff)
        object.__setattr__(self, "symmetric", _require_bool("symmetric", self.symmetric))
        object.__setattr__(
            self,
            "include_self_loops",
            _require_bool("include_self_loops", self.include_self_loops),
        )


@dataclass(frozen=True, kw_only=True)
class ResidueContactGraphConfig(_ResidueGraphFeatureConfig):
    representation_name: ClassVar[str] = "residue_contact_graph"
    granularity: ClassVar[Literal["residue"]] = "residue"

    cutoff: float = field(
        default=8.0,
        metadata=_metadata("Contact distance cutoff.", unit="angstrom"),
    )
    include_self_loops: bool = field(
        default=False,
        metadata=_metadata("Include one self-loop per residue."),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "cutoff",
            _require_float("cutoff", self.cutoff, minimum=0.0),
        )
        object.__setattr__(
            self,
            "include_self_loops",
            _require_bool("include_self_loops", self.include_self_loops),
        )


@dataclass(frozen=True, kw_only=True)
class ResidueDistogramConfig(_ResidueSelectionConfig):
    representation_name: ClassVar[str] = "residue_distogram"
    granularity: ClassVar[Literal["residue"]] = "residue"

    include_self: bool = field(
        default=True,
        metadata=_metadata("Include diagonal distances."),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "include_self", _require_bool("include_self", self.include_self))


def _normalize_binned_dtype(value: Any) -> str:
    return _normalize_choice("dtype", value, ("uint8", "uint16", "int16"))


@dataclass(frozen=True, kw_only=True)
class ResidueDistogramBinsConfig(_ResidueBinningConfig):
    representation_name: ClassVar[str] = "residue_distogram_bins"
    granularity: ClassVar[Literal["residue"]] = "residue"

    include_self: bool = field(
        default=True,
        metadata=_metadata("Include diagonal distance bins."),
    )
    dtype: Literal["uint8", "uint16", "int16"] = field(
        default="uint8",
        metadata=_metadata(
            "Binned matrix dtype.",
            choices=("uint8", "uint16", "int16"),
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "include_self", _require_bool("include_self", self.include_self))
        object.__setattr__(self, "dtype", _normalize_binned_dtype(self.dtype))


@dataclass(frozen=True, kw_only=True)
class ResidueContactMapConfig(_ResidueSelectionConfig):
    representation_name: ClassVar[str] = "residue_contact_map"
    granularity: ClassVar[Literal["residue"]] = "residue"

    cutoff: float = field(
        default=8.0,
        metadata=_metadata("Contact distance cutoff.", unit="angstrom"),
    )
    include_self: bool = field(
        default=False,
        metadata=_metadata("Include diagonal contacts."),
    )
    dtype: Literal["uint8", "bool"] = field(
        default="uint8",
        metadata=_metadata("Contact matrix dtype.", choices=("uint8", "bool")),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "cutoff",
            _require_float("cutoff", self.cutoff, minimum=0.0),
        )
        object.__setattr__(self, "include_self", _require_bool("include_self", self.include_self))
        object.__setattr__(
            self,
            "dtype",
            _normalize_choice("dtype", self.dtype, ("uint8", "bool")),
        )


@dataclass(frozen=True, kw_only=True)
class ResidueContactMapBinsConfig(_ResidueBinningConfig):
    representation_name: ClassVar[str] = "residue_contact_map_bins"
    granularity: ClassVar[Literal["residue"]] = "residue"

    include_self: bool = field(
        default=False,
        metadata=_metadata("Include diagonal contact bins."),
    )
    dtype: Literal["uint8", "uint16", "int16"] = field(
        default="uint8",
        metadata=_metadata(
            "Binned matrix dtype.",
            choices=("uint8", "uint16", "int16"),
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "include_self", _require_bool("include_self", self.include_self))
        object.__setattr__(self, "dtype", _normalize_binned_dtype(self.dtype))


@dataclass(frozen=True, kw_only=True)
class ResidueVoxelGridConfig(_ResidueSelectionConfig):
    representation_name: ClassVar[str] = "residue_voxel_grid"
    granularity: ClassVar[Literal["residue"]] = "residue"

    grid_size: int = field(default=64, metadata=_metadata("Grid length along each axis."))
    voxel_size: float | None = field(
        default=None,
        metadata=_metadata("Configured voxel spacing.", unit="angstrom"),
    )
    fit: bool = field(
        default=True,
        metadata=_metadata("Derive spacing to fit the selected structure."),
    )
    margin: float = field(
        default=4.0,
        metadata=_metadata("Fit margin around the structure.", unit="angstrom"),
    )
    channels: FeatureMode = field(
        default_factory=_grid_mode_default,
        metadata=_metadata("Requested voxel channel mode."),
    )
    sigma: float = field(
        default=1.0,
        metadata=_metadata("Density Gaussian width.", unit="voxel"),
    )
    pool: Literal["max", "sum", "mean"] = field(
        default="max",
        metadata=_metadata("Voxel accumulation mode.", choices=("max", "sum", "mean")),
    )
    clamp: bool = field(
        default=False,
        metadata=_metadata("Clamp the final grid to the unit interval."),
    )
    dtype: Literal["float32", "float16"] = field(
        default="float32",
        metadata=_metadata("Voxel tensor dtype.", choices=("float32", "float16")),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "grid_size",
            _require_int("grid_size", self.grid_size, minimum=0, inclusive=False),
        )
        object.__setattr__(
            self,
            "voxel_size",
            _require_optional_positive_float("voxel_size", self.voxel_size),
        )
        object.__setattr__(self, "fit", _require_bool("fit", self.fit))
        object.__setattr__(
            self,
            "margin",
            _require_float("margin", self.margin, minimum=0.0, inclusive=True),
        )
        object.__setattr__(
            self,
            "channels",
            FeatureMode.from_value(self.channels, kind="grid"),
        )
        object.__setattr__(
            self,
            "sigma",
            _require_float("sigma", self.sigma, minimum=0.0),
        )
        object.__setattr__(
            self,
            "pool",
            _normalize_choice("pool", self.pool, ("max", "sum", "mean")),
        )
        object.__setattr__(self, "clamp", _require_bool("clamp", self.clamp))
        object.__setattr__(
            self,
            "dtype",
            _normalize_choice("dtype", self.dtype, ("float32", "float16")),
        )


_CONFIG_TYPES_BY_REPRESENTATION: Mapping[
    str,
    type[CanonicalRepresentationConfig],
] = MappingProxyType(
    {
        config_type.representation_name: config_type
        for config_type in (
            AtomPointCloudConfig,
            ResiduePointCloudConfig,
            AtomKNNGraphConfig,
            ResidueKNNGraphConfig,
            ResidueContactGraphConfig,
            ResidueDistogramConfig,
            ResidueDistogramBinsConfig,
            ResidueContactMapConfig,
            ResidueContactMapBinsConfig,
            ResidueVoxelGridConfig,
        )
    }
)
