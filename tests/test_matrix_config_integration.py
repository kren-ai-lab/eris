from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import eris.encoders.geometry.contact_map_bins_ca as contact_bins_module
import eris.encoders.geometry.contact_map_ca as contact_module
import eris.encoders.geometry.distogram_bins_ca as distogram_bins_module
import eris.encoders.geometry.distogram_ca as distogram_module
from eris.core.artifacts import StructureArtifact
from eris.core.base_encoder import EncodingContext
from eris.core.base_encoder.base_encoder import BaseEncoder
from eris.core.config import (
    ResidueContactMapBinsConfig,
    ResidueContactMapConfig,
    ResidueDistogramBinsConfig,
    ResidueDistogramConfig,
)
from eris.core.registry import get_encoder
from eris.core.types import StructureInput
from eris.encoders.geometry.contact_map_bins_ca import ContactMapBinsCA
from eris.encoders.geometry.contact_map_ca import ContactMapCA
from eris.encoders.geometry.distogram_bins_ca import DistogramBinsCA
from eris.encoders.geometry.distogram_ca import DistogramCA

DATA = Path("tests/data/1AQ2.pdb")
INPUT = StructureInput(source=str(DATA))

MatrixEncoder = DistogramCA | DistogramBinsCA | ContactMapCA | ContactMapBinsCA
MatrixConfig = (
    ResidueDistogramConfig
    | ResidueDistogramBinsConfig
    | ResidueContactMapConfig
    | ResidueContactMapBinsConfig
)


def _encode(encoder: MatrixEncoder) -> StructureArtifact:
    return encoder.encode(INPUT)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _assert_same_data(first: StructureArtifact, second: StructureArtifact) -> None:
    assert first.data.keys() == second.data.keys()
    for key in first.data:
        np.testing.assert_array_equal(first.data[key], second.data[key])


def _context(pos: np.ndarray) -> EncodingContext:
    pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
    n = pos.shape[0]
    raw = {
        "pos": pos,
        "residue_ids": np.asarray([f"A:{index}" for index in range(n)]),
        "resname": np.asarray(["ALA"] * n),
        "aa_index": np.zeros(n, dtype=np.int16),
        "plddt": np.zeros(n, dtype=np.float32),
        "bfactor_ca": np.zeros(n, dtype=np.float32),
    }
    return EncodingContext(
        inp=INPUT,
        raw_structure=raw,
        centroid=np.zeros((1, 3), dtype=np.float32),
        pos=pos,
    )


@pytest.fixture(scope="module")
def default_artifacts() -> dict[str, tuple[MatrixEncoder, StructureArtifact]]:
    encoders: tuple[MatrixEncoder, ...] = (
        DistogramCA(use_cache=False, cache=object()),
        DistogramBinsCA(use_cache=False, cache=object()),
        ContactMapCA(use_cache=False, cache=object()),
        ContactMapBinsCA(use_cache=False, cache=object()),
    )
    return {encoder.name: (encoder, _encode(encoder)) for encoder in encoders}


@pytest.mark.parametrize(
    ("encoder", "expected_config"),
    [
        (DistogramCA(), ResidueDistogramConfig()),
        (DistogramBinsCA(), ResidueDistogramBinsConfig()),
        (ContactMapCA(), ResidueContactMapConfig()),
        (ContactMapBinsCA(), ResidueContactMapBinsConfig()),
    ],
)
def test_default_encoder_state_is_owned_by_typed_config(
    encoder: MatrixEncoder,
    expected_config: MatrixConfig,
) -> None:
    assert encoder.config == expected_config
    assert set(vars(encoder)) == {"config", "use_cache", "cache"}
    assert encoder.get_params() == expected_config.to_dict()
    assert encoder.get_params()["scale"] is None


@pytest.mark.parametrize(
    ("encoder_type", "config", "kwargs"),
    [
        (
            DistogramCA,
            ResidueDistogramConfig(
                center=False,
                scale=2.0,
                residue_center="mean",
                include_self=False,
            ),
            {
                "center": False,
                "scale": 2.0,
                "residue_center": "mean",
                "include_self": False,
            },
        ),
        (
            DistogramBinsCA,
            ResidueDistogramBinsConfig(num_bins=12, dtype="uint16"),
            {"num_bins": 12, "dtype": "uint16"},
        ),
        (
            ContactMapCA,
            ResidueContactMapConfig(cutoff=10.0, dtype="bool"),
            {"cutoff": 10.0, "dtype": "bool"},
        ),
        (
            ContactMapBinsCA,
            ResidueContactMapBinsConfig(edges=[2, 4, 8]),
            {"edges": [2, 4, 8]},
        ),
    ],
)
def test_flat_kwargs_and_config_construction_are_equivalent(
    encoder_type: type[MatrixEncoder],
    config: MatrixConfig,
    kwargs: dict[str, Any],
) -> None:
    flat = encoder_type(use_cache=False, cache=object(), **kwargs)
    canonical = encoder_type(config=config, use_cache=False, cache=object())

    assert flat.config == canonical.config == config
    assert flat.get_params() == canonical.get_params()
    assert flat.fingerprint() == canonical.fingerprint()
    _assert_same_data(_encode(flat), _encode(canonical))


@pytest.mark.parametrize(
    ("name", "config", "kwargs"),
    [
        ("residue_distogram", ResidueDistogramConfig(include_self=False), {"include_self": False}),
        (
            "residue_distogram_bins",
            ResidueDistogramBinsConfig(num_bins=12),
            {"num_bins": 12},
        ),
        ("residue_contact_map", ResidueContactMapConfig(cutoff=10), {"cutoff": 10}),
        (
            "residue_contact_map_bins",
            ResidueContactMapBinsConfig(edges=[2, 4, 8]),
            {"edges": [2, 4, 8]},
        ),
    ],
)
def test_registry_supports_config_and_flat_kwargs(
    name: str,
    config: MatrixConfig,
    kwargs: dict[str, Any],
) -> None:
    flat = get_encoder(name, **kwargs)
    canonical = get_encoder(name, config=config)

    assert flat.config == canonical.config == config
    assert flat.fingerprint() == canonical.fingerprint()


@pytest.mark.parametrize(
    ("encoder_type", "config", "kwargs"),
    [
        (DistogramCA, ResidueDistogramConfig(), {"include_self": False}),
        (DistogramBinsCA, ResidueDistogramBinsConfig(), {"num_bins": 12}),
        (ContactMapCA, ResidueContactMapConfig(), {"cutoff": 10}),
        (ContactMapBinsCA, ResidueContactMapBinsConfig(), {"edges": [2, 4, 8]}),
    ],
)
def test_config_rejects_simultaneous_representation_kwargs(
    encoder_type: type[MatrixEncoder],
    config: MatrixConfig,
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(TypeError, match="config cannot be combined"):
        encoder_type(config=config, **kwargs)


def test_constructors_reject_wrong_config_types() -> None:
    with pytest.raises(TypeError, match="ResidueDistogramConfig"):
        DistogramCA(config=ResidueContactMapConfig())
    with pytest.raises(TypeError, match="ResidueDistogramBinsConfig"):
        DistogramBinsCA(config=ResidueContactMapBinsConfig())
    with pytest.raises(TypeError, match="ResidueContactMapConfig"):
        ContactMapCA(config=ResidueDistogramConfig())
    with pytest.raises(TypeError, match="ResidueContactMapBinsConfig"):
        ContactMapBinsCA(config=ResidueDistogramBinsConfig())


@pytest.mark.parametrize(
    ("encoder_type", "kwargs", "error", "message"),
    [
        (DistogramCA, {"residue_center": "invalid"}, ValueError, "residue_center"),
        (DistogramBinsCA, {"dtype": "int8"}, ValueError, "dtype"),
        (ContactMapCA, {"cutoff": 0}, ValueError, "cutoff"),
        (ContactMapBinsCA, {"edges": np.asarray([2, 4])}, TypeError, "edges"),
        (ContactMapCA, {"unknown": True}, TypeError, "unexpected keyword"),
    ],
)
def test_flat_kwargs_use_typed_config_validation(
    encoder_type: type[MatrixEncoder],
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        encoder_type(**kwargs)


@pytest.mark.parametrize(
    ("encoder_type", "config"),
    [
        (DistogramCA, ResidueDistogramConfig(include_self=False)),
        (DistogramBinsCA, ResidueDistogramBinsConfig(num_bins=12)),
        (ContactMapCA, ResidueContactMapConfig(cutoff=10)),
        (ContactMapBinsCA, ResidueContactMapBinsConfig(edges=[2, 4, 8])),
    ],
)
def test_execution_options_are_separate_from_config_identity(
    encoder_type: type[MatrixEncoder],
    config: MatrixConfig,
) -> None:
    cache = object()
    operational = encoder_type(config=config, use_cache=False, cache=cache)
    default_execution = encoder_type(config=config)

    assert operational.config is config
    assert operational.use_cache is False
    assert operational.cache is cache
    assert operational.fingerprint() == default_execution.fingerprint()
    assert "use_cache" not in operational.get_params()
    assert "cache" not in operational.get_params()


def test_representation_properties_are_read_only() -> None:
    raw = DistogramCA()
    binned = DistogramBinsCA()
    contact = ContactMapCA()

    with pytest.raises(AttributeError):
        raw.include_self = False
    with pytest.raises(AttributeError):
        raw.residue_center = "mean"
    with pytest.raises(AttributeError):
        binned.num_bins = 12
    with pytest.raises(AttributeError):
        binned.edges = (2, 4, 8)
    with pytest.raises(AttributeError):
        contact.cutoff = 10
    with pytest.raises(AttributeError):
        contact.dtype = "bool"


def test_custom_constructors_run_base_post_init_exactly_once(monkeypatch) -> None:
    calls: list[BaseEncoder] = []
    original = BaseEncoder.__post_init__

    def spy(encoder: BaseEncoder) -> None:
        calls.append(encoder)
        original(encoder)

    monkeypatch.setattr(BaseEncoder, "__post_init__", spy)
    encoders = (DistogramCA(), DistogramBinsCA(), ContactMapCA(), ContactMapBinsCA())

    assert calls == list(encoders)


@pytest.mark.parametrize(
    ("module", "encoder_type", "config_type"),
    [
        (distogram_module, DistogramCA, ResidueDistogramConfig),
        (distogram_bins_module, DistogramBinsCA, ResidueDistogramBinsConfig),
        (contact_module, ContactMapCA, ResidueContactMapConfig),
        (contact_bins_module, ContactMapBinsCA, ResidueContactMapBinsConfig),
    ],
)
@pytest.mark.parametrize("residue_center", ["ca", "mean", "com"])
def test_residue_policies_are_passed_explicitly_to_loader(
    monkeypatch,
    module: Any,
    encoder_type: type[MatrixEncoder],
    config_type: type[MatrixConfig],
    residue_center: str,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = {"pos": np.ones((1, 3), dtype=np.float32)}

    def fake_load(inp: StructureInput, **kwargs: Any) -> dict[str, np.ndarray]:
        captured["inp"] = inp
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "load_and_extract_residue", fake_load)
    encoder = encoder_type(config=config_type(residue_center=residue_center))

    assert encoder._load_structure(EncodingContext(inp=INPUT)) is sentinel
    assert captured == {
        "inp": INPUT,
        "amino_acids_only": True,
        "mode": residue_center,
        "altloc": "first",
        "missing_ca": "skip",
    }
    assert encoder.granularity_contract.level == "residue"
    assert encoder.config.coordinate_source == {
        "ca": "ca",
        "mean": "residue_mean",
        "com": "residue_com",
    }[residue_center]


def test_default_outputs_match_frozen_scientific_regression(
    default_artifacts: dict[str, tuple[MatrixEncoder, StructureArtifact]],
) -> None:
    expected = {
        "residue_distogram": (
            "dist",
            np.dtype("float32"),
            "71dcd6473890488a19c3d229c2399fcaa0beb67c58fd8ff3a84b04c74d9a9e1f",
        ),
        "residue_distogram_bins": (
            "dist_bins",
            np.dtype("uint8"),
            "5e58169a324aa555f8e8ec0e85da632b287a4cf29cab3b9c64a3dd16567de9cf",
        ),
        "residue_contact_map": (
            "contact",
            np.dtype("uint8"),
            "a89e1d8ca4616c3d80133d72445f70839beb08c0ae4aefbb9ddd12d1e6965745",
        ),
        "residue_contact_map_bins": (
            "contact_bins",
            np.dtype("uint8"),
            "89b01123e31295fdc203ed35d2a25e829bdb260ec0f85f2f2dbfd6929414a408",
        ),
    }

    for name, (encoder, artifact) in default_artifacts.items():
        key, dtype, digest = expected[name]
        matrix = artifact.data[key]
        assert matrix.shape == (534, 534)
        assert matrix.dtype == dtype
        np.testing.assert_array_equal(matrix, matrix.T)
        assert _digest(matrix) == digest
        assert artifact.meta["params"] == encoder.config.to_dict()
        artifact.finalize()
        artifact.validate(strict=True)

    for name in ("residue_distogram_bins", "residue_contact_map_bins"):
        edges = default_artifacts[name][1].data["bin_edges"]
        assert edges.shape == (37,)
        assert edges.dtype == np.float32
        assert edges[0] == 2.0
        assert edges[-1] == 20.0
        assert _digest(edges) == (
            "fcc41ff246ba4075f9acae35d7623c7d0033fd23b72965e6a6fe6981e0f85cb3"
        )


@pytest.mark.parametrize(
    ("encoder", "key", "digest"),
    [
        (
            DistogramCA(residue_center="mean", use_cache=False, cache=object()),
            "dist",
            "4ce15a32b365b4f5cd4f03aeb5c164d7cebb1dc310b5869cbc641bc4dbd988e9",
        ),
        (
            DistogramCA(residue_center="com", use_cache=False, cache=object()),
            "dist",
            "575e9428f0e96089140794bd81afbf3eefc932c362d3486cfa632e1dd07cd25a",
        ),
        (
            DistogramBinsCA(residue_center="mean", use_cache=False, cache=object()),
            "dist_bins",
            "d62f77607d5bed60fb16d490cd75323ae8abe85cad5967eadf9bfb0645dc9fb3",
        ),
        (
            DistogramBinsCA(residue_center="com", use_cache=False, cache=object()),
            "dist_bins",
            "a483951626b1725f3e33e6eefc0649873905e2539ffe44c601ebf6d038e3e4a7",
        ),
        (
            ContactMapCA(residue_center="mean", use_cache=False, cache=object()),
            "contact",
            "638e64f5021d85cfbb46fc585beb4a061a92df696191ef0ab6bafd6b8ccc63a8",
        ),
        (
            ContactMapCA(residue_center="com", use_cache=False, cache=object()),
            "contact",
            "d93c39f27276ac348798543dd0da5e48d52d13ee9e20294a80944c3ec89b1bcb",
        ),
        (
            ContactMapBinsCA(residue_center="mean", use_cache=False, cache=object()),
            "contact_bins",
            "6ec2fe4dde34ed2d9925d32592d8990cae778b8ec7a567753e2d42362ecc8804",
        ),
        (
            ContactMapBinsCA(residue_center="com", use_cache=False, cache=object()),
            "contact_bins",
            "4c0c23aca82822bec94d9186c420aecaf122e5b1f95d8f8ebf7c25aa520bc213",
        ),
    ],
)
def test_mean_and_com_outputs_match_frozen_regression(
    encoder: MatrixEncoder,
    key: str,
    digest: str,
) -> None:
    artifact = _encode(encoder)
    assert artifact.meta["level"] == "residue"
    assert _digest(artifact.data[key]) == digest


@pytest.mark.parametrize(
    ("encoder", "key", "dtype", "digest"),
    [
        (
            DistogramCA(include_self=False, use_cache=False, cache=object()),
            "dist",
            np.dtype("float32"),
            "808ac0ab2608117dea0f731672a14f2b3b75094351c9df8d8ebb81b623d082f4",
        ),
        (
            DistogramBinsCA(include_self=False, use_cache=False, cache=object()),
            "dist_bins",
            np.dtype("uint8"),
            "89b01123e31295fdc203ed35d2a25e829bdb260ec0f85f2f2dbfd6929414a408",
        ),
        (
            ContactMapCA(include_self=True, use_cache=False, cache=object()),
            "contact",
            np.dtype("uint8"),
            "4de5dadb445d3a939e5241bb6e0e867dde59cdedaaca501a58bdf1b43ce4559b",
        ),
        (
            ContactMapBinsCA(include_self=True, use_cache=False, cache=object()),
            "contact_bins",
            np.dtype("uint8"),
            "5e58169a324aa555f8e8ec0e85da632b287a4cf29cab3b9c64a3dd16567de9cf",
        ),
        (
            DistogramBinsCA(edges=[2, 4, 8], use_cache=False, cache=object()),
            "dist_bins",
            np.dtype("uint8"),
            "c02da4d1f6c133bb76aba86d21b6568a229bbc62e6a4a1dd426d2f60bfb9460a",
        ),
        (
            ContactMapBinsCA(edges=[2, 4, 8], use_cache=False, cache=object()),
            "contact_bins",
            np.dtype("uint8"),
            "be9735129a88181353479c8adf9dc43ec6ab77b51c994c451a09081f604bc599",
        ),
        (
            DistogramBinsCA(
                add_no_contact_class=False,
                use_cache=False,
                cache=object(),
            ),
            "dist_bins",
            np.dtype("uint8"),
            "f37d0234001d3b25f68dff1f36b951cb48d21c5c403e9bcd68edb33d79155423",
        ),
        (
            ContactMapBinsCA(
                add_no_contact_class=False,
                use_cache=False,
                cache=object(),
            ),
            "contact_bins",
            np.dtype("uint8"),
            "f37d0234001d3b25f68dff1f36b951cb48d21c5c403e9bcd68edb33d79155423",
        ),
        (
            ContactMapCA(dtype="bool", use_cache=False, cache=object()),
            "contact",
            np.dtype("bool"),
            "a89e1d8ca4616c3d80133d72445f70839beb08c0ae4aefbb9ddd12d1e6965745",
        ),
        (
            DistogramBinsCA(dtype="uint16", use_cache=False, cache=object()),
            "dist_bins",
            np.dtype("uint16"),
            "a5aea13a7599aa4070e7fb859168d395d587fed3ef9b3b0c306628abbecf62a6",
        ),
        (
            ContactMapBinsCA(dtype="int16", use_cache=False, cache=object()),
            "contact_bins",
            np.dtype("int16"),
            "09600665e401ba94cbc1c4fc510c25587e45144b25d15b88c879fc354016a6ea",
        ),
    ],
)
def test_approved_variants_match_frozen_regression_and_remain_strict(
    encoder: MatrixEncoder,
    key: str,
    dtype: np.dtype[Any],
    digest: str,
) -> None:
    artifact = _encode(encoder)

    assert artifact.data[key].dtype == dtype
    assert _digest(artifact.data[key]) == digest
    assert artifact.meta["params"] == encoder.config.to_dict()
    artifact.finalize()
    artifact.validate(strict=True)


def test_raw_distogram_diagonal_symmetry_and_single_residue() -> None:
    one = _context(np.asarray([[1, 2, 3]], dtype=np.float32))
    included = DistogramCA()._create_data(one)["dist"]
    excluded = DistogramCA(include_self=False)._create_data(one)["dist"]

    np.testing.assert_array_equal(included, np.asarray([[0]], dtype=np.float32))
    assert excluded.shape == (1, 1)
    assert np.isnan(excluded[0, 0])


def test_contact_map_preserves_threshold_dtype_and_diagonal_semantics() -> None:
    ctx = _context(np.asarray([[0, 0, 0], [8, 0, 0]], dtype=np.float32))
    excluded = ContactMapCA(cutoff=8)._create_data(ctx)["contact"]
    included = ContactMapCA(cutoff=8, include_self=True)._create_data(ctx)["contact"]
    boolean = ContactMapCA(cutoff=8, dtype="bool")._create_data(ctx)["contact"]

    np.testing.assert_array_equal(excluded, np.asarray([[0, 1], [1, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(included, np.ones((2, 2), dtype=np.uint8))
    assert boolean.dtype == np.bool_
    assert set(np.unique(excluded)) <= {0, 1}


@pytest.mark.parametrize("encoder_type", [DistogramBinsCA, ContactMapBinsCA])
def test_explicit_edges_preserve_boundary_and_no_contact_semantics(
    encoder_type: type[DistogramBinsCA] | type[ContactMapBinsCA],
) -> None:
    ctx = _context(
        np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [9, 0, 0]])
    )
    with_no_contact = encoder_type(edges=[2, 4, 8], include_self=True)._create_data(ctx)
    without_no_contact = encoder_type(
        edges=[2, 4, 8],
        include_self=True,
        add_no_contact_class=False,
    )._create_data(ctx)
    key = "dist_bins" if encoder_type is DistogramBinsCA else "contact_bins"

    np.testing.assert_array_equal(
        with_no_contact[key][0],
        np.asarray([1, 1, 1, 2, 2, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        without_no_contact[key][0],
        np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        with_no_contact["bin_edges"],
        np.asarray([2, 4, 8], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("encoder", "key", "expected"),
    [
        (DistogramBinsCA(include_self=True), "dist_bins", 1),
        (DistogramBinsCA(include_self=False), "dist_bins", 0),
        (ContactMapBinsCA(include_self=True), "contact_bins", 1),
        (ContactMapBinsCA(include_self=False), "contact_bins", 0),
        (
            DistogramBinsCA(include_self=True, add_no_contact_class=False),
            "dist_bins",
            0,
        ),
        (
            ContactMapBinsCA(include_self=True, add_no_contact_class=False),
            "contact_bins",
            0,
        ),
    ],
)
def test_binned_single_residue_diagonal_semantics(
    encoder: DistogramBinsCA | ContactMapBinsCA,
    key: str,
    expected: int,
) -> None:
    matrix = encoder._create_data(_context(np.asarray([[0, 0, 0]])))[key]
    np.testing.assert_array_equal(matrix, np.asarray([[expected]], dtype=np.uint8))


@pytest.mark.parametrize(
    ("encoder_type", "dtype"),
    [
        (DistogramBinsCA, "uint8"),
        (DistogramBinsCA, "uint16"),
        (ContactMapBinsCA, "int16"),
    ],
)
def test_binned_runtime_uses_the_configured_dtype(
    encoder_type: type[DistogramBinsCA] | type[ContactMapBinsCA],
    dtype: str,
) -> None:
    key = "dist_bins" if encoder_type is DistogramBinsCA else "contact_bins"
    matrix = encoder_type(dtype=dtype)._create_data(_context(np.asarray([[0, 0, 0]])))[key]
    assert matrix.dtype == np.dtype(dtype)


@pytest.mark.parametrize("encoder_type", [DistogramBinsCA, ContactMapBinsCA])
@pytest.mark.parametrize(
    ("dtype", "num_bins", "add_no_contact_class", "valid"),
    [
        ("uint8", 255, True, True),
        ("uint8", 256, True, False),
        ("uint8", 256, False, True),
        ("uint8", 257, False, False),
        ("uint16", 65535, True, True),
        ("uint16", 65536, True, False),
        ("int16", 32767, True, True),
        ("int16", 32768, True, False),
    ],
)
def test_generated_bin_capacity_boundaries_are_validated_before_encoding(
    encoder_type: type[DistogramBinsCA] | type[ContactMapBinsCA],
    dtype: str,
    num_bins: int,
    add_no_contact_class: bool,
    valid: bool,
) -> None:
    kwargs = {
        "dtype": dtype,
        "num_bins": num_bins,
        "add_no_contact_class": add_no_contact_class,
    }
    if valid:
        assert encoder_type(**kwargs).config.num_bins == num_bins
    else:
        with pytest.raises(
            ValueError,
            match=rf"{dtype}.*{num_bins} bins.*maximum class ID.*capacity",
        ):
            encoder_type(**kwargs)


@pytest.mark.parametrize("encoder_type", [DistogramBinsCA, ContactMapBinsCA])
def test_explicit_edge_capacity_is_validated_before_encoding(
    encoder_type: type[DistogramBinsCA] | type[ContactMapBinsCA],
) -> None:
    valid_edges = tuple(range(256))
    invalid_edges = tuple(range(257))

    assert encoder_type(edges=valid_edges, dtype="uint8").config.edges == valid_edges
    with pytest.raises(ValueError, match="uint8.*256 bins.*maximum class ID 256.*255"):
        encoder_type(edges=invalid_edges, dtype="uint8")


def test_requested_and_effective_binning_state_remain_separate(
    default_artifacts: dict[str, tuple[MatrixEncoder, StructureArtifact]],
) -> None:
    for name in ("residue_distogram_bins", "residue_contact_map_bins"):
        encoder, artifact = default_artifacts[name]
        assert encoder.config.edges is None
        assert artifact.meta["params"] == encoder.config.to_dict()
        assert artifact.meta["params"]["edges"] is None
        assert artifact.meta["params"]["num_bins"] == 36
        assert artifact.data["bin_edges"].shape == (37,)

    explicit = DistogramBinsCA(edges=[2, 4, 8], use_cache=False, cache=object())
    artifact = _encode(explicit)
    assert explicit.config.edges == (2.0, 4.0, 8.0)
    assert artifact.meta["params"]["edges"] == [2.0, 4.0, 8.0]
    assert artifact.meta["params"]["dmin"] is None
    assert artifact.meta["params"]["dmax"] is None
    assert artifact.meta["params"]["num_bins"] is None
    np.testing.assert_array_equal(
        artifact.data["bin_edges"],
        np.asarray([2, 4, 8], dtype=np.float32),
    )


def test_explicit_edge_list_and_tuple_have_canonical_identity() -> None:
    from_list = ContactMapBinsCA(edges=[2, 4, 8])
    from_tuple = ContactMapBinsCA(edges=(2, 4, 8))

    assert from_list.config == from_tuple.config
    assert from_list.get_params() == from_tuple.get_params()
    assert from_list.fingerprint() == from_tuple.fingerprint()


@pytest.mark.parametrize(
    ("encoder_type", "kwargs"),
    [
        (DistogramCA, {"center": False}),
        (DistogramCA, {"residue_center": "mean"}),
        (DistogramCA, {"scale": 2}),
        (DistogramCA, {"include_self": False}),
        (ContactMapCA, {"cutoff": 10}),
        (ContactMapCA, {"dtype": "bool"}),
        (DistogramBinsCA, {"edges": [2, 4, 8]}),
        (DistogramBinsCA, {"dmin": 1}),
        (DistogramBinsCA, {"dmax": 21}),
        (DistogramBinsCA, {"num_bins": 12}),
        (DistogramBinsCA, {"add_no_contact_class": False}),
        (ContactMapBinsCA, {"dtype": "uint16"}),
    ],
)
def test_semantic_config_changes_fingerprint(
    encoder_type: type[MatrixEncoder],
    kwargs: dict[str, Any],
) -> None:
    assert encoder_type().fingerprint() != encoder_type(**kwargs).fingerprint()
