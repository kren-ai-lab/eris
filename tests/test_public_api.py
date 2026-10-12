from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import eris
import eris.api as api_module
import eris.core.base_encoder.base_encoder as base_encoder_module
from eris.core.artifacts import StructureArtifact
from eris.core.cache import DiskCache
from eris.core.config import ResidueContactGraphConfig, ResidueKNNGraphConfig
from eris.core.io import StructureLoadError
from eris.core.registry import get_encoder
from eris.core.types import StructureInput
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA
from eris.exceptions import ConfigError, RepresentationNotFoundError

PDB = Path("tests/data/1AQ2.pdb")


def _write_multichain_structure(path: Path) -> Path:
    lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  ",
        "TER",
        "ATOM      2  CA  GLY B   1       1.000   0.000   0.000  1.00 20.00           C  ",
        "TER",
        "HETATM    3  C1  LIG C   1       2.000   0.000   0.000  1.00 30.00           C  ",
        "TER",
        "HETATM    4  O   HOH D   1       3.000   0.000   0.000  1.00 40.00           O  ",
        "TER",
        "END",
        "",
    ]
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def _assert_equivalent(left: StructureArtifact, right: StructureArtifact) -> None:
    assert left.meta == right.meta
    assert list(left.data) == list(right.data)
    for name in left.data:
        np.testing.assert_array_equal(left.data[name], right.data[name])


def test_all_discovered_explicit_representations_encode_strict_artifacts():
    for name in eris.list_representations():
        artifact = eris.encode(str(PDB), representation=name, use_cache=False)

        assert isinstance(artifact, StructureArtifact)
        assert artifact.meta["schema_version"] == 1
        assert artifact.meta["representation_name"] == name
        artifact.validate(strict=True)


def test_public_pool_aggregates_real_residue_node_features_without_mutation():
    source = eris.encode(PDB, representation="residue_point_cloud", use_cache=False)
    source_values = source.data["node_attr"].copy()

    pooled = eris.pool(source)

    np.testing.assert_allclose(
        pooled.data["pooled_attr"],
        np.mean(source_values.astype(np.float64), axis=0),
    )
    assert pooled.data["pooled_attr"].shape == (source_values.shape[1],)
    assert pooled.meta["coordinate_source"] == source.meta["coordinate_source"]
    assert pooled.meta["parameters"]["source"] == "node_attr"
    np.testing.assert_array_equal(source.data["node_attr"], source_values)


@pytest.mark.parametrize(
    "representation",
    ["residue_knn_graph", "residue_voxel_grid"],
)
def test_default_public_and_low_level_encoding_are_equivalent(representation):
    public = eris.encode(PDB, representation=representation, use_cache=False)
    low_level = get_encoder(representation, use_cache=False).encode(StructureInput(source=str(PDB)))

    _assert_equivalent(public, low_level)


def test_parameter_override_matches_low_level_and_changes_config_identity():
    public = eris.encode(
        PDB,
        representation="residue_knn_graph",
        use_cache=False,
        k=8,
    )
    low_level = get_encoder("residue_knn_graph", use_cache=False, k=8).encode(
        StructureInput(source=str(PDB))
    )
    default = eris.encode(
        PDB,
        representation="residue_knn_graph",
        use_cache=False,
    )

    _assert_equivalent(public, low_level)
    assert public.meta["parameters"]["k"] == 8
    assert (
        public.meta["provenance"]["config_fingerprint"]
        != default.meta["provenance"]["config_fingerprint"]
    )


def test_typed_config_is_passed_through_and_matches_low_level(monkeypatch):
    config = ResidueKNNGraphConfig(k=7)
    captured = {}
    real_get_encoder = api_module.get_encoder

    def capture_config(name, **kwargs):
        captured["config"] = kwargs["config"]
        return real_get_encoder(name, **kwargs)

    monkeypatch.setattr(api_module, "get_encoder", capture_config)
    public = eris.encode(
        PDB,
        representation="residue_knn_graph",
        config=config,
        use_cache=False,
    )
    low_level = get_encoder(
        "residue_knn_graph",
        config=config,
        use_cache=False,
    ).encode(StructureInput(source=str(PDB)))

    assert captured["config"] is config
    _assert_equivalent(public, low_level)


def test_str_path_and_structure_input_are_equivalent():
    structure_input = StructureInput(source=str(PDB))
    artifacts = [
        eris.encode(str(PDB), representation="residue_contact_map", use_cache=False),
        eris.encode(PDB, representation="residue_contact_map", use_cache=False),
        eris.encode(
            structure_input,
            representation="residue_contact_map",
            use_cache=False,
        ),
    ]

    _assert_equivalent(artifacts[0], artifacts[1])
    _assert_equivalent(artifacts[0], artifacts[2])


def test_path_input_applies_explicit_structure_selection():
    artifact = eris.encode(
        PDB,
        representation="residue_contact_map",
        chain_id="A",
        model_index=0,
        use_cache=False,
    )

    assert artifact.meta["input"]["source"] == str(PDB)
    assert artifact.meta["input"]["chain_ids"] == ["A"]
    assert artifact.meta["input"]["model_index"] == 0


def test_multichain_selection_is_canonical_and_excludes_nonprotein_chains(tmp_path: Path):
    source = _write_multichain_structure(tmp_path / "multi.pdb")

    default = eris.encode(source, representation="residue_point_cloud", use_cache=False)
    reversed_selection = eris.encode(
        source,
        representation="residue_point_cloud",
        chain_id=["B", "A"],
        use_cache=False,
    )
    scalar = eris.encode(
        source,
        representation="residue_point_cloud",
        chain_id="B",
        use_cache=False,
    )

    assert default.meta["input"]["chain_ids"] == ["A", "B"]
    assert default.data["chain_id"].tolist() == ["A", "B"]
    assert reversed_selection.meta["input"]["chain_ids"] == ["A", "B"]
    assert (
        reversed_selection.meta["provenance"]["config_fingerprint"]
        == default.meta["provenance"]["config_fingerprint"]
    )
    assert scalar.meta["input"]["chain_ids"] == ["B"]
    assert scalar.data["residue_ids"].tolist() == ["B:1::GLY"]
    assert (
        scalar.meta["provenance"]["config_fingerprint"]
        != default.meta["provenance"]["config_fingerprint"]
    )

    cache = DiskCache(tmp_path / "cache")
    encoder = ResiduePointCloudCA(use_cache=True, cache=cache)
    first = encoder.encode(StructureInput(source=str(source), chain_id=["A", "B"]))
    reordered = encoder.encode(StructureInput(source=str(source), chain_id=["B", "A"]))
    encoder.encode(StructureInput(source=str(source), chain_id="B"))
    np.testing.assert_array_equal(first.data["residue_ids"], reordered.data["residue_ids"])
    assert len(list((cache.cache_dir / "artifacts").iterdir())) == 2


@pytest.mark.parametrize("chain_id", [[], ["A", "A"]])
def test_invalid_multichain_requests_fail_before_execution(chain_id):
    with pytest.raises((ConfigError, ValueError), match="empty|duplicate|must not be empty"):
        eris.encode(
            PDB,
            representation="residue_point_cloud",
            chain_id=chain_id,
            use_cache=False,
        )


def test_unknown_chain_request_fails_clearly(tmp_path: Path):
    source = _write_multichain_structure(tmp_path / "multi.pdb")
    with pytest.raises(StructureLoadError, match="not found"):
        eris.encode(
            source,
            representation="residue_point_cloud",
            chain_id=["A", "Z"],
            use_cache=False,
        )


def test_all_explicit_residue_representations_support_multichain_selection(tmp_path: Path):
    source = _write_multichain_structure(tmp_path / "all-residue.pdb")
    residue_names = [
        name
        for name in eris.list_representations()
        if eris.get_representation_info(name)["granularity"] == "residue"
    ]

    for name in residue_names:
        artifact = eris.encode(source, representation=name, use_cache=False)
        assert artifact.meta["input"]["chain_ids"] == ["A", "B"]
        assert artifact.data["chain_id"].tolist() == ["A", "B"]


def test_multichain_selection_is_resolved_independently_per_model(tmp_path: Path):
    source = tmp_path / "models.pdb"
    source.write_text(
        "\n".join(
            [
                "MODEL        1",
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  ",
                "TER",
                "ENDMDL",
                "MODEL        2",
                "ATOM      2  CA  GLY B   1       1.000   0.000   0.000  1.00 20.00           C  ",
                "TER",
                "ENDMDL",
                "END",
                "",
            ]
        ),
        encoding="ascii",
    )

    first = eris.encode(
        source, representation="residue_point_cloud", model_index=0, use_cache=False
    )
    second = eris.encode(
        source, representation="residue_point_cloud", model_index=1, use_cache=False
    )

    assert first.meta["input"]["chain_ids"] == ["A"]
    assert first.data["residue_ids"].tolist() == ["A:1::ALA"]
    assert second.meta["input"]["chain_ids"] == ["B"]
    assert second.data["residue_ids"].tolist() == ["B:1::GLY"]


def test_default_selection_includes_parser_identified_noncanonical_polymer_chain(
    tmp_path: Path,
):
    source = tmp_path / "noncanonical.pdb"
    source.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  ",
                "TER",
                "ATOM      2  CA  MSE B   1       1.000   0.000   0.000  1.00 20.00           C  ",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="ascii",
    )

    artifact = eris.encode(source, representation="residue_point_cloud", use_cache=False)

    assert artifact.meta["input"]["chain_ids"] == ["A", "B"]
    assert artifact.data["chain_id"].tolist() == ["A", "B"]
    assert artifact.data["resname"].tolist() == ["ALA", "MSE"]


def test_structure_input_preserves_identity_and_allows_identical_overrides(monkeypatch):
    structure_input = StructureInput(source=str(PDB), chain_id="A", model_index=1)
    expected = get_encoder("residue_contact_map", use_cache=False).encode(
        StructureInput(source=str(PDB))
    )
    captured = []

    class CapturingEncoder:
        def encode(self, value):
            captured.append(value)
            return expected

    monkeypatch.setattr(api_module, "get_encoder", lambda *args, **kwargs: CapturingEncoder())
    without_overrides = eris.encode(
        structure_input,
        representation="residue_contact_map",
        use_cache=False,
    )
    with_identical_overrides = eris.encode(
        structure_input,
        representation="residue_contact_map",
        chain_id="A",
        model_index=1,
        use_cache=False,
    )

    assert without_overrides is expected
    assert with_identical_overrides is expected
    assert captured == [structure_input, structure_input]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"config": ResidueKNNGraphConfig(), "k": 8},
            "config cannot be combined",
        ),
        (
            {"config": ResidueContactGraphConfig()},
            "must be an ResidueKNNGraphConfig",
        ),
    ],
)
def test_invalid_config_combinations_fail_before_encoder(monkeypatch, kwargs, match):
    def fail_encoder(*args, **values):
        del args, values
        raise AssertionError("encoder must not be constructed")

    monkeypatch.setattr(api_module, "get_encoder", fail_encoder)
    with pytest.raises(ConfigError, match=match):
        eris.encode(PDB, representation="residue_knn_graph", **kwargs)


@pytest.mark.parametrize(
    "override",
    [
        {"chain_id": "B"},
    ],
)
def test_contradictory_structure_input_overrides_fail_before_encoder(monkeypatch, override):
    structure_input = StructureInput(source=str(PDB), chain_id="A", model_index=1)

    def fail_encoder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("encoder must not be constructed")

    monkeypatch.setattr(api_module, "get_encoder", fail_encoder)
    with pytest.raises(ConfigError, match="override conflicts with StructureInput"):
        eris.encode(
            structure_input,
            representation="residue_knn_graph",
            **override,
        )


@pytest.mark.parametrize("representation", ["missing"])
def test_unknown_or_learned_names_use_public_not_found_error(representation):
    with pytest.raises(
        RepresentationNotFoundError,
        match=f"Unknown representation '{representation}'",
    ):
        eris.encode(PDB, representation=representation)


def test_output_none_does_not_persist_user_output(monkeypatch):
    def fail_save(*args, **kwargs):
        del args, kwargs
        raise AssertionError("output=None must not save an artifact")

    monkeypatch.setattr(StructureArtifact, "save", fail_save)

    artifact = eris.encode(
        PDB,
        representation="residue_contact_map",
        use_cache=False,
    )
    assert isinstance(artifact, StructureArtifact)


def test_explicit_output_saves_same_artifact_and_round_trips(tmp_path, monkeypatch):
    output = tmp_path / "artifact"
    saved = []
    real_save = StructureArtifact.save

    def capture_save(self, outdir, **kwargs):
        saved.append(self)
        return real_save(self, outdir, **kwargs)

    monkeypatch.setattr(StructureArtifact, "save", capture_save)
    artifact = eris.encode(
        PDB,
        representation="residue_knn_graph",
        output=output,
        use_cache=False,
    )

    assert saved == [artifact]
    assert (output / "meta.json").is_file()
    assert (output / "data.npz").is_file()
    loaded = StructureArtifact.load(output, strict=True)
    _assert_equivalent(artifact, loaded)


def test_cache_hit_is_persisted_to_independent_user_output(tmp_path, monkeypatch):
    cache = DiskCache(tmp_path / "cache")
    monkeypatch.setattr(base_encoder_module, "DiskCache", lambda: cache)

    first = eris.encode(
        PDB,
        representation="residue_point_cloud",
        use_cache=True,
    )

    def fail_create_data(self, ctx):
        del self, ctx
        raise AssertionError("second call must use the internal artifact cache")

    monkeypatch.setattr(ResiduePointCloudCA, "_create_data", fail_create_data)
    output = tmp_path / "user-output"
    second = eris.encode(
        PDB,
        representation="residue_point_cloud",
        output=output,
        use_cache=True,
    )

    _assert_equivalent(first, second)
    _assert_equivalent(second, StructureArtifact.load(output, strict=True))
    assert output.parent != cache.cache_dir


def test_root_import_and_explicit_encode_do_not_import_optional_frameworks():
    script = """
import sys
import eris

artifact = eris.encode(
    'tests/data/1AQ2.pdb',
    representation='residue_contact_map',
    use_cache=False,
)
assert artifact.meta['representation_name'] == 'residue_contact_map'
assert hasattr(eris, 'embed')
blocked = {'torch', 'torchdrug', 'torch_scatter', 'torch_cluster', 'rdkit'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)
