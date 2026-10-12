from __future__ import annotations

import json
import subprocess
import sys

import pytest

import eris
import eris.discovery as discovery
from eris.core.learned import BackendAvailability, EmbedderRegistry, learned_registry
from eris.core.learned.backends.gearnet import GearNetEdgeConfig
from eris.core.learned.backends.gearnet.dependencies import probe_gearnet_availability
from eris.core.registry import get_encoder_info, list_encoders
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from eris.exceptions import RepresentationNotFoundError

EXPLICIT_NAMES = [
    "atom_knn_graph",
    "atom_point_cloud",
    "residue_contact_graph",
    "residue_contact_map",
    "residue_contact_map_bins",
    "residue_distogram",
    "residue_distogram_bins",
    "residue_knn_graph",
    "residue_point_cloud",
    "residue_voxel_grid",
]

REPRESENTATION_INFO_KEYS = {
    "name",
    "family",
    "type",
    "granularity",
    "coordinate_source",
    "description",
    "defaults",
    "parameters",
}

EMBEDDER_INFO_KEYS = {
    "name",
    "family",
    "type",
    "coordinate_source",
    "backend_id",
    "encoder_version",
    "output_modes",
    "granularity_by_output_mode",
    "parameters",
    "availability",
}


def _failing_factory():
    raise AssertionError("backend factory must not run during discovery")


def _registry(*, availability_probe) -> EmbedderRegistry:
    registry = EmbedderRegistry()
    registry.register(
        representation_name="gearnet_edge",
        backend_id="gearnet-edge",
        implementation="fake.module.GearNetEdgeBackend",
        encoder_version="1",
        config_type=GearNetEdgeConfig,
        backend_factory=_failing_factory,
        availability_probe=availability_probe,
    )
    return registry


def test_root_exports_documented_public_surface_without_learned_execution():
    assert eris.__all__ == [
        "RepresentationNotFoundError",
        "__version__",
        "batch_embed",
        "batch_encode",
        "batch_export",
        "embed",
        "encode",
        "get_embedder_info",
        "get_representation_info",
        "list_embedders",
        "list_representations",
        "pack",
        "pool",
        "to_pyg",
        "to_torch",
    ]
    assert eris.RepresentationNotFoundError is RepresentationNotFoundError
    assert callable(eris.encode)
    assert hasattr(eris, "embed")
    assert not hasattr(eris, "export_pyg")
    assert not hasattr(eris, "export_torch")
    assert not hasattr(eris, "pack_artifacts")
    assert not hasattr(eris, "save_artifact_to_pyg")
    assert not hasattr(eris, "save_artifact_to_torch")


def test_public_listings_are_exact_deterministic_and_registry_backed():
    assert eris.list_representations() == EXPLICIT_NAMES == list_encoders()
    assert eris.list_representations() == eris.list_representations()
    assert eris.list_embedders() == ["gearnet_edge"] == learned_registry.names()
    assert eris.list_embedders() == eris.list_embedders()
    assert "gearnet_edge" not in eris.list_representations()


def test_representation_info_has_exact_public_json_safe_shape():
    info = eris.get_representation_info("residue_knn_graph")

    assert set(info) == REPRESENTATION_INFO_KEYS
    assert info["name"] == "residue_knn_graph"
    assert info["family"] == "explicit"
    assert info["type"] == "graph"
    assert info["granularity"] == "residue"
    assert info["coordinate_source"] == "ca"
    assert isinstance(info["description"], str)
    assert info["defaults"] == get_encoder_info("residue_knn_graph").defaults()
    assert info["parameters"] == get_encoder_info("residue_knn_graph").params_schema()
    json.dumps(info)


def test_embedder_info_has_exact_public_json_safe_shape(monkeypatch):
    registry = _registry(
        availability_probe=lambda: BackendAvailability(
            available=False,
            reason="Install the supported GearNet environment.",
        )
    )
    monkeypatch.setattr(discovery, "_get_learned_registry", lambda: registry)

    info = eris.get_embedder_info("gearnet_edge")

    assert set(info) == EMBEDDER_INFO_KEYS
    assert info == {
        "name": "gearnet_edge",
        "family": "learned",
        "type": "embedding",
        "coordinate_source": "model_specific",
        "backend_id": "gearnet-edge",
        "encoder_version": "1",
        "output_modes": ["node", "global", "both"],
        "granularity_by_output_mode": {
            "node": "residue",
            "global": "structure",
            "both": "mixed",
        },
        "parameters": [
            {
                "name": "output_mode",
                "type": "'node' | 'global' | 'both'",
                "default": "both",
                "required": False,
                "nullable": False,
                "description": None,
                "choices": ["node", "global", "both"],
            }
        ],
        "availability": {
            "available": False,
            "reason": "Install the supported GearNet environment.",
        },
    }
    json.dumps(info)


def test_returned_metadata_is_a_defensive_nested_copy(monkeypatch):
    registry = _registry(availability_probe=lambda: BackendAvailability(available=True))
    monkeypatch.setattr(discovery, "_get_learned_registry", lambda: registry)

    explicit = eris.get_representation_info("residue_knn_graph")
    explicit["defaults"]["center"] = False
    explicit["parameters"][0]["description"] = "mutated"

    learned = eris.get_embedder_info("gearnet_edge")
    learned["output_modes"].append("mutated")
    learned["parameters"][0]["choices"].append("mutated")
    learned["availability"]["available"] = False

    next_explicit = eris.get_representation_info("residue_knn_graph")
    next_learned = eris.get_embedder_info("gearnet_edge")
    assert next_explicit["defaults"]["center"] is True
    assert next_explicit["parameters"][0]["description"] != "mutated"
    assert next_learned["output_modes"] == ["node", "global", "both"]
    assert next_learned["parameters"][0]["choices"] == ["node", "global", "both"]
    assert next_learned["availability"]["available"] is True


def test_public_getters_translate_only_unknown_registry_names():
    with pytest.raises(
        RepresentationNotFoundError,
        match="Unknown representation 'missing'",
    ):
        eris.get_representation_info("missing")

    with pytest.raises(
        RepresentationNotFoundError,
        match="Unknown embedder 'missing'",
    ):
        eris.get_embedder_info("missing")


def test_list_embedders_neither_probes_nor_constructs_backend(monkeypatch):
    calls = {"probe": 0}

    def failing_probe():
        calls["probe"] += 1
        raise AssertionError("availability must not run during listing")

    registry = _registry(availability_probe=failing_probe)
    monkeypatch.setattr(discovery, "_get_learned_registry", lambda: registry)

    assert eris.list_embedders() == ["gearnet_edge"]
    assert calls == {"probe": 0}


def test_get_embedder_info_probes_once_without_constructing_backend(monkeypatch):
    calls = {"probe": 0}

    def probe():
        calls["probe"] += 1
        return BackendAvailability(available=True)

    registry = _registry(availability_probe=probe)
    monkeypatch.setattr(discovery, "_get_learned_registry", lambda: registry)

    assert eris.get_embedder_info("gearnet_edge")["availability"] == {
        "available": True,
        "reason": None,
    }
    assert calls == {"probe": 1}


def test_real_embedder_discovery_supports_modern_core_python():
    availability = probe_gearnet_availability()
    info = eris.get_embedder_info("gearnet_edge")

    assert "gearnet_edge" in eris.list_embedders()
    assert info["availability"] == {
        "available": availability.available,
        "reason": availability.reason,
    }


def test_representation_info_does_not_instantiate_encoder(monkeypatch):
    def failing_init(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("encoder construction must not run during discovery")

    monkeypatch.setattr(ResidueKNNGraph, "__init__", failing_init)

    assert eris.get_representation_info("residue_knn_graph")["type"] == "graph"


def test_core_only_public_discovery_does_not_import_optional_frameworks():
    script = """
import sys
import eris

assert eris.list_representations()
assert eris.list_embedders() == ['gearnet_edge']
blocked = {'torch', 'torchdrug', 'torch_scatter', 'torch_cluster', 'rdkit', 'biotite'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_discovery_preserves_separate_registries():
    explicit_before = list_encoders()
    learned_before = learned_registry.names()

    eris.list_representations()
    eris.list_embedders()

    assert list_encoders() == explicit_before == EXPLICIT_NAMES
    assert learned_registry.names() == learned_before == ["gearnet_edge"]
