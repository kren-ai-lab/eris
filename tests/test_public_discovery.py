from __future__ import annotations

import json

import pytest

import eris
from eris.core.learned import BackendAvailability, learned_registry
from eris.core.learned.backends.gearnet import dependencies
from eris.core.learned.backends.gearnet.backend import GearNetEdgeBackend
from eris.core.learned.backends.gearnet.metadata import GEARNET_EDGE_ENCODER_VERSION
from eris.core.registry import get_encoder_info, list_encoders
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from eris.exceptions import RepresentationNotFoundError

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


def _fake_probe(monkeypatch, availability_probe) -> None:
    def failing_init(self):  # noqa: ARG001
        raise AssertionError("backend construction must not run during discovery")

    monkeypatch.setattr(GearNetEdgeBackend, "__init__", failing_init)
    monkeypatch.setattr(dependencies, "probe_gearnet_availability", availability_probe)


def test_root_exports_documented_public_surface_without_learned_execution():
    assert eris.__all__ == [
        "RepresentationNotFoundError",
        "__version__",
        "batch_embed",
        "batch_encode",
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


def test_public_listings_are_registry_backed_and_separate():
    assert eris.list_representations() == list_encoders()
    assert eris.list_embedders() == ["gearnet_edge"] == learned_registry.names()
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
    _fake_probe(
        monkeypatch,
        lambda: BackendAvailability(
            available=False,
            reason="Install the supported GearNet environment.",
        ),
    )

    info = eris.get_embedder_info("gearnet_edge")

    assert set(info) == EMBEDDER_INFO_KEYS
    assert info["name"] == "gearnet_edge"
    assert info["encoder_version"] == GEARNET_EDGE_ENCODER_VERSION
    assert info["availability"] == {
        "available": False,
        "reason": "Install the supported GearNet environment.",
    }
    json.dumps(info)


def test_returned_metadata_is_a_defensive_nested_copy(monkeypatch):
    _fake_probe(monkeypatch, lambda: BackendAvailability(available=True))

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

    _fake_probe(monkeypatch, failing_probe)

    assert eris.list_embedders() == ["gearnet_edge"]
    assert calls == {"probe": 0}


def test_get_embedder_info_probes_once_without_constructing_backend(monkeypatch):
    calls = {"probe": 0}

    def probe():
        calls["probe"] += 1
        return BackendAvailability(available=True)

    _fake_probe(monkeypatch, probe)

    assert eris.get_embedder_info("gearnet_edge")["availability"] == {
        "available": True,
        "reason": None,
    }
    assert calls == {"probe": 1}


def test_representation_info_does_not_instantiate_encoder(monkeypatch):
    def failing_init(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("encoder construction must not run during discovery")

    monkeypatch.setattr(ResidueKNNGraph, "__init__", failing_init)

    assert eris.get_representation_info("residue_knn_graph")["type"] == "graph"
