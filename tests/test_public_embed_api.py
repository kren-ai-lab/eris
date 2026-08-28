from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import eris
import eris.api as api_module
from eris.core.artifacts import StructureArtifact
from eris.core.types import StructureInput
from eris.exceptions import ConfigError, ExecutionError, RepresentationNotFoundError

PDB = Path("tests/data/1AQ2.pdb")


def test_root_exports_public_embed_without_internal_learned_exports():
    assert eris.embed is api_module.embed
    assert "embed" in eris.__all__
    assert not hasattr(eris, "StructuralEmbedderRunner")
    assert not hasattr(eris, "StructuralExecutionConfig")
    assert not hasattr(eris, "learned_registry")
    signature = inspect.signature(eris.embed)
    assert signature.parameters["representation"].default is inspect.Parameter.empty
    assert signature.parameters["output_mode"].default == "both"
    assert signature.parameters["checkpoint_path"].default is None
    assert signature.parameters["device"].default == "auto"
    assert signature.parameters["precision"].default == "fp32"
    assert signature.return_annotation == "StructureArtifact"


def test_embed_delegates_registry_config_execution_and_checkpoint(monkeypatch):
    captured = {}
    real_runner = __import__("eris.core.learned", fromlist=["StructuralEmbedderRunner"])

    class CapturingRunner:
        def __init__(self):
            pass

        def run(self, structure_input, **kwargs):
            captured["input"] = structure_input
            captured.update(kwargs)
            return eris.encode(PDB, representation="residue_contact_map", use_cache=False)

    monkeypatch.setattr(real_runner, "StructuralEmbedderRunner", CapturingRunner)
    artifact = eris.embed(
        str(PDB),
        representation="gearnet_edge",
        chain_id="A",
        model_index=0,
        output_mode="node",
        checkpoint_path=PDB,
        device="cpu",
        precision="fp32",
        use_cache=False,
    )

    assert isinstance(artifact, StructureArtifact)
    assert captured["input"] == StructureInput(source=str(PDB), chain_id="A", model_index=0)
    assert captured["config"].representation_name == "gearnet_edge"
    assert captured["config"].output_mode == "node"
    assert captured["model"].architecture == "GearNet-Edge"
    assert captured["model"].checkpoint.identifier == "mc_gearnet_edge.pth"
    assert captured["checkpoint_path"] == PDB
    assert captured["execution"].device == "cpu"
    assert captured["execution"].precision == "fp32"
    assert captured["use_artifact_cache"] is False


def test_embed_accepts_structure_input_and_preserves_it(monkeypatch):
    learned = __import__("eris.core.learned", fromlist=["StructuralEmbedderRunner"])
    structure_input = StructureInput(source=str(PDB), chain_id="A", model_index=1)
    captured = []

    class CapturingRunner:
        def run(self, value, **kwargs):
            captured.append((value, kwargs))
            return eris.encode(PDB, representation="residue_contact_map", use_cache=False)

    monkeypatch.setattr(learned, "StructuralEmbedderRunner", CapturingRunner)
    eris.embed(structure_input, representation="gearnet_edge")
    assert captured[0][0] is structure_input

    with pytest.raises(ConfigError, match="override conflicts with StructureInput"):
        eris.embed(structure_input, representation="gearnet_edge", model_index=0)


@pytest.mark.parametrize("output_mode", ["node", "global", "both"])
def test_embed_accepts_only_contract_output_modes(monkeypatch, output_mode):
    learned = __import__("eris.core.learned", fromlist=["StructuralEmbedderRunner"])

    class CapturingRunner:
        def run(self, value, **kwargs):
            return eris.encode(PDB, representation="residue_contact_map", use_cache=False)

    monkeypatch.setattr(learned, "StructuralEmbedderRunner", CapturingRunner)
    assert isinstance(
        eris.embed(PDB, representation="gearnet_edge", output_mode=output_mode),
        StructureArtifact,
    )

    with pytest.raises(ConfigError, match="output_mode"):
        eris.embed(PDB, representation="gearnet_edge", output_mode="invalid")


@pytest.mark.parametrize("representation", ["residue_knn_graph", "missing"])
def test_embed_rejects_non_learned_names(representation):
    with pytest.raises(RepresentationNotFoundError, match="Unknown embedder"):
        eris.embed(PDB, representation=representation)


def test_embed_persists_and_returns_same_artifact(monkeypatch, tmp_path):
    learned = __import__("eris.core.learned", fromlist=["StructuralEmbedderRunner"])
    artifact = eris.encode(PDB, representation="residue_contact_map", use_cache=False)
    saved = []

    def capture_save(self, output):
        saved.append((self, output))

    class CapturingRunner:
        def run(self, value, **kwargs):
            return artifact

    monkeypatch.setattr(learned, "StructuralEmbedderRunner", CapturingRunner)
    monkeypatch.setattr(StructureArtifact, "save", capture_save)
    output = tmp_path / "artifact"
    result = eris.embed(PDB, representation="gearnet_edge", output=output)
    assert result is artifact
    assert saved == [(artifact, output)]


def test_embed_unavailable_backend_error_is_preserved(monkeypatch):
    learned = __import__("eris.core.learned", fromlist=["StructuralEmbedderRunner"])

    class FailingRunner:
        def run(self, value, **kwargs):
            raise ExecutionError("Embedder 'gearnet_edge' backend is unavailable")

    monkeypatch.setattr(learned, "StructuralEmbedderRunner", FailingRunner)
    with pytest.raises(ExecutionError, match="backend is unavailable"):
        eris.embed(PDB, representation="gearnet_edge")


def test_root_import_and_discovery_remain_framework_lazy():
    script = """
import sys
import eris
assert eris.list_embedders() == ['gearnet_edge']
assert hasattr(eris, 'embed')
blocked = {'torch', 'torchdrug', 'torch_scatter', 'torch_cluster', 'rdkit'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
    )
