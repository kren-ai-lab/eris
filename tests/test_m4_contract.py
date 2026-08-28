from __future__ import annotations

import subprocess
import sys

import numpy as np
from test_learned_runner import _config, _model, _runner_environment

from eris.core.artifacts import StructureArtifact
from eris.core.learned import StructuralExecutionConfig, learned_registry
from eris.core.types import StructureInput


def test_m4_fake_backend_contract_is_complete_lazy_and_precision_safe(tmp_path):
    runner, state, artifact_cache, _, source, checkpoint = _runner_environment(tmp_path)
    structure_input = StructureInput(source=str(source))
    model = _model(checkpoint)

    first = runner.run(
        structure_input,
        config=_config("both"),
        model=model,
        checkpoint_path=checkpoint,
        execution=StructuralExecutionConfig(device="cpu", precision="fp32"),
    )

    first.validate(strict=True)
    assert learned_registry.names() == ["gearnet_edge"]
    assert list(first.data) == [
        "node_embedding",
        "global_embedding",
        "node_ids",
        "valid_mask",
    ]
    assert first.data["node_ids"].tolist() == ["A:1::ALA", "A:2::GLY", "A:3::SER"]
    assert first.data["node_embedding"].shape[0] == first.data["node_ids"].shape[0]
    assert first.meta["representation_family"] == "learned"
    assert first.meta["model"]["checkpoint"]["sha256"] == model.checkpoint.sha256
    assert first.meta["provenance"]["environment"]["execution"]["resolved"] == {
        "device": "cpu",
        "precision": "fp32",
    }
    assert state.loaded_checkpoint_path is not None
    assert state.loaded_checkpoint_path.is_file()

    outdir = tmp_path / "round-trip"
    first.save(outdir)
    loaded = StructureArtifact.load(outdir)
    assert loaded.meta == first.meta
    for name, expected in first.data.items():
        np.testing.assert_array_equal(loaded.data[name], expected)

    second = runner.run(
        structure_input,
        config=_config("both"),
        model=model,
        execution=StructuralExecutionConfig(device="cuda:0", precision="fp32"),
    )
    assert second.meta == first.meta
    assert state.probe_calls == state.factory_calls == 1
    assert state.preprocess_calls == state.load_calls == state.infer_calls == 1

    third = runner.run(
        structure_input,
        config=_config("both"),
        model=model,
        execution=StructuralExecutionConfig(device="cpu", precision="fp16"),
    )
    assert state.infer_calls == 2
    assert artifact_cache.keys[0] == artifact_cache.keys[1]
    assert artifact_cache.keys[2] != artifact_cache.keys[0]
    assert third.meta["provenance"]["environment"]["execution"]["resolved"]["precision"] == "fp16"


def test_m4_contract_imports_no_optional_framework_and_exposes_no_public_runner():
    script = """
import sys
import eris
import eris.core.learned
import eris.core.learned.runner

blocked = {'torch', 'torchdrug', 'torch_geometric', 'transformers'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
assert not hasattr(eris, 'embed')
assert not hasattr(eris, 'StructuralEmbedderRunner')
assert eris.core.learned.learned_registry.names() == ['gearnet_edge']
"""

    subprocess.run([sys.executable, "-c", script], check=True)
