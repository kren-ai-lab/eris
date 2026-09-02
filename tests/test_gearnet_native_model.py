"""Numerical parity gates for the native GearNet-Edge model."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eris.core.learned.backends.gearnet.native_graph import construct_native_gearnet_graph  # noqa: E402
from eris.core.learned.backends.gearnet.native_model import (  # noqa: E402
    construct_native_gearnet_edge_model,
    load_native_gearnet_edge_model,
)
from eris.core.learned.backends.gearnet.preprocessing import (  # noqa: E402
    load_selected_biological_structure,
    select_gearnet_transport_records,
)
from eris.core.learned.checkpoints import CheckpointDescriptor, verify_checkpoint  # noqa: E402
from eris.core.types import StructureInput  # noqa: E402


DATA_DIR = Path(__file__).with_name("data")
CHECKPOINT_ENV = "ERIS_GEARNET_CHECKPOINT"
CHECKPOINT_DESCRIPTOR = CheckpointDescriptor(
    identifier="mc_gearnet_edge.pth",
    sha256="6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec",
    revision="Zenodo record v2",
    source="https://zenodo.org/records/7723075",
)


def _checkpoint():
    configured = os.environ.get(CHECKPOINT_ENV)
    if not configured:
        pytest.skip(f"{CHECKPOINT_ENV} is not configured")
    path = Path(configured)
    if not path.is_file():
        pytest.fail(f"{CHECKPOINT_ENV} is not a file: {path}")
    return verify_checkpoint(path, CHECKPOINT_DESCRIPTOR)


def _graph():
    selected = load_selected_biological_structure(
        StructureInput(source=DATA_DIR / "gearnet_reference.pdb", chain_id="A", model_index=0)
    )
    return construct_native_gearnet_graph(select_gearnet_transport_records(selected))


def _golden() -> dict[str, np.ndarray]:
    with np.load(DATA_DIR / "gearnet_reference_embeddings.npz", allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def test_native_model_matches_frozen_checkpoint_state_dict_layout_exactly():
    model = construct_native_gearnet_edge_model()

    assert len(model.state_dict()) == 138
    assert model.state_dict()["layers.0.self_loop.weight"].shape == (512, 21)
    assert model.state_dict()["layers.5.linear.weight"].shape == (512, 3584)
    assert model.state_dict()["edge_layers.0.self_loop.weight"].shape == (21, 59)


def test_native_model_strict_load_and_reference_embedding_parity():
    model = load_native_gearnet_edge_model(_checkpoint())
    graph = _graph()
    golden = _golden()

    with torch.inference_mode():
        output = model(graph)

    np.testing.assert_allclose(
        output["node_feature"].numpy(), golden["node_embedding"], rtol=1e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        output["graph_feature"][0].numpy(),
        golden["global_embedding"],
        rtol=1e-6,
        atol=2e-6,
    )
