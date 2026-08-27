from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdrug")

from torchdrug import models  # noqa: E402

from eris.core.learned.backends.gearnet.model import (  # noqa: E402  # noqa: E402
    GEARNET_EDGE_CHECKPOINT_IDENTIFIER,
    GEARNET_EDGE_CHECKPOINT_SHA256,
    _load_state_dict_strict,
    construct_gearnet_edge_model,
    load_verified_gearnet_edge_model,
)
from eris.core.learned.checkpoints import CheckpointDescriptor, verify_checkpoint  # noqa: E402
from eris.core.learned.execution import ResolvedExecution  # noqa: E402
from eris.exceptions import ExecutionError  # noqa: E402


def test_exact_real_model_construction():
    model = construct_gearnet_edge_model()

    assert type(model) is models.GearNet
    assert type(model).__name__ == "GeometryAwareRelationalGraphNeuralNetwork"
    assert model.output_dim == 3072
    assert len(model.state_dict()) == 138
    assert model.state_dict()["layers.0.self_loop.weight"].shape == (512, 21)
    assert model.state_dict()["layers.5.linear.weight"].shape == (512, 3584)
    assert model.state_dict()["edge_layers.0.self_loop.weight"].shape == (21, 59)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape"])
def test_strict_real_model_rejects_incompatible_state(mutation):
    model = construct_gearnet_edge_model()
    state = OrderedDict(model.state_dict())
    first_key = next(iter(state))
    if mutation == "missing":
        state.pop(first_key)
    elif mutation == "unexpected":
        state["eris_unexpected"] = torch.zeros(1)
    else:
        state[first_key] = state[first_key][:-1]

    with pytest.raises(ExecutionError, match="incompatible"):
        _load_state_dict_strict(model, state)


def test_official_extensionless_checkpoint_loads_strictly(tmp_path):
    configured = os.environ.get("ERIS_GEARNET_CHECKPOINT")
    if not configured:
        pytest.skip("ERIS_GEARNET_CHECKPOINT is not configured")
    official_path = Path(configured)
    if not official_path.is_file():
        pytest.fail(f"ERIS_GEARNET_CHECKPOINT is not a file: {official_path}")

    extensionless = tmp_path / "checkpoint"
    extensionless.write_bytes(official_path.read_bytes())
    descriptor = CheckpointDescriptor(
        identifier=GEARNET_EDGE_CHECKPOINT_IDENTIFIER,
        sha256=GEARNET_EDGE_CHECKPOINT_SHA256,
        revision="Zenodo record v2",
        source="https://zenodo.org/records/7723075",
    )
    checkpoint = verify_checkpoint(extensionless, descriptor)

    model = load_verified_gearnet_edge_model(
        checkpoint,
        execution=ResolvedExecution(device="cpu", precision="fp32"),
    )

    assert model.training is False
    assert all(not module.training for module in model.modules())
    batchnorms = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert len(batchnorms) == 18
    assert all(not module.training for module in batchnorms)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(
        not parameter.is_floating_point() or parameter.dtype == torch.float32
        for parameter in model.parameters()
    )
    assert all(buffer.device.type == "cpu" for buffer in model.buffers())
    assert all(
        not buffer.is_floating_point() or buffer.dtype == torch.float32
        for buffer in model.buffers()
    )
