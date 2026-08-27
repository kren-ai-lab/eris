from __future__ import annotations

import pickle
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

from eris.core.learned.backends.gearnet import model as model_module
from eris.core.learned.checkpoints import CheckpointDescriptor, VerifiedCheckpoint
from eris.core.learned.execution import ResolvedExecution
from eris.exceptions import ExecutionError


class FakeDevice:
    def __init__(self, device_type: str = "cpu") -> None:
        self.type = device_type


class FakeTensor:
    def __init__(
        self,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        floating: bool = True,
    ) -> None:
        self.device = FakeDevice(device)
        self.dtype = dtype
        self._floating = floating

    def is_floating_point(self) -> bool:
        return self._floating


class FakeModel:
    __module__ = "torchdrug.models.gearnet"

    constructor_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).constructor_kwargs = kwargs
        self.output_dim = 3072
        self.training = True
        self.child = SimpleNamespace(training=True)
        self.parameter = FakeTensor()
        self.buffer = FakeTensor(floating=False, dtype="int64")
        self.strict: bool | None = None
        self.loaded_state: object = None
        self.load_error: RuntimeError | None = None

    def load_state_dict(self, state_dict, *, strict: bool):
        if self.load_error is not None:
            raise self.load_error
        self.strict = strict
        self.loaded_state = state_dict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def named_parameters(self):
        return [("weight", self.parameter)]

    def named_buffers(self):
        return [("counter", self.buffer)]

    def modules(self):
        return [self, self.child]

    def eval(self):
        self.training = False
        self.child.training = False
        return self


class FakeTorch:
    Tensor = FakeTensor
    float32 = "float32"

    def __init__(self, state_dict=None, error: Exception | None = None) -> None:
        self.state_dict = OrderedDict(weight=FakeTensor()) if state_dict is None else state_dict
        self.error = error
        self.load_calls: list[tuple[Path, dict[str, object]]] = []

    def load(self, path: Path, **kwargs: object):
        self.load_calls.append((path, kwargs))
        if self.error is not None:
            raise self.error
        return self.state_dict


def _checkpoint(tmp_path: Path, **overrides: object) -> VerifiedCheckpoint:
    descriptor_values = {
        "identifier": model_module.GEARNET_EDGE_CHECKPOINT_IDENTIFIER,
        "sha256": model_module.GEARNET_EDGE_CHECKPOINT_SHA256,
        "revision": "Zenodo record v2",
        "source": "https://zenodo.org/records/7723075",
    }
    descriptor_values.update(overrides.pop("descriptor", {}))
    values = {
        "path": tmp_path / "checkpoint",
        "descriptor": CheckpointDescriptor(**descriptor_values),
        "sha256": model_module.GEARNET_EDGE_CHECKPOINT_SHA256,
    }
    values.update(overrides)
    return VerifiedCheckpoint(**values)


def _frameworks(torch: FakeTorch | None = None):
    return torch or FakeTorch(), SimpleNamespace(GearNet=FakeModel)


def test_gearnet_model_modules_are_framework_lazy():
    script = """
import sys
import eris
import eris.core.learned
import eris.core.learned.backends.gearnet
import eris.core.learned.backends.gearnet.model

blocked = {'torch', 'torchdrug'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional GearNet frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_exact_constructor_is_frozen(monkeypatch):
    monkeypatch.setattr(model_module, "_import_frameworks", _frameworks)

    model = model_module.construct_gearnet_edge_model()

    assert type(model) is FakeModel
    assert model.output_dim == 3072
    assert FakeModel.constructor_kwargs == {
        "input_dim": 21,
        "hidden_dims": [512, 512, 512, 512, 512, 512],
        "num_relation": 7,
        "edge_input_dim": 59,
        "num_angle_bin": 8,
        "short_cut": True,
        "batch_norm": True,
        "concat_hidden": True,
        "readout": "sum",
    }


@pytest.mark.parametrize(
    "execution",
    [
        ResolvedExecution(device="cuda:0", precision="fp32"),
        ResolvedExecution(device="cpu", precision="fp16"),
        ResolvedExecution(device="cpu", precision="bf16"),
    ],
)
def test_unsupported_execution_fails_before_framework_import(
    tmp_path,
    monkeypatch,
    execution,
):
    def unexpected_import():
        raise AssertionError("framework import must not occur")

    monkeypatch.setattr(model_module, "_import_frameworks", unexpected_import)

    with pytest.raises(ExecutionError, match="only device='cpu' and precision='fp32'"):
        model_module.load_verified_gearnet_edge_model(
            _checkpoint(tmp_path),
            execution=execution,
        )


def test_loader_requires_verified_official_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "_import_frameworks", _frameworks)
    execution = ResolvedExecution(device="cpu", precision="fp32")

    with pytest.raises(TypeError, match="VerifiedCheckpoint"):
        model_module.load_verified_gearnet_edge_model(
            tmp_path / "checkpoint",
            execution=execution,
        )
    with pytest.raises(ExecutionError, match="inconsistent SHA-256"):
        model_module.load_verified_gearnet_edge_model(
            _checkpoint(tmp_path, sha256="a" * 64),
            execution=execution,
        )
    with pytest.raises(ExecutionError, match="frozen mc_gearnet_edge.pth"):
        model_module.load_verified_gearnet_edge_model(
            _checkpoint(tmp_path, descriptor={"identifier": "other.pth"}),
            execution=execution,
        )


def test_safe_strict_load_and_eval(tmp_path, monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setattr(model_module, "_import_frameworks", lambda: _frameworks(fake_torch))
    checkpoint = _checkpoint(tmp_path)

    model = model_module.load_verified_gearnet_edge_model(
        checkpoint,
        execution=ResolvedExecution(device="cpu", precision="fp32"),
    )

    assert fake_torch.load_calls == [
        (
            checkpoint.path,
            {"map_location": "cpu", "weights_only": True},
        )
    ]
    assert model.strict is True
    assert model.loaded_state is fake_torch.state_dict
    assert model.training is False
    assert all(module.training is False for module in model.modules())
    assert model.parameter._floating is True


@pytest.mark.parametrize(
    ("state_dict", "message"),
    [
        ([], "root must be a state_dict mapping"),
        (OrderedDict([(1, FakeTensor())]), "keys must all be strings"),
        (OrderedDict(weight=object()), "values must all be tensors"),
    ],
)
def test_state_dict_structure_is_validated(tmp_path, monkeypatch, state_dict, message):
    fake_torch = FakeTorch(state_dict=state_dict)
    monkeypatch.setattr(model_module, "_import_frameworks", lambda: _frameworks(fake_torch))

    with pytest.raises(ExecutionError, match=message):
        model_module.load_verified_gearnet_edge_model(
            _checkpoint(tmp_path),
            execution=ResolvedExecution(device="cpu", precision="fp32"),
        )


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("checkpoint disappeared"),
        PermissionError("checkpoint unreadable"),
        EOFError("truncated archive"),
        pickle.UnpicklingError("weights-only rejection"),
        RuntimeError("serialization failure"),
    ],
)
def test_expected_deserialization_failures_are_actionable(
    tmp_path,
    monkeypatch,
    error,
):
    fake_torch = FakeTorch(error=error)
    monkeypatch.setattr(model_module, "_import_frameworks", lambda: _frameworks(fake_torch))

    with pytest.raises(ExecutionError, match="Unable to safely load") as captured:
        model_module.load_verified_gearnet_edge_model(
            _checkpoint(tmp_path),
            execution=ResolvedExecution(device="cpu", precision="fp32"),
        )

    assert captured.value.__cause__ is error


def test_strict_loading_failure_preserves_cause(tmp_path, monkeypatch):
    fake_torch, fake_models = _frameworks()

    class FailingModel(FakeModel):
        __module__ = "torchdrug.models.gearnet"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.load_error = RuntimeError("missing key")

    fake_models.GearNet = FailingModel
    monkeypatch.setattr(model_module, "_import_frameworks", lambda: (fake_torch, fake_models))

    with pytest.raises(ExecutionError, match="incompatible") as captured:
        model_module.load_verified_gearnet_edge_model(
            _checkpoint(tmp_path),
            execution=ResolvedExecution(device="cpu", precision="fp32"),
        )

    assert isinstance(captured.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("parameter", FakeTensor(device="cuda"), "not on CPU"),
        ("parameter", FakeTensor(dtype="float16"), "not torch.float32"),
        ("buffer", FakeTensor(dtype="float16"), "not torch.float32"),
    ],
)
def test_cpu_fp32_contract_is_enforced(
    tmp_path,
    monkeypatch,
    attribute,
    value,
    message,
):
    fake_torch, fake_models = _frameworks()

    class InvalidModel(FakeModel):
        __module__ = "torchdrug.models.gearnet"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            setattr(self, attribute, value)

    fake_models.GearNet = InvalidModel
    monkeypatch.setattr(model_module, "_import_frameworks", lambda: (fake_torch, fake_models))

    with pytest.raises(ExecutionError, match=message):
        model_module.load_verified_gearnet_edge_model(
            _checkpoint(tmp_path),
            execution=ResolvedExecution(device="cpu", precision="fp32"),
        )
