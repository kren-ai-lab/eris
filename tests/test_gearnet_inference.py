from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from eris.core.learned.backends.gearnet import backend as backend_module
from eris.core.learned.backends.gearnet.preprocessing import (
    BiologicalIdentityTable,
    BiologicalResidueIdentity,
    GearNetPreparedInput,
)
from eris.core.learned.config import StructuralEmbedderConfig
from eris.core.learned.execution import ResolvedExecution
from eris.core.learned.protocol import StructuralEmbedderBackend
from eris.core.learned.types import validate_embedding_result
from eris.core.types import StructureInput
from eris.exceptions import ExecutionError


class FakeDevice:
    def __init__(self, device_type: str = "cpu") -> None:
        self.type = device_type


class FakeTensor:
    def __init__(
        self,
        array: np.ndarray,
        *,
        dtype: str,
        device: str = "cpu",
        contiguous: bool = True,
    ) -> None:
        self.array = array
        self.dtype = dtype
        self.device = FakeDevice(device)
        self._contiguous = contiguous

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def is_floating_point(self) -> bool:
        return self.dtype in {"float16", "float32", "float64"}

    def is_contiguous(self) -> bool:
        return self._contiguous

    def contiguous(self) -> FakeTensor:
        return FakeTensor(
            np.ascontiguousarray(self.array),
            dtype=self.dtype,
            device=self.device.type,
        )

    def float(self) -> FakeTensor:
        return FakeTensor(self.array.astype(np.float32), dtype="float32")

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array

    def __getitem__(self, key) -> FakeTensor:
        return FakeTensor(
            self.array[key],
            dtype=self.dtype,
            device=self.device.type,
            contiguous=self._contiguous,
        )


class FakeFinite:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> FakeFinite:
        return self

    def item(self) -> bool:
        return self.value


class FakeTorch:
    Tensor = FakeTensor
    int64 = "int64"
    float32 = "float32"

    def __init__(self) -> None:
        self.inference_active = False

    def isfinite(self, value: FakeTensor) -> FakeFinite:
        return FakeFinite(bool(np.isfinite(value.array).all()))

    @contextmanager
    def inference_mode(self):
        assert not self.inference_active
        self.inference_active = True
        try:
            yield
        finally:
            self.inference_active = False


class FakeModel:
    def __init__(
        self,
        torch: FakeTorch,
        *,
        output: object | None = None,
        error: Exception | None = None,
        parameters: list[tuple[str, FakeTensor]] | None = None,
        buffers: list[tuple[str, FakeTensor]] | None = None,
    ) -> None:
        self.torch = torch
        self.training = False
        self.output = self._valid_output() if output is None else output
        self.error = error
        self.parameters = [] if parameters is None else parameters
        self.buffers = [] if buffers is None else buffers
        self.calls: list[tuple[object, FakeTensor, object, object]] = []

    @staticmethod
    def _valid_output() -> dict[str, FakeTensor]:
        return {
            "node_feature": FakeTensor(
                np.arange(2 * 3072, dtype=np.float32).reshape(2, 3072),
                dtype="float32",
            ),
            "graph_feature": FakeTensor(
                np.arange(3072, dtype=np.float32).reshape(1, 3072),
                dtype="float32",
            ),
        }

    def named_parameters(self):
        return self.parameters

    def named_buffers(self):
        return self.buffers

    def __call__(self, graph, node_feature, *, all_loss, metric):
        assert self.torch.inference_active
        self.calls.append((graph, node_feature, all_loss, metric))
        if self.error is not None:
            raise self.error
        return self.output


def _table() -> BiologicalIdentityTable:
    return BiologicalIdentityTable(
        records=tuple(
            BiologicalResidueIdentity(
                index=index,
                chain_id="A",
                residue_number=index + 1,
                insertion_code="",
                residue_name=name,
                node_id=f"A:{index + 1}::{name}",
            )
            for index, name in enumerate(("ALA", "GLY"))
        )
    )


def _prepared() -> GearNetPreparedInput:
    graph = SimpleNamespace(
        num_node=2,
        num_relation=7,
        node_feature=FakeTensor(np.ones((2, 21), dtype=np.int64), dtype="int64"),
        edge_feature=FakeTensor(np.ones((3, 59), dtype=np.float32), dtype="float32"),
    )
    return GearNetPreparedInput(
        graph=graph,
        carrier=np.asarray([0, 1], dtype=np.int64),
        biological_table=_table(),
        node_ids=np.asarray(["A:1::ALA", "A:2::GLY"], dtype=np.str_),
    )


def _config(output_mode: str = "both") -> StructuralEmbedderConfig:
    return StructuralEmbedderConfig(
        representation_name="test_gearnet_edge",
        output_mode=output_mode,
    )


def _execution() -> ResolvedExecution:
    return ResolvedExecution(device="cpu", precision="fp32")


@pytest.fixture
def runtime(monkeypatch):
    torch = FakeTorch()
    monkeypatch.setattr(backend_module, "_import_torch", lambda: torch)
    return torch


def test_backend_module_is_framework_lazy():
    script = """
import sys
import eris.core.learned.backends.gearnet.backend

blocked = {'torch', 'torchdrug'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional GearNet frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_backend_capabilities_are_cpu_fp32_only():
    backend = backend_module.GearNetEdgeBackend()

    capabilities = backend.execution_capabilities()

    assert backend.backend_id == "gearnet-edge"
    assert capabilities.devices == ("cpu",)
    assert capabilities.precisions == ("fp32",)
    assert capabilities.preferred_device == "cpu"
    assert isinstance(backend, StructuralEmbedderBackend)


def test_preprocess_delegates_to_m5_3_foundation(monkeypatch):
    structure = StructureInput(source="fixture.pdb", chain_id="A", model_index=0)
    config = _config()
    selected = SimpleNamespace(input_format="pdb", biological_table=object())
    records = object()
    prepared = object()
    calls = []
    monkeypatch.setattr(
        backend_module,
        "load_selected_biological_structure",
        lambda value: calls.append(("load", value)) or selected,
    )
    monkeypatch.setattr(
        backend_module,
        "select_gearnet_transport_records",
        lambda value: calls.append(("select", value)) or records,
    )
    monkeypatch.setattr(
        backend_module,
        "construct_gearnet_input",
        lambda value, table: calls.append(("construct", value, table)) or prepared,
    )

    result = backend_module.GearNetEdgeBackend().preprocess(structure, config=config)

    assert result.prepared is prepared
    assert result.input_format == "pdb"
    assert calls == [
        ("load", structure),
        ("select", selected),
        ("construct", records, selected.biological_table),
    ]


def test_load_delegates_to_model_loading_foundation(monkeypatch):
    checkpoint = object()
    execution = _execution()
    model = object()
    calls = []
    monkeypatch.setattr(
        backend_module,
        "load_verified_gearnet_edge_model",
        lambda value, *, execution: calls.append((value, execution)) or model,
    )

    result = backend_module.GearNetEdgeBackend().load(checkpoint, execution=execution)

    assert result is model
    assert calls == [(checkpoint, execution)]


@pytest.mark.parametrize(
    "execution",
    [
        ResolvedExecution(device="cuda:0", precision="fp32"),
        ResolvedExecution(device="cpu", precision="fp16"),
        ResolvedExecution(device="cpu", precision="bf16"),
    ],
)
def test_infer_rejects_unsupported_execution_before_framework_import(
    monkeypatch,
    execution,
):
    monkeypatch.setattr(
        backend_module,
        "_import_torch",
        lambda: (_ for _ in ()).throw(AssertionError("framework import must not occur")),
    )

    with pytest.raises(ExecutionError, match="only device='cpu' and precision='fp32'"):
        backend_module.GearNetEdgeBackend().infer(
            object(),
            _prepared(),
            config=_config(),
            execution=execution,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("node_width", r"node_feature must have shape \(2, 21\)"),
        ("edge_width", "edge_feature width must be 59"),
        ("relation", "must declare 7 relations"),
        ("carrier_length", "carrier must align"),
        ("carrier_duplicate", "carrier indexes must be unique"),
        ("node_ids_length", "node_ids must align"),
        ("node_ids_duplicate", "node_ids must be unique"),
        ("node_ids_mapping", "authoritative carrier"),
    ],
)
def test_prepared_invariants_fail_before_framework_import(monkeypatch, mutation, match):
    prepared = _prepared()
    if mutation == "node_width":
        prepared.graph.node_feature = FakeTensor(np.ones((2, 20), dtype=np.int64), dtype="int64")
    elif mutation == "edge_width":
        prepared.graph.edge_feature = FakeTensor(
            np.ones((3, 58), dtype=np.float32), dtype="float32"
        )
    elif mutation == "relation":
        prepared.graph.num_relation = 6
    elif mutation == "carrier_length":
        prepared = replace(prepared, carrier=np.asarray([0], dtype=np.int64))
    elif mutation == "carrier_duplicate":
        prepared = replace(prepared, carrier=np.asarray([0, 0], dtype=np.int64))
    elif mutation == "node_ids_length":
        prepared = replace(prepared, node_ids=np.asarray(["A:1::ALA"], dtype=np.str_))
    elif mutation == "node_ids_duplicate":
        prepared = replace(
            prepared,
            node_ids=np.asarray(["A:1::ALA", "A:1::ALA"], dtype=np.str_),
        )
    else:
        prepared = replace(
            prepared,
            node_ids=np.asarray(["A:2::GLY", "A:1::ALA"], dtype=np.str_),
        )
    monkeypatch.setattr(
        backend_module,
        "_import_torch",
        lambda: (_ for _ in ()).throw(AssertionError("framework import must not occur")),
    )

    with pytest.raises(ExecutionError, match=match):
        backend_module.GearNetEdgeBackend().infer(
            SimpleNamespace(training=False),
            prepared,
            config=_config(),
            execution=_execution(),
        )


def test_training_model_is_rejected_before_framework_import(monkeypatch):
    monkeypatch.setattr(
        backend_module,
        "_import_torch",
        lambda: (_ for _ in ()).throw(AssertionError("framework import must not occur")),
    )

    with pytest.raises(ExecutionError, match="eval mode"):
        backend_module.GearNetEdgeBackend().infer(
            SimpleNamespace(training=True),
            _prepared(),
            config=_config(),
            execution=_execution(),
        )


@pytest.mark.parametrize(
    ("tensor", "match"),
    [
        (
            FakeTensor(np.zeros(1, dtype=np.float32), dtype="float32", device="cuda"),
            "parameter 'weight' is not on CPU",
        ),
        (
            FakeTensor(np.zeros(1, dtype=np.float16), dtype="float16"),
            "parameter 'weight' is not torch.float32",
        ),
    ],
)
def test_infer_revalidates_model_cpu_fp32(runtime, tensor, match):
    model = FakeModel(runtime, parameters=[("weight", tensor)])

    with pytest.raises(ExecutionError, match=match):
        backend_module.GearNetEdgeBackend().infer(
            model,
            _prepared(),
            config=_config(),
            execution=_execution(),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "node_feature",
            FakeTensor(np.ones((2, 21), dtype=np.float32), dtype="float32"),
            "node_feature must be torch.int64",
        ),
        (
            "edge_feature",
            FakeTensor(np.ones((3, 59), dtype=np.float64), dtype="float64"),
            "edge_feature must be torch.float32",
        ),
        (
            "node_feature",
            FakeTensor(
                np.ones((2, 21), dtype=np.int64),
                dtype="int64",
                device="cuda",
            ),
            "graph tensors must be on CPU",
        ),
        (
            "edge_feature",
            FakeTensor(
                np.full((3, 59), np.nan, dtype=np.float32),
                dtype="float32",
            ),
            "edge_feature must contain only finite values",
        ),
    ],
)
def test_infer_rejects_invalid_framework_input_tensors(runtime, field, value, match):
    prepared = _prepared()
    setattr(prepared.graph, field, value)

    with pytest.raises(ExecutionError, match=match):
        backend_module.GearNetEdgeBackend().infer(
            FakeModel(runtime),
            prepared,
            config=_config(),
            execution=_execution(),
        )


@pytest.mark.parametrize("output_mode", ["node", "global", "both"])
def test_exact_forward_and_output_modes(runtime, output_mode):
    model = FakeModel(runtime)
    prepared = _prepared()

    result = backend_module.GearNetEdgeBackend().infer(
        model,
        prepared,
        config=_config(output_mode),
        execution=_execution(),
    )

    assert len(model.calls) == 1
    graph, node_input, all_loss, metric = model.calls[0]
    assert graph is prepared.graph
    assert node_input.dtype == "float32"
    assert node_input.shape == (2, 21)
    assert all_loss is None
    assert metric is None
    assert runtime.inference_active is False
    assert result.valid_mask is None
    if output_mode == "node":
        assert result.node_embedding is not None
        assert result.global_embedding is None
        np.testing.assert_array_equal(result.node_ids, prepared.node_ids)
    elif output_mode == "global":
        assert result.node_embedding is None
        assert result.node_ids is None
        assert result.global_embedding is not None
    else:
        assert result.node_embedding is not None
        assert result.global_embedding is not None
        np.testing.assert_array_equal(result.node_ids, prepared.node_ids)
    validate_embedding_result(result, output_mode=output_mode)


@pytest.mark.parametrize(
    ("output", "match"),
    [
        ([], "must be a mapping"),
        ({"node_feature": object()}, "missing required keys: graph_feature"),
        (
            {
                "node_feature": FakeTensor(np.zeros((2, 3071), dtype=np.float32), dtype="float32"),
                "graph_feature": FakeTensor(np.zeros((1, 3072), dtype=np.float32), dtype="float32"),
            },
            r"node_feature.*shape \(2, 3072\)",
        ),
        (
            {
                "node_feature": FakeTensor(np.zeros((2, 3072), dtype=np.float64), dtype="float64"),
                "graph_feature": FakeTensor(np.zeros((1, 3072), dtype=np.float32), dtype="float32"),
            },
            "node_feature.*torch.float32",
        ),
        (
            {
                "node_feature": FakeTensor(
                    np.zeros((2, 3072), dtype=np.float32), dtype="float32", device="cuda"
                ),
                "graph_feature": FakeTensor(np.zeros((1, 3072), dtype=np.float32), dtype="float32"),
            },
            "node_feature.*CPU",
        ),
        (
            {
                "node_feature": FakeTensor(
                    np.full((2, 3072), np.nan, dtype=np.float32), dtype="float32"
                ),
                "graph_feature": FakeTensor(np.zeros((1, 3072), dtype=np.float32), dtype="float32"),
            },
            "node_feature.*finite",
        ),
        (
            {
                "node_feature": FakeTensor(np.zeros((2, 3072), dtype=np.float32), dtype="float32"),
                "graph_feature": FakeTensor(np.zeros((2, 3072), dtype=np.float32), dtype="float32"),
            },
            r"graph_feature.*shape \(1, 3072\)",
        ),
        (
            {
                "node_feature": FakeTensor(np.zeros((2, 3072), dtype=np.float32), dtype="float32"),
                "graph_feature": FakeTensor(np.zeros((1, 3072), dtype=np.float64), dtype="float64"),
            },
            "graph_feature.*torch.float32",
        ),
        (
            {
                "node_feature": FakeTensor(np.zeros((2, 3072), dtype=np.float32), dtype="float32"),
                "graph_feature": FakeTensor(
                    np.zeros((1, 3072), dtype=np.float32), dtype="float32", device="cuda"
                ),
            },
            "graph_feature.*CPU",
        ),
        (
            {
                "node_feature": FakeTensor(np.zeros((2, 3072), dtype=np.float32), dtype="float32"),
                "graph_feature": FakeTensor(
                    np.full((1, 3072), np.inf, dtype=np.float32), dtype="float32"
                ),
            },
            "graph_feature.*finite",
        ),
    ],
)
def test_forward_output_contract_is_strict(runtime, output, match):
    model = FakeModel(runtime, output=output)

    with pytest.raises(ExecutionError, match=match):
        backend_module.GearNetEdgeBackend().infer(
            model,
            _prepared(),
            config=_config(),
            execution=_execution(),
        )


@pytest.mark.parametrize("error", [RuntimeError("forward failed"), OSError("JIT failed")])
def test_expected_forward_errors_preserve_cause(runtime, error):
    model = FakeModel(runtime, error=error)

    with pytest.raises(ExecutionError, match="GearNet-Edge CPU/fp32") as captured:
        backend_module.GearNetEdgeBackend().infer(
            model,
            _prepared(),
            config=_config(),
            execution=_execution(),
        )

    assert captured.value.__cause__ is error
    if isinstance(error, OSError) and sys.platform == "win32":
        assert "MSVC x64/cl.exe and Ninja" in str(captured.value)
