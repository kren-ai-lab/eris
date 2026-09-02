from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import eris.core.learned.backends.gearnet.backend as backend_module
from eris.core.learned.backends.gearnet.config import GearNetEdgeConfig
from eris.core.learned.backends.gearnet.native_graph import NativeGearNetGraph
from eris.core.learned.backends.gearnet.preprocessing import GearNetPreparedInput
from eris.core.learned.config import StructuralEmbedderConfig
from eris.core.learned.execution import ResolvedExecution
from eris.core.types import StructureInput
from eris.exceptions import ExecutionError


def _config() -> StructuralEmbedderConfig:
    return GearNetEdgeConfig(representation_name="gearnet_edge", output_mode="both")


def test_backend_import_is_framework_lazy():
    assert backend_module.GearNetEdgeBackend().backend_id == "gearnet-edge"


def test_preprocess_uses_native_graph_construction(monkeypatch):
    structure = StructureInput(source="fixture.pdb", chain_id="A", model_index=0)
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
        "construct_native_gearnet_input",
        lambda value, table: calls.append(("construct", value, table)) or prepared,
    )

    result = backend_module.GearNetEdgeBackend().preprocess(structure, config=_config())

    assert result.prepared is prepared
    assert result.input_format == "pdb"
    assert calls == [
        ("load", structure),
        ("select", selected),
        ("construct", records, selected.biological_table),
    ]


def test_infer_rejects_non_native_prepared_graph_before_torch_import(monkeypatch):
    prepared = GearNetPreparedInput(
        graph=object(),
        carrier=np.asarray([0], dtype=np.int64),
        biological_table=SimpleNamespace(node_ids_for=lambda carrier: ("A:1::ALA",)),
        node_ids=np.asarray(["A:1::ALA"], dtype=np.str_),
    )
    monkeypatch.setattr(backend_module, "_import_torch", lambda: pytest.fail("unexpected torch"))

    with pytest.raises(ExecutionError, match="NativeGearNetGraph"):
        backend_module.GearNetEdgeBackend().infer(
            SimpleNamespace(training=False),
            prepared,
            config=_config(),
            execution=ResolvedExecution(device="cpu", precision="fp32"),
        )


def test_infer_rejects_invalid_native_edge_feature_before_forward(monkeypatch):
    graph = NativeGearNetGraph(
        node_feature=np.zeros((1, 21), dtype=np.int64),
        node_position=np.zeros((1, 3), dtype=np.float32),
        edge_source=np.zeros(1, dtype=np.int64),
        edge_destination=np.zeros(1, dtype=np.int64),
        edge_relation=np.zeros(1, dtype=np.int64),
        edge_feature=np.zeros((1, 59), dtype=np.float64),
        line_edge_source=np.zeros(1, dtype=np.int64),
        line_edge_destination=np.zeros(1, dtype=np.int64),
        line_angle_relation=np.zeros(1, dtype=np.int64),
    )
    prepared = GearNetPreparedInput(
        graph=graph,
        carrier=np.asarray([0], dtype=np.int64),
        biological_table=SimpleNamespace(node_ids_for=lambda carrier: ("A:1::ALA",)),
        node_ids=np.asarray(["A:1::ALA"], dtype=np.str_),
    )
    monkeypatch.setattr(backend_module, "_import_torch", lambda: pytest.fail("unexpected torch"))

    with pytest.raises(ExecutionError, match="edge_feature must be float32"):
        backend_module.GearNetEdgeBackend().infer(
            SimpleNamespace(training=False),
            prepared,
            config=_config(),
            execution=ResolvedExecution(device="cpu", precision="fp32"),
        )
