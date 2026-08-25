from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import (
    build_input_metadata,
    compute_config_fingerprint,
)
from eris.core.artifacts import StructureArtifact
from eris.core.learned import (
    CheckpointDescriptor,
    StructuralEmbedderConfig,
    StructuralEmbeddingResult,
    StructuralModelDescriptor,
    build_learned_artifact,
)
from eris.exceptions import ExecutionError

NODE_IDS = np.array(["B:9::GLY", "opaque-node-β", "A:2::ALA"], dtype=np.str_)
VALID_MASK = np.array([True, False, True], dtype=np.bool_)
ENCODER_METADATA = {
    "implementation": "tests.fake_backend.FakeStructuralEncoder",
    "version": "test-adapter-v1",
}


def _input_metadata():
    return build_input_metadata(
        source="memory://synthetic-structure",
        source_sha256="a" * 64,
        input_format="pdb",
        model_index=0,
        chain_id=None,
    )


def _model() -> StructuralModelDescriptor:
    return StructuralModelDescriptor(
        architecture="synthetic-structural-model",
        checkpoint=CheckpointDescriptor(
            identifier="synthetic-checkpoint",
            revision="revision-1",
            source="https://models.example/synthetic.bin",
            sha256="b" * 64,
        ),
    )


def _result(
    output_mode: str,
    *,
    with_mask: bool = False,
    dtype=np.float32,
) -> StructuralEmbeddingResult:
    node_embedding = np.arange(12, dtype=dtype).reshape(3, 4)
    global_embedding = np.arange(2, dtype=dtype)
    if output_mode == "node":
        return StructuralEmbeddingResult(
            node_embedding=node_embedding,
            node_ids=NODE_IDS,
            valid_mask=VALID_MASK if with_mask else None,
        )
    if output_mode == "global":
        return StructuralEmbeddingResult(global_embedding=global_embedding)
    return StructuralEmbeddingResult(
        node_embedding=node_embedding,
        global_embedding=global_embedding,
        node_ids=NODE_IDS,
        valid_mask=VALID_MASK if with_mask else None,
    )


def _artifact(
    output_mode: str,
    *,
    with_mask: bool = False,
    dtype=np.float32,
) -> StructureArtifact:
    return build_learned_artifact(
        config=StructuralEmbedderConfig(
            representation_name="synthetic_structure_embedding",
            output_mode=output_mode,
        ),
        result=_result(output_mode, with_mask=with_mask, dtype=dtype),
        model=_model(),
        input_metadata=_input_metadata(),
        encoder_metadata=ENCODER_METADATA,
    )


@pytest.mark.parametrize(
    ("output_mode", "with_mask", "expected_order", "granularity", "counts"),
    [
        ("node", False, ["node_embedding", "node_ids"], "residue", {"n_nodes": 3}),
        (
            "node",
            True,
            ["node_embedding", "node_ids", "valid_mask"],
            "residue",
            {"n_nodes": 3},
        ),
        ("global", False, ["global_embedding"], "structure", {}),
        (
            "both",
            False,
            ["node_embedding", "global_embedding", "node_ids"],
            "mixed",
            {"n_nodes": 3},
        ),
        (
            "both",
            True,
            ["node_embedding", "global_embedding", "node_ids", "valid_mask"],
            "mixed",
            {"n_nodes": 3},
        ),
    ],
)
def test_modes_build_strict_valid_artifacts_with_canonical_identity_and_order(
    output_mode,
    with_mask,
    expected_order,
    granularity,
    counts,
):
    artifact = _artifact(output_mode, with_mask=with_mask)

    artifact.validate(strict=True)
    assert artifact.meta["representation_family"] == "learned"
    assert artifact.meta["representation_type"] == "embedding"
    assert artifact.meta["granularity"] == granularity
    assert artifact.meta["coordinate_source"] == "model_specific"
    assert list(artifact.data) == expected_order
    assert [output["name"] for output in artifact.meta["outputs"]] == expected_order
    assert artifact.meta["counts"] == counts
    assert artifact.meta["feature_keys"] == {}
    assert artifact.meta["shapes"] == {
        name: list(array.shape) for name, array in artifact.data.items()
    }
    assert artifact.meta["dtypes"] == {
        name: str(array.dtype) for name, array in artifact.data.items()
    }


def test_output_declarations_use_canonical_roles_granularities_and_layouts():
    artifact = _artifact("both", with_mask=True)

    assert artifact.meta["outputs"] == [
        {
            "name": "node_embedding",
            "role": "node_embedding",
            "granularity": "residue",
            "layout": "ND",
        },
        {
            "name": "global_embedding",
            "role": "global_embedding",
            "granularity": "structure",
            "layout": "D",
        },
        {
            "name": "node_ids",
            "role": "node_ids",
            "granularity": "residue",
            "layout": "N",
        },
        {
            "name": "valid_mask",
            "role": "valid_mask",
            "granularity": "residue",
            "layout": "N",
        },
    ]


def test_builder_preserves_row_order_mask_alignment_array_identity_and_dtype():
    result = _result("both", with_mask=True, dtype=np.float64)
    artifact = build_learned_artifact(
        config=StructuralEmbedderConfig(
            representation_name="synthetic_structure_embedding",
            output_mode="both",
        ),
        result=result,
        model=_model(),
        input_metadata=_input_metadata(),
        encoder_metadata=ENCODER_METADATA,
    )

    assert artifact.data["node_embedding"] is result.node_embedding
    assert artifact.data["global_embedding"] is result.global_embedding
    assert artifact.data["node_ids"] is result.node_ids
    assert artifact.data["valid_mask"] is result.valid_mask
    np.testing.assert_array_equal(artifact.data["node_ids"], NODE_IDS)
    np.testing.assert_array_equal(artifact.data["valid_mask"], VALID_MASK)
    assert artifact.data["node_embedding"].dtype == np.float64
    assert artifact.data["global_embedding"].dtype == np.float64
    assert artifact.data["node_embedding"].shape[1] != artifact.data["global_embedding"].shape[0]
    assert artifact.data["node_embedding"].shape[0] == 3
    assert np.isfinite(artifact.data["node_embedding"]).all()


def test_builder_uses_full_model_literal_config_fingerprint_and_encoder_metadata():
    config = StructuralEmbedderConfig(
        representation_name="synthetic_structure_embedding",
        output_mode="global",
    )
    model = _model()
    artifact = build_learned_artifact(
        config=config,
        result=_result("global"),
        model=model,
        input_metadata=_input_metadata(),
        encoder_metadata=ENCODER_METADATA,
    )

    assert artifact.meta["model"] == model.to_dict()
    assert artifact.meta["model"]["checkpoint"]["source"] == (
        "https://models.example/synthetic.bin"
    )
    assert "path" not in artifact.meta["model"]["checkpoint"]
    assert artifact.meta["parameters"] == config.to_dict()
    assert artifact.meta["provenance"]["config_fingerprint"] == compute_config_fingerprint(
        config.representation_name,
        config.to_dict(),
    )
    assert artifact.meta["encoder"] == ENCODER_METADATA
    assert artifact.meta["encoder"]["version"] != artifact.meta["provenance"]["eris_version"]
    assert "environment" not in artifact.meta["provenance"]


def test_builder_optional_provenance_environment_is_generic_strict_and_round_trips(tmp_path):
    environment = {
        "execution": {
            "requested": {"device": "auto", "precision": "fp32"},
            "resolved": {"device": "cpu", "precision": "fp32"},
        },
        "existing": {"value": "preserved"},
    }
    artifact = build_learned_artifact(
        config=StructuralEmbedderConfig(
            representation_name="synthetic_structure_embedding",
            output_mode="both",
        ),
        result=_result("both"),
        model=_model(),
        input_metadata=_input_metadata(),
        encoder_metadata=ENCODER_METADATA,
        provenance_environment=environment,
    )

    assert artifact.meta["provenance"]["environment"] == environment
    artifact.validate(strict=True)
    artifact.save(tmp_path)
    loaded = StructureArtifact.load(tmp_path)
    assert loaded.meta["provenance"]["environment"] == environment


def test_generic_builder_accepts_opaque_unicode_ids_and_subset_sized_results():
    node_embedding = np.array([[1.0, 2.0]], dtype=np.float32)
    node_ids = np.array(["biological node Ω"], dtype=np.str_)
    artifact = build_learned_artifact(
        config=StructuralEmbedderConfig(
            representation_name="synthetic_subset_embedding",
            output_mode="node",
        ),
        result=StructuralEmbeddingResult(
            node_embedding=node_embedding,
            node_ids=node_ids,
        ),
        model=_model(),
        input_metadata=_input_metadata(),
        encoder_metadata=ENCODER_METADATA,
    )

    assert artifact.data["node_embedding"] is node_embedding
    assert artifact.data["node_ids"] is node_ids
    assert artifact.meta["counts"] == {"n_nodes": 1}


@pytest.mark.parametrize("output_mode", ["node", "global", "both"])
def test_learned_artifacts_round_trip_in_canonical_output_order(tmp_path: Path, output_mode):
    artifact = _artifact(output_mode, with_mask=output_mode != "global")
    outdir = tmp_path / output_mode

    artifact.save(outdir)
    loaded = StructureArtifact.load(outdir, strict=True)

    loaded.validate(strict=True)
    assert loaded.meta == artifact.meta
    assert list(loaded.data) == [output["name"] for output in artifact.meta["outputs"]]
    for name, expected in artifact.data.items():
        np.testing.assert_array_equal(loaded.data[name], expected)


@pytest.mark.parametrize(
    ("output_mode", "result", "match"),
    [
        (
            "node",
            StructuralEmbeddingResult(
                node_embedding=np.ones((2, 3), dtype=np.float32),
                node_ids=np.array(["node-1"], dtype=np.str_),
            ),
            "align",
        ),
        (
            "node",
            StructuralEmbeddingResult(
                node_embedding=np.ones((2, 3), dtype=np.float32),
                node_ids=np.array(["node-1", "node-1"], dtype=np.str_),
            ),
            "unique",
        ),
        (
            "node",
            StructuralEmbeddingResult(
                node_embedding=np.ones((2, 3), dtype=np.float32),
                node_ids=np.array(["node-1", "node-2"], dtype=np.str_),
                valid_mask=np.array([True], dtype=np.bool_),
            ),
            "align",
        ),
        (
            "global",
            StructuralEmbeddingResult(
                node_embedding=np.ones((1, 3), dtype=np.float32),
                node_ids=np.array(["node-1"], dtype=np.str_),
            ),
            "requires global_embedding",
        ),
    ],
)
def test_builder_reuses_result_validation(output_mode, result, match):
    with pytest.raises(ExecutionError, match=match):
        build_learned_artifact(
            config=StructuralEmbedderConfig(
                representation_name="synthetic_invalid_embedding",
                output_mode=output_mode,
            ),
            result=result,
            model=_model(),
            input_metadata=_input_metadata(),
            encoder_metadata=ENCODER_METADATA,
        )


def test_false_mask_does_not_allow_nonfinite_placeholder_embeddings():
    node_embedding = np.array([[1.0, 2.0], [np.nan, 0.0]], dtype=np.float32)

    with pytest.raises(ExecutionError, match="finite"):
        build_learned_artifact(
            config=StructuralEmbedderConfig(
                representation_name="synthetic_invalid_embedding",
                output_mode="node",
            ),
            result=StructuralEmbeddingResult(
                node_embedding=node_embedding,
                node_ids=np.array(["node-1", "node-2"], dtype=np.str_),
                valid_mask=np.array([True, False], dtype=np.bool_),
            ),
            model=_model(),
            input_metadata=_input_metadata(),
            encoder_metadata=ENCODER_METADATA,
        )
