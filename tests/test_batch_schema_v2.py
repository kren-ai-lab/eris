from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eris.core.artifacts import StructureArtifact
from eris.core.registry import get_encoder
from eris.core.types import StructureInput
from eris.pipelines.batch_encode import batch_encode

PDB = Path("tests/data/1AQ2.pdb")
ENCODER = "residue_point_cloud"


def _manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"id,path\nprotein,{PDB}\n", encoding="utf-8")
    return manifest


def _legacy_output(out: Path) -> Path:
    target = out / "protein" / ENCODER
    target.mkdir(parents=True)
    np.savez_compressed(target / "data.npz", pos=np.zeros((1, 3), dtype=np.float32))
    (target / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder_name": ENCODER,
                "encoder_version": "0.1.0",
                "encoder_fingerprint": "legacy",
                "type": "cloud",
                "level": "residue",
                "params": {},
                "input": "legacy",
                "feature_keys": {},
                "shapes": {"pos": [1, 3]},
                "dtypes": {"pos": "float32"},
                "counts": {"n_nodes": 1},
            }
        ),
        encoding="utf-8",
    )
    return target


def test_batch_requires_resume_to_reuse_existing_strict_valid_schema_v2(tmp_path):
    out = tmp_path / "out"
    target = out / "protein" / ENCODER
    artifact = get_encoder(ENCODER, use_cache=False).encode(StructureInput(source=str(PDB)))
    artifact.save(target)

    results = batch_encode(
        manifest_path=_manifest(tmp_path),
        encoders=[ENCODER],
        out=out,
        no_cache=True,
    )

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].error_type == "FileExistsError"
    assert json.loads((target / "meta.json").read_text(encoding="utf-8"))["schema_version"] == 2


def test_batch_resume_reuses_existing_strict_valid_schema_v2(tmp_path):
    out = tmp_path / "out"
    target = out / "protein" / ENCODER
    artifact = get_encoder(ENCODER, use_cache=False).encode(StructureInput(source=str(PDB)))
    artifact.save(target)

    results = batch_encode(
        manifest_path=_manifest(tmp_path),
        encoders=[ENCODER],
        out=out,
        no_cache=True,
        resume=True,
    )

    assert len(results) == 1
    assert results[0].status == "success"
    assert results[0].artifact_dir == str(target)


def test_batch_resume_does_not_silently_reuse_existing_schema_v1(tmp_path):
    out = tmp_path / "out"
    target = _legacy_output(out)

    results = batch_encode(
        manifest_path=_manifest(tmp_path),
        encoders=[ENCODER],
        out=out,
        no_cache=True,
        resume=True,
    )

    assert len(results) == 1
    assert results[0].status == "success"
    assert StructureArtifact.load(target, strict=True).meta["schema_version"] == 2


def test_batch_fail_fast_persists_existing_output_failure(tmp_path):
    out = tmp_path / "out"
    _legacy_output(out)
    results_path = tmp_path / "results.csv"

    with pytest.raises(FileExistsError):
        batch_encode(
            manifest_path=_manifest(tmp_path),
            encoders=[ENCODER],
            out=out,
            no_cache=True,
            fail_fast=True,
            results_path=results_path,
        )

    results = results_path.read_text(encoding="utf-8")
    assert "failed" in results
    assert "FileExistsError" in results
