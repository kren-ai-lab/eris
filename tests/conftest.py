from __future__ import annotations

from typing import Any, Dict
from morpheo.core.artifacts import StructureArtifact

def mk_artifact(*, data: Dict[str, Any], meta: Dict[str, Any]) -> StructureArtifact:
    base = {
        # core contract
        "encoder_name": meta.get("encoder_name", "test_encoder"),
        "encoder_version": meta.get("encoder_version", "0.0.0"),
        "encoder_fingerprint": meta.get("encoder_fingerprint", "test"),
        "type": meta.get("type", "graph"),
        "level": meta.get("level", "ca"),
        "family": meta.get("family", "residue_graph"),
        "params": meta.get("params", {}),
        "input": meta.get("input", {"source": "memory", "chain_id": None, "model_index": 0, "format": None}),
        "feature_keys": meta.get("feature_keys", {}),
    }
    base.update(meta)

    art = StructureArtifact(data=data, meta=base)
    art.finalize()
    return art
