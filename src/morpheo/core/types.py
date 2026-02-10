from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import json
import numpy as np


@dataclass(frozen=True)
class StructureInput:
    source: str
    chain_id: Optional[str] = None
    model_index: int = 0

    def to_string(self) -> str:
        return f"source={self.source}|chain={self.chain_id}|model={self.model_index}"


@dataclass
class StructureArtifact:
    encoder_name: str
    data: Dict[str, np.ndarray]
    meta: Dict[str, Any]

    def save(self, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outdir / "data.npz", **self.data)
        (outdir / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @staticmethod
    def load(indir: Path) -> "StructureArtifact":
        meta = json.loads((indir / "meta.json").read_text())
        npz = np.load(indir / "data.npz")
        data = {k: npz[k] for k in npz.files}
        return StructureArtifact(
            encoder_name=meta.get("encoder_name", "unknown"),
            data=data,
            meta=meta,
        )


class StructuralEncoder(Protocol):
    name: str

    def fingerprint(self) -> str: ...
    def encode(self, inp: StructureInput) -> StructureArtifact: ...
