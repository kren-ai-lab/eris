"""Helpers shared by the test modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from eris.core.artifact_schema import build_input_metadata
from eris.core.hashing import sha256_file
from eris.core.learned import (
    CheckpointDescriptor,
    StructuralEmbedderConfig,
    StructuralEmbeddingResult,
    StructuralModelDescriptor,
    build_learned_artifact,
)

if TYPE_CHECKING:
    from eris.core.artifact_schema import InputMetadata
    from eris.core.artifacts import StructureArtifact

DATA_DIR = Path("tests/data")
PDB = DATA_DIR / "1AQ2.pdb"
MMCIF = DATA_DIR / "1AQ2.cif"
MULTIMER_PDB = DATA_DIR / "3EC1.pdb"

TWO_CHAIN_PDB = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  \n"
    "TER\n"
    "ATOM      2  CA  GLY B   1       1.000   0.000   0.000  1.00 20.00           C  \n"
    "TER\n"
    "END\n"
)


def atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    xyz: float | tuple[float, float, float],
    *,
    chain_id: str = "A",
    insertion_code: str = " ",
    altloc: str = " ",
    record: str = "ATOM",
    occupancy: float = 1.0,
    bfactor: float = 10.0,
    element: str | None = None,
) -> str:
    """One fixed-column PDB atom record; a scalar ``xyz`` places the atom on the x axis."""
    x, y, z = (xyz, 0.0, 0.0) if isinstance(xyz, int | float) else xyz
    return (
        f"{record:<6}{serial:5d} {atom_name:>4s}{altloc:1s}{residue_name:>3s} {chain_id}"
        f"{residue_number:4d}{insertion_code:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {element or atom_name[0]:>2s}  "
    )


def write_pdb(path: Path, lines: list[str], *, header: list[str] | None = None) -> Path:
    """Write atom records, inserting ``TER`` whenever the chain changes."""
    body: list[str] = list(header or [])
    previous_chain: str | None = None
    for line in lines:
        chain = line[21]
        if previous_chain is not None and chain != previous_chain:
            body.append("TER")
        body.append(line)
        previous_chain = chain
    path.write_text("\n".join([*body, "TER", "END", ""]), encoding="ascii")
    return path


def input_metadata(source: Path | str = "memory://synthetic", digest: str = "a") -> InputMetadata:
    real = isinstance(source, Path)
    return build_input_metadata(
        source=source,
        source_sha256=sha256_file(source) if real else digest * 64,
        input_format="pdb",
        model_index=0,
        chain_ids=["A"],
    )


def learned_artifact(
    output_mode: Literal["both", "global", "node"] = "both",
    *,
    source: Path | str = "memory://synthetic",
    dtype: type = np.float32,
    with_mask: bool = False,
    representation_name: str = "synthetic_structure_embedding",
) -> StructureArtifact:
    """Synthetic learned embedding artifact with two residues and a 3-wide embedding."""
    node = output_mode in {"node", "both"}
    glob = output_mode in {"global", "both"}
    result = StructuralEmbeddingResult(
        node_embedding=np.arange(6, dtype=dtype).reshape(2, 3) if node else None,
        global_embedding=np.asarray([7.0, 8.0, 9.0], dtype=dtype) if glob else None,
        node_ids=np.asarray(["A:1::ALA", "A:2::GLY"], dtype=np.str_) if node else None,
        valid_mask=np.asarray([True, False]) if node and with_mask else None,
    )
    return build_learned_artifact(
        config=StructuralEmbedderConfig(representation_name=representation_name, output_mode=output_mode),
        result=result,
        model=StructuralModelDescriptor(
            architecture="synthetic-structural-model",
            checkpoint=CheckpointDescriptor(
                identifier="synthetic-checkpoint",
                revision="revision-1",
                source="memory://synthetic-checkpoint",
                sha256="b" * 64,
            ),
        ),
        input_metadata=input_metadata(source),
        encoder_metadata={"implementation": "tests.synthetic.LearnedEncoder", "version": "1"},
    )
