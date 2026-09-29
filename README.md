# Eris

Eris converts protein three-dimensional structures into reproducible numerical representations
for machine learning. It supports explicit structural arrays derived from PDB/mmCIF coordinates
and learned GearNet-Edge embeddings.

Core capabilities include:

- atom and residue point clouds;
- atom and residue graphs;
- contact maps and distograms;
- residue voxel grids;
- configurable residue coordinates and deterministic node features;
- strict, portable artifacts with provenance;
- batch, pack, dataset, Torch, and PyG workflows;
- native GearNet-Edge node and global embeddings.

## Installation

Current development metadata supports Python 3.11 and 3.12 (`>=3.11,<3.13`). Public distribution
naming and publication remain release-hardening work, so install from a repository checkout or
source distribution:

```console
python -m pip install .
```

Optional extras are:

```console
python -m pip install ".[gearnet]"  # GearNet-Edge runtime
python -m pip install ".[pyg]"      # PyTorch Geometric export
python -m pip install ".[dev]"      # contributor tools
```

The GearNet checkpoint is not bundled or downloaded automatically.

## Quick start

Python:

```python
import eris

artifact = eris.encode(
    "protein.pdb",
    representation="residue_knn_graph",
    chain_id="A",
    k=8,
    output="artifact_knn",
)
```

Command line:

```console
eris encode protein.pdb --representation residue_knn_graph --params '{"k": 8}' --out artifact_knn
eris info validate artifact_knn
```

Use `eris --help`, `eris representations list`, or
`eris.get_representation_info(name)` for discovery.

## Explicit representations

Eris provides ten explicit structural representations:

| Name | Type | Granularity | Description |
| --- | --- | --- | --- |
| `atom_point_cloud` | cloud | atom | Atomic coordinates and features |
| `residue_point_cloud` | cloud | residue | Configurable residue coordinates and features |
| `atom_knn_graph` | graph | atom | Atomic k-nearest-neighbor graph |
| `residue_knn_graph` | graph | residue | Residue k-nearest-neighbor graph |
| `residue_contact_graph` | graph | residue | Residue graph defined by spatial contacts |
| `residue_distogram` | matrix | residue | Pairwise residue distances |
| `residue_distogram_bins` | matrix | residue | Discretized pairwise distances |
| `residue_contact_map` | matrix | residue | Binary residue contact map |
| `residue_contact_map_bins` | matrix | residue | Distance-binned contact map |
| `residue_voxel_grid` | grid | residue | Multi-channel three-dimensional residue grid |

Residue representations support alpha-carbon, residue-atom mean, and center-of-mass coordinates.
Node features can combine amino-acid identity, physicochemical descriptors, true AAIndex1
records, raw B-factor, explicitly interpreted pLDDT, and generic external residue values. Built-in
and external blocks compose into one deterministic hybrid `node_attr` matrix.

See [representation defaults](docs/representation_defaults.md) and
[external residue features](docs/external_residue_features.md) for the scientific contracts.

## GearNet-Edge

The learned representation `gearnet_edge` provides:

- `node`: residue embeddings with biological node IDs;
- `global`: one learned structure embedding;
- `both`: both outputs.

It uses the verified external `mc_gearnet_edge.pth` checkpoint through a native, optional PyTorch
runtime. Runtime, preprocessing, checkpoint, and provenance requirements are documented in the
[GearNet-Edge backend contract](docs/gearnet_edge_backend.md).

Deterministic explicit-graph `graph_attr` statistics are not learned GearNet embeddings.

## Artifacts and workflows

Representations are returned as `StructureArtifact` objects and can be saved as:

```text
artifact/
|-- meta.json
`-- data.npz
```

The Python API exposes encoding, embedding, batch processing, packing, and Torch/PyG export.
Artifacts retain exact feature keys and can be loaded independently of their original external
feature file after finalization.

See the [Artifact Schema](docs/artifact_schema.md) and
[architecture](docs/architecture.md) for the complete contracts.

## Documentation

- [Documentation map](docs/index.md)
- [Architecture](docs/architecture.md)
- [Representation defaults](docs/representation_defaults.md)
- [Artifact Schema](docs/artifact_schema.md)
- [External and hybrid residue features](docs/external_residue_features.md)
- [GearNet-Edge backend](docs/gearnet_edge_backend.md)
- [Contributor workflow](DEVELOPMENT.md)

## License and acknowledgement

Eris is released under the [MIT License](LICENSE).

GearNet-Edge uses the official Multiview Contrast checkpoint from
[Zenodo record 10.5281/zenodo.7723075](https://doi.org/10.5281/zenodo.7723075). Eris supplies a
native runtime for that checkpoint and did not train or create it. Citation and final release
metadata remain part of release hardening.
