# Eris

Eris encodes protein structures into explicit or learned numerical representations for machine
learning. It reads PDB and mmCIF files, preserves structure and feature provenance, and returns
validated `StructureArtifact` objects.

Main capabilities:

- atom and residue point clouds;
- atom and residue graphs;
- contact maps, distograms, and residue voxel grids;
- deterministic residue features, including AAIndex1 and external numerical features;
- batch encoding, artifact packing, datasets, and Torch/PyG export;
- GearNet-Edge residue and structure embeddings.

## Installation

Eris requires Python 3.11–3.14.

```console
pip install erislib
```

Install optional integrations as needed:

```console
pip install "erislib[gearnet]"  # GearNet-Edge runtime
pip install "erislib[pyg]"      # PyTorch Geometric export
```

For a development setup, see [DEVELOPMENT.md](DEVELOPMENT.md).

The GearNet checkpoint is not bundled. On the first cache miss, Eris streams the registered
official checkpoint over HTTPS, verifies its registered SHA-256, and installs it in the shared
checkpoint cache. For offline use, pre-populate that cache or pass a local checkpoint with
`checkpoint_path` / `--checkpoint`.

## Python example

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

Use `eris.list_representations()` and `eris.get_representation_info(name)` to inspect the
registered explicit representations and their parameters.

## CLI example

```console
eris encode protein.pdb --representation residue_knn_graph --params '{"k": 8}' --out artifact_knn
eris info validate artifact_knn
```

Run `eris --help`, `eris representations list`, or `eris embedders list` for discovery.

## Explicit representations

| Name | Type | Granularity | Primary output |
| --- | --- | --- | --- |
| `atom_point_cloud` | cloud | atom | atom coordinates and features |
| `residue_point_cloud` | cloud | residue | residue coordinates and features |
| `atom_knn_graph` | graph | atom | atom k-nearest-neighbor graph |
| `residue_knn_graph` | graph | residue | residue k-nearest-neighbor graph |
| `residue_contact_graph` | graph | residue | distance-cutoff residue graph |
| `residue_distogram` | matrix | residue | pairwise distances |
| `residue_distogram_bins` | matrix | residue | distance-bin classes |
| `residue_contact_map` | matrix | residue | binary contacts |
| `residue_contact_map_bins` | matrix | residue | distance-bin contact classes |
| `residue_voxel_grid` | grid | residue | multi-channel 3D grid |

Residue coordinates can use the alpha carbon, residue-atom mean, or center of mass. Residue
features can combine amino-acid identity, fixed physicochemical descriptors, AAIndex1 records,
raw B-factor, explicitly interpreted pLDDT, backbone and side-chain torsions, local structural
features, and aligned external values. Residue graphs can also include chain relations, relative
direction, and interface annotations. See [structural feature contracts](docs/structural_features.md),
[representation defaults](docs/representation_defaults.md), and
[external residue features](docs/external_residue_features.md).

## GearNet-Edge

The `gearnet_edge` learned representation emits residue embeddings (`node`), one structure
embedding (`global`), or both. It uses the official `mc_gearnet_edge.pth` checkpoint through an
optional CPU/fp32 PyTorch runtime. See the
[GearNet-Edge backend contract](docs/gearnet_edge_backend.md) for checkpoint, preprocessing, and
runtime requirements.

## Documentation

- [Documentation map](docs/index.md)
- [Architecture](docs/architecture.md)
- [Representation defaults](docs/representation_defaults.md)
- [Artifact Schema](docs/artifact_schema.md)
- [External residue features](docs/external_residue_features.md)
- [GearNet-Edge backend](docs/gearnet_edge_backend.md)
- [Development](DEVELOPMENT.md)

## License and acknowledgement

Eris is released under the [MIT License](LICENSE).

GearNet-Edge uses the official Multiview Contrast checkpoint from
[Zenodo record 10.5281/zenodo.7723075](https://doi.org/10.5281/zenodo.7723075). Eris provides a
native runtime for that checkpoint and did not train it.
