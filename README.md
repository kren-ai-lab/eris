# Eris

Eris builds reproducible numerical representations from protein 3D structures.

Eris v0.1 has two representation families:

- explicit structural representations, such as point clouds, graphs, matrices, and voxel grids;
- the optional learned `gearnet_edge` structural embedder.

The core package works without Torch, TorchDrug, RDKit, MSVC, or Ninja. Learned GearNet execution
is an optional environment with a separately supplied checkpoint.

## Installation

Core Eris targets Python `>=3.10,<3.13`.

From a checkout or source distribution:

```console
python -m pip install .
```

For development:

```console
python -m pip install -e ".[dev]"
```

The core install supports explicit representations and public discovery, encoding, artifact
inspection, and cache inspection. It does not install the GearNet runtime stack.

## Explicit representations

The public v0.1 explicit representations are exactly:

- `atom_point_cloud`
- `residue_point_cloud`
- `atom_knn_graph`
- `residue_knn_graph`
- `residue_contact_graph`
- `residue_distogram`
- `residue_distogram_bins`
- `residue_contact_map`
- `residue_contact_map_bins`
- `residue_voxel_grid`

Discover them from the CLI:

```console
eris representations list
eris representations info residue_knn_graph
```

Encode one structure:

```console
eris encode protein.pdb \
  --representation residue_knn_graph \
  --params '{"k": 8}' \
  --out artifact_knn
```

Inspect and validate the resulting artifact:

```console
eris info validate artifact_knn
eris info show artifact_knn
```

The equivalent Python API is:

```python
import eris

artifact = eris.encode(
    "protein.pdb",
    representation="residue_knn_graph",
    k=8,
)
artifact.validate(strict=True)
artifact.save("artifact_knn")
```

## Batch, pack, dataset, and export workflow

For multiple structures, provide a CSV manifest with `id,path,chain,model` and run one
representation per batch invocation:

```console
eris batch encode structures.csv \
  --representation residue_knn_graph \
  --out artifacts
eris batch embed structures.csv \
  --representation gearnet_edge \
  --checkpoint path/to/mc_gearnet_edge.pth \
  --out learned-artifacts
eris pack artifacts --out index.csv
```

Batch results are recorded with an explicit `success` or `failed` status; failures are not
silently removed. `pack` creates a deterministic metadata-only index and does not rewrite or
archive artifact arrays. `ErisFolderDataset` consumes that index directly.

Downstream exports operate on one strict Artifact v2 directory:

```console
eris export to-torch --artifact-dir artifacts/1/residue_knn_graph --out graph.pt
eris export to-pyg --artifact-dir artifacts/1/residue_knn_graph --out graph_pyg.pt
```

Torch export supports graph, cloud, matrix, grid, and learned outputs. PyG export is limited to
artifacts that already contain public graph topology; matrices, grids, clouds, and learned
embeddings are never converted into artificial graphs.

## Artifact format

Eris public operations write Artifact Schema v2 directories:

```text
artifact/
├── meta.json
└── data.npz
```

`meta.json` records reproducible metadata: representation identity, input identity, parameters,
encoder or model provenance, outputs, feature semantics, shapes, dtypes, and counts. `data.npz`
stores the arrays. Schema v1 was a pre-release format and is not a supported public v0.1 artifact
format.

## Optional GearNet-Edge learned embeddings

The v0.1 learned backend is:

- public representation: `gearnet_edge`;
- model: GearNet-Edge;
- output modes: `node`, `global`, `both`;
- checkpoint: official external `mc_gearnet_edge.pth`, not bundled and not downloaded
  automatically.

`node` mode emits residue-level embeddings, `global` mode emits a structure-level embedding, and
`both` emits both. GearNet-Edge v0.1 embeddings are 3072-dimensional.

The GearNet runtime is optional and currently validated as a Python 3.10 CPU/fp32 environment.
Install order matters because PyPI alone may not resolve the compiled PyG extension wheels:

```console
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter==2.1.2 torch-cluster==1.6.3 --find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html
python -m pip install ".[gearnet]"
```

On Windows, the first GearNet/TorchDrug execution can require MSVC x64 Build Tools and Ninja for
PyTorch's native extension/JIT path. This does not apply to Eris core or explicit
representations.

Discover the learned backend and its model metadata without running GearNet:

```console
eris embedders list
eris embedders info gearnet_edge
eris models list
eris models info gearnet_edge
```

Run learned embedding with a local official checkpoint:

```console
eris embed protein.pdb \
  --representation gearnet_edge \
  --output-mode both \
  --checkpoint path/to/mc_gearnet_edge.pth \
  --device cpu \
  --precision fp32 \
  --out artifact_gearnet

eris info validate artifact_gearnet
eris info show artifact_gearnet
```

The equivalent Python API is:

```python
import eris

artifact = eris.embed(
    "protein.pdb",
    representation="gearnet_edge",
    output_mode="both",
    checkpoint_path="mc_gearnet_edge.pth",
    device="cpu",
    precision="fp32",
)
artifact.validate(strict=True)
```

## Cache inspection and clearing

Eris separates the Artifact cache from the Checkpoint cache.

Inspect cache state:

```console
eris cache info
```

Clear only explicit/learned representation artifacts:

```console
eris cache clear --target artifacts
```

Clear only learned model checkpoint assets:

```console
eris cache clear --target checkpoints
```

Clear both Eris-managed cache namespaces:

```console
eris cache clear --target all
```

Add `--yes` to skip the interactive confirmation, for example:

```console
eris cache clear --target artifacts --yes
```

Deleting checkpoints can require supplying `--checkpoint path/to/mc_gearnet_edge.pth` again for a
future learned run. Cache paths are implementation details, not public API.

## Public Python API

The stable M6 public root API is:

```python
eris.__version__
eris.list_representations()
eris.get_representation_info("residue_knn_graph")
eris.list_embedders()
eris.get_embedder_info("gearnet_edge")
eris.encode(...)
eris.embed(...)
eris.batch_encode(...)
eris.batch_embed(...)
eris.pack(...)
eris.to_torch(...)
eris.to_pyg(...)
eris.RepresentationNotFoundError
```

Internal Registries, encoder classes, learned runners, backend implementations, descriptors, and
cache classes are not part of the root public API.
