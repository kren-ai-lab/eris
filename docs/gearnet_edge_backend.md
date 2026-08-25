# Eris v0.1 — GearNet-Edge Backend Contract

This document is the authoritative M5.1 scientific and M5.2 dependency contract for the required
learned structural backend in Eris v0.1. It freezes the checkpoint identity, upstream
architecture, deterministic inference preprocessing, output semantics, licensing, provenance,
and optional runtime boundary needed by M5.3–M5.6. It does not claim that the production backend
or successful GearNet inference is implemented.

## Selected backend and checkpoint

The official Eris v0.1 learned backend is the GearNet-family **GearNet-Edge** variant pretrained
with the **Multiview Contrast** objective. The upstream model is configured as `GearNet` with edge
message passing, and its official checkpoint is `mc_gearnet_edge.pth`.

The checkpoint identity is frozen as:

| field | value |
| --- | --- |
| Zenodo record | v2, DOI [`10.5281/zenodo.7723075`](https://doi.org/10.5281/zenodo.7723075) |
| filename | `mc_gearnet_edge.pth` |
| exact size | `80,700,937` bytes |
| MD5 | `c35402108f14e43f20feb475918f9c26` |
| SHA-256 | `6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec` |

The SHA-256 value above is the authoritative byte identity for the official Eris backend. The
existing M4 checkpoint contract must verify it before loading the checkpoint.

The file is a modern `torch.save` ZIP archive containing an `OrderedDict` state dictionary with
138 tensor entries. It does not serialize a model class. This agrees with the official
[`script/pretrain.py`](https://github.com/DeepGraphLearning/GearNet/blob/main/script/pretrain.py),
which saves `model.state_dict()`.

Eris v0.1 must not bundle this approximately 80 MB checkpoint in the source repository, wheel,
or sdist. Checkpoint acquisition and loading policy is implemented in later M5 work.

## Frozen architecture

The architecture is frozen from the official
[`mc_gearnet_edge.yaml`](https://github.com/DeepGraphLearning/GearNet/blob/main/config/pretrain/mc_gearnet_edge.yaml)
configuration and the TorchDrug
[`models.GearNet`](https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/models/gearnet.py)
implementation:

| field | value |
| --- | --- |
| TorchDrug Registry model | `models.GearNet` |
| implementation class | `GeometryAwareRelationalGraphNeuralNetwork` |
| `input_dim` | `21` |
| `hidden_dims` | `[512, 512, 512, 512, 512, 512]` |
| `concat_hidden` | `true` |
| `short_cut` | `true` |
| `batch_norm` | `true` |
| activation | default `relu` |
| readout | `sum` |
| `num_relation` | `7` |
| `edge_input_dim` | `59` |
| `num_angle_bin` | `8` |
| resulting `output_dim` | `3072` |

With `concat_hidden = true`, the six 512-dimensional hidden outputs concatenate to produce the
3072-dimensional model output.

## Frozen inference preprocessing

Inference uses the upstream residue view and `AlphaCarbonNode`: there is one model node for each
retained residue with a CA atom. The residue input feature dimension is 21.

The graph construction is:

- `SequentialEdge(max_distance=2)`;
- `SpatialEdge(radius=10.0, min_distance=5, max_distance=None, max_num_neighbors=32)`;
- `KNNEdge(k=10, min_distance=5, max_distance=None)`;
- `edge_feature="gearnet"`.

The relations are five sequential relations, one spatial relation, and one kNN relation, for a
total of seven. The 59-dimensional GearNet edge feature is composed of:

| component | dimensions |
| --- | ---: |
| source residue type | 20 |
| target residue type | 20 |
| relation | 7 |
| sequential distance | 11 |
| spatial distance | 1 |
| **total** | **59** |

The upstream graph construction is defined in TorchDrug's
[`geometry/graph.py`](https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/layers/geometry/graph.py)
and edge/node layers in
[`geometry/function.py`](https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/layers/geometry/function.py).

### Pretraining-only augmentations

The following Multiview Contrast transformations are pretraining augmentations and must not run
during embedding inference:

- `SubsequenceNode(max_length=50)`;
- `SubspaceNode(entity_level="residue", min_neighbor=15, min_radius=15.0)`;
- `IdentityNode`;
- `RandomEdgeMask(mask_rate=0.15)`.

Inference constructs the deterministic model graph above and runs the frozen GearNet-Edge model;
it does not construct random contrastive views.

## Canonical Eris outputs

For one input structure, the canonical learned outputs are:

| Eris output | upstream value | shape |
| --- | --- | --- |
| `node_embedding` | `output["node_feature"]` | `(N, 3072)` |
| `global_embedding` | `output["graph_feature"][0]` | `(3072,)` |

`N` is the number of model nodes retained by the frozen preprocessing. The upstream `sum` readout
produces `graph_feature`; Eris must not add another pooling operation.

Every node-embedding row must map exactly to its biological input entity. M5.1 freezes that
scientific requirement, but the concrete `node_ids` and optional `valid_mask` implementation is
deferred to M5.3.

## Optional dependency boundary

The Eris core remains independent of Torch, TorchDrug, compiled PyG extensions, and RDKit and
continues to support Python `>=3.10,<3.13`. The GearNet backend has a deliberately narrower Eris
v0.1 runtime contract:

| dependency | frozen requirement |
| --- | --- |
| Python | `>=3.10,<3.11` |
| NumPy | `>=1.26,<2` |
| PyTorch | `==2.11.0` |
| TorchDrug | `==0.2.1` |
| `torch-scatter` | `==2.1.2` |
| `torch-cluster` | `==1.6.3` |
| `rdkit-pypi` | `==2022.9.5` |
| Ninja | `>=1.11,<2` |

Local PEP 440 wheel labels do not change the required public versions. For example,
`torch==2.11.0+cpu`, `torch-scatter==2.1.2+pt211cpu`, and
`torch-cluster==1.6.3+pt211cpu` satisfy this contract.

TorchDrug 0.2.1 officially supports Python through 3.10 and documents PyTorch support only
through 2.0. PyTorch 2.11 with TorchDrug 0.2.1 is therefore an **Eris-validated experimental
compatibility baseline**, not an upstream-supported TorchDrug combination. It was selected to
retain the relevant modern PyTorch checkpoint-loading security fixes and matching compiled PyG
wheels.

### CPU reference installation

CPU is the M5.2 installation baseline for Windows and Linux. Install the environment in this
order:

1. Install PyTorch 2.11 from the official CPU index:

   ```console
   python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
   ```

2. Install the matching compiled PyG extensions from the official PyG wheel index:

   ```console
   python -m pip install torch-scatter==2.1.2 torch-cluster==1.6.3 --find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html
   ```

3. Install Eris with its declarative optional dependency group:

   ```console
   python -m pip install "eris[gearnet]"
   ```

The ordering is intentional. PyPI alone does not provide portable binary resolution for the
required compiled PyG extensions, so a clean install may otherwise attempt source compilation.
The Eris extra declares package requirements but does not encode platform- or accelerator-specific
external indexes.

The extra does not install or download the checkpoint. Dependency availability also does not
guarantee checkpoint presence, GPU availability, network availability, successful JIT compilation,
or successful GearNet inference.

### Availability semantics

The GearNet dependency probe is explicit and lazy. Core imports and learned Registry discovery do
not import optional frameworks. When called directly, the probe checks the Python range,
distribution metadata, public versions, top-level import presence, and an isolated import smoke.
It does not instantiate GearNet, inspect a GPU, obtain a checkpoint, deliberately compile RSPMM,
or run a forward pass.

## Precision and accelerator boundary

FP32 and CPU are the required current M5 baselines. CUDA installation instructions, the CUDA
compatibility matrix, FP16/BF16 behavior, actual model construction, checkpoint loading, and
forward/inference validation remain deferred to M5.3 and M5.4. M5.2 does not promise CUDA support.

## Licensing

The code and checkpoint licenses are separate:

- GearNet source code: [MIT](https://github.com/DeepGraphLearning/GearNet/blob/main/LICENSE);
- TorchDrug source code:
  [Apache-2.0](https://github.com/DeepGraphLearning/torchdrug/blob/master/LICENSE);
- checkpoint and Zenodo record:
  [CC BY 4.0](https://zenodo.org/api/records/7723075).

The checkpoint license is taken from its Zenodo record metadata; it is not inferred from either
source-code license. Any later redistribution must preserve the applicable attribution and
license obligations.

## Citation and upstream provenance

The scientific project and paper are:

> Zhang, Zuobai; Xu, Minghao; Jamasb, Arian; Chenthamarakshan, Vijil; Lozano,
> Aurelie; Das, Payel; Tang, Jian. “Protein Representation Learning by Geometric
> Structure Pretraining.” International Conference on Learning Representations, 2023.

Primary provenance:

- [GearNet official repository](https://github.com/DeepGraphLearning/GearNet);
- [paper, arXiv:2203.06125](https://arxiv.org/abs/2203.06125);
- [official Zenodo checkpoint record v2](https://zenodo.org/records/7723075);
- [TorchDrug official repository](https://github.com/DeepGraphLearning/torchdrug).

The upstream source revisions inspected while freezing this contract were:

- GearNet: `28bf75f5dfb6cd7b9bf4394d153a87f69bddf1a6`;
- TorchDrug: `6066fbd82360abb5f270cba1eca560af01b8cc90`.

These revisions are reproducibility references for the inspected source. They are not asserted to
be the historical commits used to train the checkpoint. The checkpoint's scientific identity is
instead frozen by its official record, filename, architecture/configuration contract, and hashes.
