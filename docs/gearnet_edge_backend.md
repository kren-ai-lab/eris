# Eris v0.1 — GearNet-Edge Backend Contract

This document is the authoritative M5.1–M5.6 contract for the required learned structural
backend in Eris v0.1. It freezes the checkpoint identity, upstream architecture, deterministic
inference preprocessing, CPU/fp32 execution, output semantics, licensing, provenance, and
optional runtime boundary. The production backend completes preprocessing, verified model loading,
inference, Artifact v2 provenance, persistence, and cache reuse through the generic learned runner.

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
or sdist. Checkpoint acquisition remains an explicit caller responsibility; the runtime verifies
and loads only a supplied local checkpoint.

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
produces `output["graph_feature"]` with shape `(1, 3072)`; Eris exposes its first row and must not
add another pooling operation.

Every node-embedding row must map exactly to its biological input entity. M5.1 freezes that
scientific requirement. M5.3 implements concrete `node_ids` through an explicit biological-index
carrier; the conservative v0.1 policy fails the whole representation rather than emitting a
`valid_mask` for retained canonical residues without CA.

## Conservative v0.1 biological selection, transport and mapping

GearNet v0.1 uses one CA-based residue node for each accepted residue in the selected model and
single chain. This is an initial conservative backend contract, not a universal biological rule.
The accepted residue names are exactly the canonical 20:

```text
ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE
LEU LYS MET PHE PRO SER THR TRP TYR VAL
```

Selection combines Gemmi's structural `Residue.het_flag == "A"` ATOM classification with
`gemmi.find_tabulated_residue(name).is_amino_acid()` and the explicit canonical-20 set. This
prevents a HETATM residue named `ALA` from passing by name alone. MSE and every other nonstandard
or modified residue are excluded without chemical conversion; in particular, MSE is not converted
to MET. Water, ligands, metals, ions, and all other HETATM residues are excluded before CA lookup,
so a calcium ion whose atom name is `CA` cannot become a model node.

Every accepted canonical residue must have a finite CA coordinate. A missing or unusable CA fails
the complete representation: there is no silent omission, alternate-atom substitution,
imputation, or `valid_mask=False`. GearNet v0.1 freezes CA alternate-location selection to
`altloc = first`; other selector modes are not exposed by `GearNetEdgeConfig`.

Eris records an ordered biological identity table before TorchDrug. Model-node records selected
by the v0.1 policy carry an explicit, potentially sparse index into that table. Filtering removes
records without reordering or compacting their biological indexes. The index is stored as the
TorchDrug residue-level attribute `eris_residue_index`, which survives packing,
`AlphaCarbonNode`, residue compaction, and the frozen `GraphConstruction`. Final `node_ids` are
resolved through that carrier and the biological table; they are not reconstructed from output
length, compacted residue position, or synthetic PDB numbering.

TorchDrug receives a deterministic, adapter-local, single-chain CA-only PDB made from records that
have already been selected. Its chain and residue numbers are transport-only identifiers. Synthetic
residue numbering begins at 2 to avoid TorchDrug 0.2.1's terminal dummy-residue collision. The
temporary PDB is removed after TorchDrug materializes the graph and its path is not provenance.
This sanitized single-chain boundary also prevents TorchDrug's raw multichain residue-key
collisions. Tests against a canonical clean PDB demonstrate exact graph equivalence for node
features, coordinates, edges, relations, and edge features.

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

FP32 on CPU is the only v0.1 GearNet execution contract. The backend advertises only `cpu` and
`fp32` and rejects CUDA, FP16, and BF16 rather than falling back or casting silently. It does not
query CUDA availability. Future accelerator or reduced-precision work requires a separately
validated contract.

## Verified M5.4 model-loading foundation

The internal M5.4 loading foundation constructs the exact frozen `models.GearNet` architecture
and loads the official `mc_gearnet_edge.pth` state dictionary strictly. The verified official
checkpoint has 138 tensor entries, matches the constructed model without missing, unexpected, or
shape-mismatched keys, and produces the expected
`GeometryAwareRelationalGraphNeuralNetwork` with `output_dim = 3072`.

Loading consumes M4's `VerifiedCheckpoint`; it does not accept an arbitrary unverified path or
rehash the file. Checkpoint paths are opaque, so the extensionless canonical `CheckpointCache`
path named `checkpoint` is supported. Deserialization uses only:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

The state-dictionary root, keys, and tensor values are validated before
`model.load_state_dict(..., strict=True)`. There is no unsafe deserialization fallback, key
rewriting, prefix removal, partial load, or alternate architecture fallback.

The current operational contract is CPU/fp32 only. Parameters and buffers are checked after
loading, and the model is placed in `eval()` mode so BatchNorm and other training-sensitive
modules use inference behavior. Parameters retain their normal `requires_grad` values.

## Verified M5.4 deterministic inference

The internal backend delegates biological preprocessing and checkpoint loading to the M5.3 and
M5.4 foundations. Immediately before inference it revalidates CPU/fp32 execution, eval mode,
model tensors, graph dimensions, relation count, carrier alignment and uniqueness, and biological
node-ID alignment. The exact forward is:

```python
with torch.inference_mode():
    output = model(
        prepared.graph,
        prepared.graph.node_feature.float(),
        all_loss=None,
        metric=None,
    )
```

The prepared integer node features remain unchanged. Eris validates the upstream tensors as
finite CPU `torch.float32` arrays with shapes `(N, 3072)` and `(1, 3072)`. It converts them directly
with `.cpu().numpy()` and exposes `graph_feature[0]` as the global embedding. No Eris pooling,
scientific cast, `detach()`, autocast, manual seed, or output-mode-dependent forward is applied.
The result is then checked by the authoritative M4 `StructuralEmbeddingResult` validator;
`valid_mask` is `None` for GearNet v0.1.

### Determinism evidence

On the frozen Python 3.10, PyTorch 2.11 CPU, TorchDrug 0.2.1, and NumPy 1.26 stack, the official
checkpoint and canonical validation fixture produced bitwise-identical node and global embeddings
across three forwards in one process and a fresh process. Both maximum absolute and relative
differences were zero, and the biological mapping was identical. This is evidence for the frozen
CPU/fp32 stack, not a universal determinism promise for future platforms, accelerators, or
precisions.

### Windows JIT and observed performance

On the validated Windows environment, the first forward materialized
`torchdrug.utils.torch.torch_ext` through PyTorch's C++ extension machinery using Ninja and a
discoverable MSVC x64 `cl.exe`. The observed first compilation took about 25.8 seconds and the
first complete forward about 26.1 seconds; warm forwards were about 45–49 milliseconds. These are
observations, not performance contracts. A fresh process reused the JIT cache. This path did not
materialize the `generalized_rspmm`, spmm, or embedding loaders.

TorchDrug and PyTorch 2.11 used different nominal/effective cache-path conventions during the
experiment. Eris neither depends on those paths nor manages, relocates, or deletes the PyTorch JIT
cache. TorchDrug also emits an upstream deprecation warning once per process because it calls
`torch.cross` without `dim`; Eris does not patch or globally silence that warning.

## Artifact Schema v2 identity and provenance

The frozen learned representation and Eris adapter identity is:

| field | value |
| --- | --- |
| `representation_name` | `gearnet_edge` |
| `backend_id` | `gearnet-edge` |
| implementation | `eris.core.learned.backends.gearnet.backend.GearNetEdgeBackend` |
| adapter/encoder version | `1` |

Adapter version `1` versions the Eris GearNet contract. It is independent of the Eris package
version, Artifact Schema version, TorchDrug version, upstream source revision, and checkpoint
revision. The Schema v2 `model` block records architecture `GearNet-Edge` and the official
checkpoint identifier, SHA-256, Zenodo record revision, and source. Local checkpoint paths,
checkpoint-cache paths, and temporary transport paths are never model identity.

### Scientific preprocessing parameters

Artifact `parameters` records the frozen canonical-20 selection, CA coordinate source,
nonstandard/MSE/hetero/water/ligand/metal/ion exclusions, fail-fast missing-CA behavior, and
effective `altloc = first` policy. It also records the AlphaCarbon node layer; sequential,
spatial, and kNN edge settings; GearNet edge features; seven relations; and 21/59 node/edge input
widths. The selected chain and model index remain in the canonical `input` block rather than being
duplicated in `parameters`.

The temporary sanitized PDB is an implementation transport and is never persisted. The Artifact
input source is the caller's current structure locator and its SHA-256 is the scientific content
identity.

### Execution and runtime provenance

Generation provenance records requested and resolved execution through the generic M4 mechanism.
For the frozen backend both resolved device and precision are `cpu` and `fp32`. Effective precision
participates in learned cache identity; device does not.

Schema v2 and M4 do not define an authoritative dependency-version block. Python, Torch,
TorchDrug, PyG-extension, MSVC, and Ninja versions therefore remain documented operational
compatibility information rather than being inserted into model identity or a new schema field.
The JIT cache location is neither provenance nor scientific identity. Extending runtime
environment capture requires a later general convention rather than a GearNet-only schema.

### Cache identity and reuse

Learned artifact cache identity includes representation name, adapter implementation and version,
canonical preprocessing parameters, input content and selection, parser hint, path-independent
model/checkpoint identity, and effective precision. It excludes structure and checkpoint locators,
checkpoint source mirrors, device locator, temporary PDB paths, and Torch extension cache paths.

On a valid cache hit, the runner refreshes only `input.source` to the current structure locator.
It preserves the original generation execution provenance, model/checkpoint source provenance,
arrays, mappings, and scientific metadata and does not probe or instantiate the backend.

### Learned output metadata

The learned builder emits `node_embedding` plus `node_ids` for node mode, `global_embedding` for
global mode, and all three in canonical order for both mode. Shapes are `(N, 3072)`, `(3072,)`,
and `(N,)`; `feature_keys` is empty because embedding dimensions do not have 3072 invented feature
names. `valid_mask` is absent for the frozen GearNet policy. Granularity is respectively
`residue`, `structure`, or `mixed`, while coordinate source is always `model_specific`.

The production learned Registry contains `gearnet_edge` exactly once, with backend ID
`gearnet-edge`, adapter version `1`, and `GearNetEdgeConfig`. Registration stores only static
metadata and lazy callables: importing Eris or inspecting the Registry does not import Torch,
TorchDrug, `torch-scatter`, `torch-cluster`, or RDKit. Availability reports an actionable optional
dependency error before factory construction or checkpoint access in a core-only environment.

## Production integration evidence

The M5.6 reference smoke uses the production Registry and generic `StructuralEmbedderRunner` from
a real `StructureInput`. With the official verified checkpoint and the frozen Python 3.10 CPU/fp32
stack, the cache-miss path covers availability, backend construction, execution resolution,
checkpoint verification/cache, Gemmi biological selection, TorchDrug graph construction, strict
model loading, inference, Artifact Schema v2 assembly, validation, persistence, and artifact-cache
write.

For output mode `both`, the validated artifact contains finite float32 `node_embedding` with shape
`(N, 3072)`, finite float32 `global_embedding` with shape `(3072,)`, and unique Unicode `node_ids`
with shape `(N,)`. Node and global modes preserve their frozen residue and structure granularities
without running a different scientific forward. Save/load preserves arrays, dtypes, shapes,
counts, parameters, provenance, model identity, and empty `feature_keys`.

A second request with identical scientific content and precision but a different source locator
hits the artifact cache before availability, factory construction, execution resolution,
checkpoint resolution, preprocessing, model loading, or inference. Only `input.source` is
refreshed. Separate empty artifact and checkpoint caches produce bitwise-equal node embeddings,
global embeddings, and node IDs, demonstrating independent generation determinism rather than
cache reuse. A retained canonical residue without CA fails with an actionable preprocessing error
before model load/forward and leaves no artifact-cache entry.

The real-stack smoke is offline and reads the official checkpoint from an explicit local path
(conventionally `ERIS_GEARNET_CHECKPOINT`); normal tests do not require the checkpoint or network.
On Windows, the first forward may materialize PyTorch's `torch_ext` JIT extension and therefore
requires discoverable MSVC x64 and Ninja. That first run can be considerably slower; the JIT cache
is reused, but its path and timings are not Eris API, provenance, or cache identity.

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
