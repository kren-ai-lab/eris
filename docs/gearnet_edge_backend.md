# GearNet-Edge backend

`gearnet_edge` is the only registered learned representation in Eris v0.1. It uses a native
optional PyTorch implementation of GearNet-Edge with frozen preprocessing and the official
Multiview Contrast checkpoint. The backend is inference-only.

## Public configuration

```python
artifact = eris.embed(
    "protein.pdb",
    representation="gearnet_edge",
    chain_id="A",
    output_mode="both",
    device="cpu",
    precision="fp32",
)
```

| Parameter | Values | Default |
| --- | --- | --- |
| `output_mode` | `node`, `global`, `both` | `both` |
| `device` | `auto` or `cpu` | `auto` |
| `precision` | `fp32` | `fp32` |
| `checkpoint_path` | optional local official checkpoint path | none |

CUDA, FP16, and BF16 are rejected. Eris does not silently fall back or cast. The checkpoint is
not bundled. Resolution uses this fixed order: an already verified cache entry, an explicitly
supplied local path, then the registered official HTTPS download. A local path is therefore the
offline override, not a separate storage system.

The equivalent CLI is:

```console
eris embed protein.pdb --representation gearnet_edge --chain A --output-mode both \
  --device cpu --precision fp32 --out artifact_gearnet
```

The first command on a clean cache requires network access. For an offline run, add
`--checkpoint mc_gearnet_edge.pth`; Eris verifies and installs that file into the same cache.

## Checkpoint identity

| Field | Value |
| --- | --- |
| Zenodo record | [10.5281/zenodo.7723075, record v2](https://zenodo.org/records/7723075) |
| Filename | `mc_gearnet_edge.pth` |
| Size | 80,700,937 bytes |
| MD5 | `c35402108f14e43f20feb475918f9c26` |
| SHA-256 | `6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec` |

SHA-256 is the authoritative byte identity. The checkpoint is a `torch.save` ZIP archive
containing an `OrderedDict` state dictionary with 138 tensors; it does not contain a serialized
model class.

The acquisition URL is trusted registration metadata and accepts no user override. Eris streams
the response to a temporary file, rejects a non-HTTPS redirect, verifies the complete SHA-256,
and then uses the normal atomic checkpoint-cache installation path. A failed transfer or digest
check installs nothing and removes the temporary file. The acquisition URL and other transport
paths are excluded from artifact metadata and scientific cache identity.

The loader accepts only a `VerifiedCheckpoint`, deserializes with:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

It validates the state-dictionary root, keys, shapes, and tensor values before strict model
loading. There is no unsafe fallback, key rewriting, prefix removal, partial load, or alternate
architecture.

## Frozen model architecture

The implementation follows the official
[`mc_gearnet_edge.yaml`](https://github.com/DeepGraphLearning/GearNet/blob/main/config/pretrain/mc_gearnet_edge.yaml):

| Field | Value |
| --- | --- |
| model | GearNet with edge message passing |
| native class | `NativeGearNetEdgeModel` |
| input dimension | 21 |
| hidden dimensions | six layers of 512 |
| concatenate hidden layers | true |
| shortcut | true |
| batch normalization | true |
| activation | ReLU |
| readout | sum |
| relations | 7 |
| edge input dimension | 59 |
| angle bins | 8 |
| output dimension | 3072 |

The implementation lives in `eris/core/learned/backends/gearnet/native_model.py`. Six
concatenated 512-dimensional hidden outputs produce the 3072-dimensional representation.

## Frozen biological selection

Preprocessing is implemented in
`eris/core/learned/backends/gearnet/preprocessing.py`.

- One selected model and one chain are processed.
- Accepted residues are ATOM-classified members of the canonical 20 amino acids.
- MSE and all other modified or nonstandard residues are excluded without conversion.
- Water, ligands, metals, ions, and other HETATM residues are excluded.
- Each retained residue must have one finite CA coordinate.
- As a backend/checkpoint-specific exception to the general explicit-encoder default, CA
  alternate-location selection is `first`: the first encountered CA conformer is retained. This
  frozen choice is recorded in GearNet preprocessing parameters and learned cache identity.
- A missing or unusable CA fails the whole representation; no residue is silently omitted and no
  `valid_mask` is emitted.

The canonical accepted names are:

```text
ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE
LEU LYS MET PHE PRO SER THR TRP TYR VAL
```

Preprocessing records an ordered biological identity table before graph construction. Every native
model node carries its original table index, and final `node_ids` are resolved through that
carrier. IDs are not reconstructed from output length or compacted positions.

## Frozen graph construction

| Component | Parameters |
| --- | --- |
| node layer | alpha carbon |
| sequential edges | `max_distance=2` |
| spatial edges | `radius=10.0`, `min_distance=5`, no maximum distance, at most 32 neighbors |
| kNN edges | `k=10`, `min_distance=5`, no maximum distance |
| edge feature | `gearnet` |

There are five sequential relations, one spatial relation, and one kNN relation.

| Edge feature component | Width |
| --- | ---: |
| source residue type | 20 |
| target residue type | 20 |
| relation | 7 |
| sequential distance | 11 |
| spatial distance | 1 |
| total | 59 |

Random edge masking, subsequence selection, subspace selection, and other Multiview Contrast
augmentations are pretraining-only and never run during embedding inference.

## Inference and outputs

The backend validates CPU/fp32 execution, model state, graph dimensions, relation count, identity
carrier, and node mapping, then runs:

```python
with torch.inference_mode():
    output = model(prepared.graph)
```

No additional pooling, scientific cast, autocast, manual seed, or output-mode-specific forward is
applied.

| Output mode | Arrays | Granularity |
| --- | --- | --- |
| `node` | `node_embedding (N,3072)` and `node_ids (N,)` | residue |
| `global` | `global_embedding (3072,)` | structure |
| `both` | all three arrays in canonical order | mixed |

`global_embedding` is `output["graph_feature"][0]` from the model's sum readout.
`node_embedding` is finite `float32` and maps row-for-row to unique biological `node_ids`.
`feature_keys` is empty because Eris does not invent names for embedding dimensions.

## Runtime dependencies

The supported v0.1 contract is:

| Dependency | Requirement |
| --- | --- |
| Python | `>=3.11,<3.15` |
| PyTorch | `>=2.11,<3` |
| device/precision | CPU/fp32 |

Numerical parity with the frozen reference embeddings is verified by
`tests/test_gearnet_native.py` when `ERIS_GEARNET_CHECKPOINT` points at the official checkpoint.

For a CPU-only environment, install PyTorch from its CPU index, then Eris:

```console
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "erislib[gearnet]"
```

The `gearnet` extra declares PyTorch but does not bundle a checkpoint or compiled GearNet
dependency. TorchDrug is not a production dependency.

Dependency probing is explicit and lazy. `import eris` and learned Registry discovery do not
import Torch. The probe checks distribution and import
availability, and an isolated import smoke; it does not resolve a checkpoint or run inference.

## Artifact identity and cache behavior

| Field | Value |
| --- | --- |
| `representation_name` | `gearnet_edge` |
| backend ID | `gearnet-edge` |
| implementation | `eris.core.learned.backends.gearnet.backend.GearNetEdgeBackend` |
| encoder/adapter version | `3` |
| model architecture | `GearNet-Edge` |

Artifact `parameters` records output mode and the frozen biological-selection and graph settings.
`model` records checkpoint identifier, revision, source, and SHA-256. The input block records
structure content hash, parser-confirmed format, chain, and model. Generation provenance records
requested and resolved execution.

Learned cache identity includes representation and adapter identity, preprocessing parameters,
input content and selection, parser hint, model/checkpoint identity, output mode, and effective
precision. It excludes structure and checkpoint paths, mirrors, device locator, and temporary
transport paths.

A valid cache hit refreshes only `input.source`. It preserves arrays, biological mappings,
generation execution, and checkpoint-source provenance and does not probe or instantiate the
backend.

## Frozen behavior

Changes to any of the following are scientific contract changes:

- checkpoint identity or strict loading;
- model architecture or weights;
- accepted residues, CA selection, or missing-CA failure policy;
- graph topology, relations, node features, or edge features;
- CPU/fp32 inference and output extraction;
- biological node mapping;
- artifact and cache identity.

Do not change these during routine refactoring or release hardening.

## Validation and release certification

Tracked tests cover dependency isolation, frozen preprocessing, native graph construction,
checkpoint loading, model behavior, learned artifacts, cache reuse, and stored reference data
under `tests/data/`. The native runtime has reproduced frozen reference arrays within
`rtol=1e-6` and `atol=2e-6` in the environment where those references were generated.

Release hardening must still certify:

- the official checkpoint hash and strict load in each supported release environment;
- frozen graph and embedding references on the final supported Python/PyTorch matrix;
- repeated CPU/fp32 determinism in that matrix;
- clean wheel installation, first-run acquisition, cache reuse, and offline inference.

Passing core tests or skipping checkpoint-dependent tests is not release-environment
certification.

## License and citation

- GearNet source: [MIT](https://github.com/DeepGraphLearning/GearNet/blob/main/LICENSE)
- TorchDrug source used as a semantic reference:
  [Apache-2.0](https://github.com/DeepGraphLearning/torchdrug/blob/master/LICENSE)
- checkpoint record: [CC BY 4.0](https://zenodo.org/records/7723075)

Scientific reference: Zhang et al., “Protein Representation Learning by Geometric Structure
Pretraining,” ICLR 2023, [arXiv:2203.06125](https://arxiv.org/abs/2203.06125).

The inspected upstream revisions were GearNet
`28bf75f5dfb6cd7b9bf4394d153a87f69bddf1a6` and TorchDrug
`6066fbd82360abb5f270cba1eca560af01b8cc90`. These identify the source used to freeze semantics,
not the commits used to train the checkpoint.
