# Eris v0.1 Architecture

This document records the frozen conceptual architecture and implemented public boundaries for
Eris v0.1. Historical rationale is identified where it remains useful; the API, CLI, artifact,
and learned-backend sections below describe current behavior unless noted otherwise.

## Purpose and framework boundaries

Eris converts protein three-dimensional structures into reproducible numerical structural
representations. It contains two first-class representation families:

- `explicit`;
- `learned`.

The surrounding framework boundaries are:

- **Sylphy:** learned sequence or molecular encoders and embeddings.
- **Roxy:** deterministic engineered descriptors.
- **Eris:** numerical representations derived specifically from 3D structure.
- **Rudeus:** exploratory analysis and visualization.

Eris must not become a duplicate of Sylphy, Roxy, or Rudeus.

## v0.1 scope

The ten explicit representations already present in the codebase are the complete v0.1 explicit
surface; no new representation family is part of this release.

GearNet-Edge pretrained with Multiview Contrast is the selected and registered learned backend for
v0.1. Its scientific architecture, checkpoint, preprocessing, output contract, and validated
production lifecycle are frozen in `docs/gearnet_edge_backend.md`. The backend implements frozen
preprocessing, verified checkpoint loading, deterministic CPU/fp32 inference, and complete
Artifact v2 provenance through the single native Eris runtime.

## Canonical public representation identifiers

The v0.1 public identifiers are frozen as:

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

The canonical invariant is:

```text
Registry key == Encoder.name == artifact representation_name == CLI identifier == documentation identifier
```

Historical or pre-release identifiers such as `distogram_ca`, `distogram_residue`,
`contact_map_ca`, `voxel_grid_ca`, and `_ns` variants do not define the public v0.1 API. No
compatibility aliases are required unless a real consumer is later identified.

## Structural terminology

`granularity` describes the represented biological entity. Allowed v0.1 values are:

- `atom`;
- `residue`;
- `structure`;
- `mixed`.

`ca` is not a granularity. `coordinate_source` describes how coordinates are defined. Examples
include:

- `atom`;
- `ca`;
- `residue_mean`;
- `residue_com`;
- `backbone`;
- `all_atom`;
- `model_specific`.

A CA-based residue graph therefore has `granularity = residue` and
`coordinate_source = ca`.

## Canonical representation table

| name | family | type | granularity | default coordinate source |
| --- | --- | --- | --- | --- |
| `atom_point_cloud` | explicit | cloud | atom | atom |
| `residue_point_cloud` | explicit | cloud | residue | ca |
| `atom_knn_graph` | explicit | graph | atom | atom |
| `residue_knn_graph` | explicit | graph | residue | ca |
| `residue_contact_graph` | explicit | graph | residue | ca |
| `residue_distogram` | explicit | matrix | residue | ca |
| `residue_distogram_bins` | explicit | matrix | residue | ca |
| `residue_contact_map` | explicit | matrix | residue | ca |
| `residue_contact_map_bins` | explicit | matrix | residue | ca |
| `residue_voxel_grid` | explicit | grid | residue | ca |

## Configuration authority

The target architecture uses typed configuration, dataclasses, or an equivalent typed
definition as the single source of truth. The same definition must drive:

- runtime defaults;
- Registry metadata;
- CLI defaults and help;
- artifact parameters;
- fingerprints;
- documentation.

The canonical typed configuration is the implemented authority; Registry metadata, defaults,
help, artifact parameters, and fingerprints derive from it.

## Explicit residue data flow

Feature-bearing explicit residue representations share one canonical processing path:

```text
Structure input
    ↓
biological residue selection
    ↓
position (CA, residue-atom mean, or center of mass)
    ↓
FeatureMode / FeatureSpec registry
    ↓
node_attr + node_attr_keys
    ↓
explicit feature-bearing representation
    ↓
StructureArtifact
    ↓
cache / batch / pack / dataset / export
```

The residue feature registry composes five semantic categories without introducing a separate
hybrid encoder:

- residue identity (`onehot`, `aa_index`, and flags);
- fixed physicochemical descriptors (`physchem`);
- selected true AAIndex1 records (`aaindex`);
- experimental or confidence metadata (`bfactor_ca` and explicitly sourced `plddt_ca`);
- source-independent external numerical features (`external`).

Selected feature blocks are concatenated into one `node_attr` matrix. Physical block order follows
canonical `FeatureSpec.order`; the order in which a caller writes tokens does not redefine the
matrix layout. Within each block, internal columns follow that feature's canonical contract: for
example, requested AAIndex accessions retain their canonicalized order and external columns follow
the payload's validated `feature_keys`. The ordered `node_attr_keys` metadata is authoritative for
the resulting columns.

GearNet-Edge does not use this explicit node-feature composition path. It remains a separate
learned structural backend with its own frozen preprocessing, model, checkpoint, and embedding
contract.

Position-only matrix representations branch after residue positioning and do not fabricate a
`node_attr` payload. The residue voxel grid applies its own registered grid-channel composition to
the selected residue bundle. Both still finalize through the same Artifact and downstream
workflow boundaries.

## Artifact Schema conceptual contract

Canonical disk persistence remains:

- `data.npz`;
- `meta.json`.

The current Artifact Schema is the stable public artifact contract. Serialized artifacts retain
`schema_version = 2`.

The conceptual top-level fields are:

- `schema_version`;
- `representation_name`;
- `representation_family`;
- `representation_type`;
- `granularity`;
- `coordinate_source`;
- `encoder` or `model`;
- `input`;
- `parameters`;
- `provenance`;
- `outputs`;
- `feature_keys`;
- `shapes`;
- `dtypes`;
- `counts`;
- optional `extensions`.

Explicit representations require `encoder` and normally do not use `model`. Learned
representations require `model`.

### Input metadata

The minimum input metadata is:

- `source`;
- `source_sha256`;
- `format`;
- `model_index`;
- `chain_id`.

### Feature semantics

`feature_keys` is authoritative. A representation creates its ordered semantic feature
definitions, and artifact finalization validates them. Finalization must never silently replace
or re-infer authoritative feature semantics.

`feature_keys` is reserved for feature dimensions such as:

- `node_attr_keys`;
- `edge_attr_keys`;
- `graph_attr_keys`;
- `grid_channels`.

Output names and layout semantics belong in `outputs`, not in `feature_keys`.

### Learned canonical arrays

The conceptual learned-representation keys are:

- `node_embedding`;
- `global_embedding`;
- `node_ids`;
- optional `valid_mask`.

Every local embedding row must map unambiguously to a biological entity.

### Learned runtime infrastructure

The internal learned runtime is implemented around a separate lazy learned Registry, backend
execution-capability resolution, verified checkpoint cache, `StructuralEmbedderRunner`, learned
Artifact Schema builder, and the shared artifact cache. Backends return their prepared model
input together with the parser-confirmed structure format; the runner preserves biological node
mapping, generation execution provenance, and model/checkpoint identity.

Artifact reuse is independent of backend and checkpoint availability after generation. Precision
participates in learned artifact-cache identity; device, checkpoint source locator, and structure
source path do not. A cache hit preserves generation execution and model-source provenance while
refreshing only the current structure locator.

The production learned Registry contains exactly one v0.1 backend, `gearnet_edge`. Its static
metadata and configuration are registered during core import, while dependency probing, factory
construction, and all Torch imports remain lazy. The M4 fake backend exists only in
tests. `eris.embed` and the public learned CLI are implemented public M6 entrypoints over this
internal runtime.

### Provenance

Explicit artifacts must contain enough Eris package and environment information to reproduce
the representation. Learned artifacts additionally require model, backend, and checkpoint
provenance, including the checkpoint SHA-256.

Artifact Schema is implemented as the canonical active runtime artifact contract. Its detailed
executable contract is documented in `docs/artifact_schema.md`.

## Public Python API

The high-level v0.1 M6 public API is:

- `eris.encode(...)`
- `eris.embed(...)`
- `eris.list_representations()`
- `eris.get_representation_info(...)`
- `eris.list_embedders()`
- `eris.get_embedder_info(...)`

These high-level functions are the public user-facing contract. Low-level APIs may coexist for
internal implementation and tests, but users should not need Registry classes, encoder classes,
learned runners, backend descriptors, or cache classes for README-level workflows.

## Public CLI

The M6 public CLI hierarchy is:

- `eris --version`
- `eris representations ...`
- `eris embedders ...`
- `eris encode ...`
- `eris embed ...`
- `eris batch encode ...`
- `eris batch embed ...`
- `eris pack ...`
- `eris export to-torch ...`
- `eris export to-pyg ...`
- `eris models ...`
- `eris cache ...`
- `eris info`

Batch, pack, dataset, and downstream export workflows are implemented M7 public boundaries.

## Batch and pack contract

The minimum structure manifest fields are:

- `id`;
- `path`;
- `chain`;
- `model`.

Programmatic `batch_encode(..., params_by_id=...)` may associate representation-parameter
overrides with these row IDs without extending the manifest schema. Overrides are shallow:
row-specific top-level values replace global `params` values. The batch CLI continues to accept
only global parameters.

The minimum batch task result fields are:

- `id`;
- `representation_name`;
- `representation_family`;
- `artifact_dir`;
- `status`;
- `error_type`;
- `error_message`.

The minimum packed index fields are:

- `id`;
- `artifact_dir`;
- `meta_path`;
- `representation_name`;
- `representation_family`;
- `representation_type`;
- `granularity`;
- `n_nodes`;
- `n_edges`;
- `shape_summary`.

Learned-specific fields may extend the index where applicable.

`ErisFolderDataset` consumes the packed index without rescanning the artifact root. Family-specific
collates (`collate_graph`, `collate_cloud`, `collate_matrix`, `collate_grid`, and
`collate_learned`) reject incompatible mixed tensor batches rather than padding, truncating, or
inventing topology. `to_torch(...)` supports graph, cloud, matrix, grid, and learned artifacts;
`to_pyg(...)` accepts only artifacts with public graph topology.

## Output policy

`out/` is generated execution output. It is not canonical test data, a golden reference, a
supported artifact corpus, or the source of truth for v0.1 schemas. It is ignored and excluded
from normal source/package content.

## Version authority

Eris has one package-version authority derived from distribution metadata and
exposed through `eris.__version__`. Artifacts record the resulting Eris version in provenance.
Model and checkpoint identity remains separate from the Eris package version.

Version handling is implemented by the distribution metadata authority described above.

## PyPI naming

The project identity is:

- project/repository: Eris;
- Python package/import: `eris`;
- CLI command: `eris`.

Current distribution metadata in `pyproject.toml` uses `eris-lib`, while existing project-policy
documents describe `eris` as the desired distribution name. This disagreement is intentionally
not resolved by the scientific implementation closeout. Final distribution-name reconciliation,
including PyPI namespace availability and ownership, belongs to release hardening. The import and
CLI names remain `eris` regardless of that decision.
