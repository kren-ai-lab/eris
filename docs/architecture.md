# Eris architecture

Eris has two representation families:

- explicit encoders compute deterministic arrays from selected structure coordinates;
- learned backends compute embeddings through an optional model runtime.

Both families return `StructureArtifact` objects and use Artifact Schema v2 for persistence.

## Code map

| Path | Responsibility |
| --- | --- |
| `src/eris/api.py` | Public `encode()` and `embed()` entry points |
| `src/eris/discovery.py` | Public representation and embedder discovery |
| `src/eris/core/config.py` | Typed explicit-representation configs and scientific defaults |
| `src/eris/core/registry.py` | Explicit encoder registration and config resolution |
| `src/eris/core/io.py` | PDB/mmCIF loading, model/chain selection, atom and residue extraction |
| `src/eris/core/base_encoder/` | Shared explicit encoding, finalization, and cache flow |
| `src/eris/core/artifacts.py` | `StructureArtifact` validation, save, and load |
| `src/eris/core/artifact_schema_v2.py` | Executable Schema v2 metadata and array contract |
| `src/eris/core/cache.py` | Path-independent artifact cache identity and storage |
| `src/eris/features/` | Node, edge, voxel, and graph-statistic feature definitions |
| `src/eris/encoders/` | Explicit cloud, graph, matrix, and grid encoders |
| `src/eris/core/learned/` | Learned registry, execution, checkpoints, runner, and artifacts |
| `src/eris/core/learned/backends/gearnet/` | Native GearNet-Edge preprocessing and inference |
| `src/eris/pipelines/` | Manifest, batch encode/embed, export, and artifact packing |
| `src/eris/datasets/` | Packed-index dataset and family-specific collates |
| `src/eris/export/` | Torch conversion and graph-only PyG conversion |
| `src/eris/cli/` | Typer commands that delegate to public Python operations |

## Explicit encoding flow

```text
eris.encode(...)
    -> resolve typed config from the explicit Registry
    -> load PDB/mmCIF and select model/chain
    -> extract atom or residue data
    -> compute coordinates and registered features
    -> construct representation arrays
    -> finalize and strictly validate Schema v2 metadata
    -> optionally cache and/or save the StructureArtifact
```

`CanonicalRepresentationConfig` subclasses are the source of truth for defaults, validation,
serialization, discovery metadata, and configuration fingerprints. A public representation name
must match the Registry key, encoder `name`, artifact `representation_name`, CLI identifier,
and documentation.

Residue representations share selection and positioning in `BaseResidueEncoder`.
`residue_center` maps `ca`, `mean`, and `com` to artifact coordinate sources `ca`,
`residue_mean`, and `residue_com`. Missing centers follow the configured `skip` or `error`
policy; skipped identities remain in `omitted_residue_ids`.

Feature-bearing residue clouds and graphs pass the extracted bundle through the residue feature
registry. `FeatureMode` validates requested tokens and parameters; physical block order follows
`FeatureSpec.order`. The encoder stores the concatenated `node_attr` and authoritative ordered
`node_attr_keys`. Matrix encoders use positions without fabricating node features. The voxel
encoder uses the grid feature registry.

## Learned encoding flow

```text
eris.embed(...)
    -> resolve learned metadata and output mode
    -> resolve CPU/fp32 execution
    -> verify the caller-supplied checkpoint
    -> apply frozen GearNet biological selection and graph construction
    -> run the native PyTorch model
    -> build and strictly validate a learned Schema v2 artifact
    -> optionally cache and/or save it
```

The learned Registry stores static metadata and lazy factories. Importing `eris`, listing
embedders, or loading an existing artifact must not import Torch or require a checkpoint.
GearNet-specific behavior is isolated under
`src/eris/core/learned/backends/gearnet/` and frozen by
[the backend contract](gearnet_edge_backend.md).

## Public boundaries

The primary Python entry points are:

- `eris.encode(...)` and `eris.embed(...)`;
- `eris.list_representations()` and `eris.get_representation_info(...)`;
- `eris.list_embedders()` and `eris.get_embedder_info(...)`;
- `eris.batch_encode(...)`, `eris.batch_embed(...)`, and `eris.batch_export(...)`;
- `eris.pack(...)`, `eris.to_torch(...)`, and `eris.to_pyg(...)`.

The CLI exposes `encode`, `embed`, `pack`, `batch`, `export`, `info`, `cache`,
`models`, `representations`, and `embedders`. CLI handlers parse command-line values and
delegate to the corresponding Python operation.

Batch structure rows contain `id`, `path`, `chain`, and `model`. Python
`batch_encode(..., params_by_id=...)` applies row-specific shallow top-level overrides to global
`params`; the CLI accepts global parameters only. `pack()` writes a metadata index consumed by
`ErisFolderDataset`.

`to_torch()` supports explicit graphs, clouds, matrices, grids, and learned embeddings.
`to_pyg()` accepts only explicit graph artifacts with a public `edge_index`; it never infers
topology or reconstructs GearNet internals.

## Adding an explicit representation

1. Add a typed config in `src/eris/core/config.py`.
2. Implement the encoder under the appropriate `src/eris/encoders/` package using the shared
   base encoder and output declarations.
3. Register it in the encoder module and import that module through `src/eris/encoders/__init__.py`.
4. Preserve the name invariant and finalize through the existing Schema v2 path.
5. Add focused encoder, discovery, artifact, CLI, and downstream tests as required.
6. Document its scientific defaults in `representation_defaults.md`.

Do not create a second configuration, serialization, hashing, or artifact mechanism.

## Adding a residue feature

1. Implement the value builder, key builder, parameter normalizer, and any fingerprint projection
   in `src/eris/features/node/`.
2. Register one `FeatureSpec` in `RESIDUE_NODE_REGISTRY` with a stable physical order.
3. Route all scientific parameters through `FeatureMode`; keep invocation-only locators out of
   the scientific fingerprint.
4. Preserve row alignment, deterministic keys, and `float32` node-feature output.
5. Add focused feature, config, artifact, cache, batch, and export tests as applicable.

External producers must also satisfy [the external NPZ contract](external_residue_features.md).

## Contracts that extensions must preserve

- Optional dependencies remain lazy and feature-local.
- Schema v2 remains `meta.json` plus `data.npz`; `feature_keys` is authoritative.
- Scientific parameters and invocation provenance remain distinct.
- Cache identity remains based on scientific content, config, implementation, and input selection,
  not local source paths.
- Biological identity, row order, array order, and exact keys are never inferred or repaired.
- Incompatible inputs fail instead of being padded, truncated, normalized, or silently omitted.

See [Artifact Schema v2](artifact_schema.md) and
[representation defaults](representation_defaults.md) for the executable contract details.
