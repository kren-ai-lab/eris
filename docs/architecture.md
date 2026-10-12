# Eris architecture

Eris has two representation families:

- explicit encoders compute deterministic arrays from selected structure coordinates;
- learned backends compute embeddings through an optional model runtime.

Both families return `StructureArtifact` objects and use the artifact schema for persistence.

## Code map

| Path | Responsibility |
| --- | --- |
| `eris/api.py` | Public `encode()` and `embed()` entry points |
| `eris/discovery.py` | Public representation and embedder discovery |
| `eris/core/config.py` | Typed explicit-representation configs and scientific defaults |
| `eris/core/registry.py` | Explicit encoder registration and config resolution |
| `eris/core/io.py` | PDB/mmCIF loading, model/chain selection, atom and residue extraction |
| `eris/core/base_encoder/` | Shared explicit encoding, finalization, and cache flow |
| `eris/core/artifacts.py` | `StructureArtifact` validation, save, and load |
| `eris/core/artifact_schema.py` | Executable artifact metadata and array contract |
| `eris/core/cache.py` | Path-independent artifact cache identity and storage |
| `eris/features/` | Node, edge, voxel, and graph-statistic feature definitions |
| `eris/pooling.py` | Residue-feature pooling and source-artifact identity |
| `eris/encoders/` | Explicit cloud, graph, matrix, and grid encoders |
| `eris/core/learned/` | Learned registry, execution, checkpoints, runner, and artifacts |
| `eris/core/learned/backends/gearnet/` | Native GearNet-Edge preprocessing and inference |
| `eris/pipelines/` | Manifest, batch encode/embed, export, and artifact packing |
| `eris/datasets/` | Packed-index dataset and family-specific collates |
| `eris/export/` | Torch conversion and graph-only PyG conversion |
| `eris/cli/` | Typer commands that delegate to public Python operations |

## Explicit encoding flow

```text
eris.encode(...)
    -> resolve typed config from the explicit Registry
    -> load PDB/mmCIF and resolve the model-order canonical chain set
    -> extract atom or residue data
    -> compute coordinates and registered features
    -> construct representation arrays
    -> finalize and strictly validate artifact metadata
    -> optionally cache and/or save the StructureArtifact
```

`CanonicalRepresentationConfig` subclasses are the source of truth for defaults, validation,
serialization, discovery metadata, and configuration fingerprints. A public representation name
must match the Registry key, encoder `name`, artifact `representation_name`, CLI identifier,
and documentation.

Residue representations share selection and positioning in `BaseResidueEncoder`.
`chain_id=None` selects every relevant protein/polymer chain in the selected model; a string
selects one chain and a sequence selects an explicit set. Empty, duplicate, and unknown
selections fail. Caller order does not affect node order, fingerprints, or cache identity.
Explicit atom and residue artifacts carry aligned `chain_id` arrays. No biological assembly is
generated.
`residue_center` maps `ca`, `mean`, and `com` to artifact coordinate sources `ca`,
`residue_mean`, and `residue_com`. Missing centers follow the configured `skip` or `error`
policy; skipped identities remain in `omitted_residue_ids`.

Feature-bearing residue clouds and graphs pass the extracted bundle through the residue feature
registry. `FeatureMode` validates requested tokens and parameters; physical block order follows
`FeatureSpec.order`. The encoder stores the concatenated `node_attr` and authoritative ordered
`node_attr_keys`. The shared residue bundle also retains aligned selected `N`, `CA`, and `C`
coordinates and validity masks. `backbone_geometry` uses those arrays with same-chain structural
adjacency and the physical C--N cutoff to emit phi, psi, and omega without crossing breaks.
`sidechain_geometry` uses the centralized standard residue-specific atom definitions and the
shared signed-dihedral implementation to emit a fixed chi1-through-chi4 schema; undefined values
remain `NaN`.
The opt-in `secondary_structure` token lazily runs Biotite P-SEA independently on those physical
segments. The opt-in `sasa` and `rsa` tokens share a lazily loaded Biotite Shrake-Rupley
calculation over the selected chains' polymer heavy atoms; RSA applies the frozen Tien maximum-ASA
table without clipping. Provider identity, algorithms, context, mappings, and normalization
references live in canonical feature parameters.
Residue kNN and contact graphs can use the edge registry to add exact same/inter-chain relations
and directed relative displacement in an internal right-handed N-CA-C source frame. Invalid
source frames yield `NaN` direction components without removing nodes or changing topology; the
frames themselves are not public node attributes.
Opt-in paired interface tokens derive residue flags and existing-edge annotations from selected
inter-chain polymer-heavy-atom contacts. Their auxiliary residue-ID subset preserves canonical
node order and does not alter topology.
Matrix encoders use positions without fabricating node features. The voxel encoder uses the grid
feature registry.

`eris.pool()` is a separate derived-artifact operation. It validates one explicitly named
residue-aligned feature matrix, computes a configured column-wise reduction, and returns a new
structure-level `residue_pool` vector artifact. Its source-artifact fingerprint covers source
identity, ordered keys, dtype, shape, and canonical selected-array bytes; the input artifact is
not mutated.

## Learned encoding flow

```text
eris.embed(...)
    -> resolve learned metadata and output mode
    -> resolve CPU/fp32 execution
    -> verify the caller-supplied checkpoint
    -> apply frozen GearNet biological selection and graph construction
    -> run the native PyTorch model
    -> build and strictly validate a learned artifact
    -> optionally cache and/or save it
```

The learned Registry stores static metadata and lazy factories. Importing `eris`, listing
embedders, or loading an existing artifact must not import Torch or require a checkpoint.
GearNet-specific behavior is isolated under
`eris/core/learned/backends/gearnet/` and frozen by
[the backend contract](gearnet_edge_backend.md).
GearNet-Edge is single-chain: resolving more than one selected chain fails clearly before
checkpoint inference.

## Public boundaries

The primary Python entry points are:

- `eris.encode(...)` and `eris.embed(...)`;
- `eris.pool(...)`;
- `eris.list_representations()` and `eris.get_representation_info(...)`;
- `eris.list_embedders()` and `eris.get_embedder_info(...)`;
- `eris.batch_encode(...)` and `eris.batch_embed(...)`;
- `eris.pack(...)`, `eris.to_torch(...)`, and `eris.to_pyg(...)`.

The CLI exposes `encode`, `embed`, `pack`, `batch`, `export`, `info`, `cache`,
`models`, `representations`, and `embedders`. CLI handlers parse command-line values and
delegate to the corresponding Python operation.

Batch structure rows contain `id`, `path`, `chain`, and `model`. Python
`chain` accepts `null`, one string, or a non-empty list of unique strings; JSON/JSONL preserve
lists directly and CSV stores a JSON list in the field. Python
`batch_encode(..., params_by_id=...)` applies row-specific shallow top-level overrides to global
`params`; the CLI accepts global parameters only. `pack()` writes a metadata index consumed by
`ErisFolderDataset`.

`to_torch()` supports explicit graphs, clouds, matrices, grids, pooled vectors, and learned
embeddings.
`to_pyg()` accepts only explicit graph artifacts with a public `edge_index`; it never infers
topology or reconstructs GearNet internals.

## Adding an explicit representation

1. Add a typed config in `eris/core/config.py`.
2. Implement the encoder under the appropriate `eris/encoders/` package using the shared
   base encoder and output declarations.
3. Register it in the encoder module and import that module through `eris/encoders/__init__.py`.
4. Preserve the name invariant and finalize through the existing artifact schema path.
5. Add focused encoder, discovery, artifact, CLI, and downstream tests as required.
6. Document its scientific defaults in `representation_defaults.md`.

Do not create a second configuration, serialization, hashing, or artifact mechanism.

## Adding a residue feature

1. Implement the value builder, key builder, parameter normalizer, and any fingerprint projection
   in `eris/features/node/`.
2. Register one `FeatureSpec` in `RESIDUE_NODE_REGISTRY` with a stable physical order.
3. Route all scientific parameters through `FeatureMode`; keep invocation-only locators out of
   the scientific fingerprint.
4. Preserve row alignment, deterministic keys, and `float32` node-feature output.
5. Add focused feature, config, artifact, cache, batch, and export tests as applicable.

External producers must also satisfy [the external NPZ contract](external_residue_features.md).

## Contracts that extensions must preserve

The contracts for polymer continuity, backbone and side-chain torsions, local frames,
secondary structure, solvent accessibility, interfaces, and structure-level pooling are frozen in
[`structural_features.md`](structural_features.md). Polymer continuity and backbone torsions are
implemented, as are local frames, relative edge direction, secondary structure, SASA/RSA,
inter-chain interface features, structure-level pooling, and side-chain torsions.

Those features must follow the same architecture described above. Public APIs
construct canonical configuration, registries resolve implementations, runners
execute them, and immutable artifacts carry the results. Optional scientific
providers stay lazy and feature-local. No derived-feature implementation may
bypass the canonical validators, serializers, fingerprints, or cache identity,
and no new registry, runner, backend, or provider object becomes public API.

- Optional dependencies remain lazy and feature-local.
- The artifact schema is `meta.json` plus `data.npz`; `feature_keys` is authoritative.
- Scientific parameters and invocation provenance remain distinct.
- Cache identity remains based on scientific content, config, implementation, and input selection,
  not local source paths.
- Biological identity, row order, array order, and exact keys are never inferred or repaired.
- Incompatible inputs fail instead of being padded, truncated, normalized, or silently omitted.

See [Artifact Schema](artifact_schema.md) and
[representation defaults](representation_defaults.md) for the executable contract details.
