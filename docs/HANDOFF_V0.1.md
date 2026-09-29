# Eris v0.1 feature-complete handoff

## 1. Handoff snapshot

- Development branch: `dev`.
- Feature implementation baseline when WU4 documentation began:
  `1c0c36b4d94ed69e28b551882495591c43128601`.
- Project name: Eris.
- Current distribution metadata name: `eris-lib`.
- Python import package and CLI command: `eris`.
- Version declared in `pyproject.toml`: `0.1.0`.
- Python range declared in `pyproject.toml`: `>=3.11,<3.13`.

This SHA is the pre-WU4 implementation baseline, not the final handoff commit. The commit produced
after this documentation work is reviewed and committed becomes the Fernández feature-complete
candidate.

Release policy is not fully reconciled. Existing policy documents refer to distribution name
`eris` and broader Python ranges, while current packaging metadata uses `eris-lib` and
`>=3.11,<3.13`. WU4 records that disagreement without choosing a final release policy. Diego
Álvarez owns its resolution during release hardening.

## 2. What Eris does

Eris turns protein three-dimensional structures into deterministic numerical representations for
machine learning. Version 0.1 accepts PDB and mmCIF inputs, provides ten explicit atom/residue
representations, persists them as strict Schema v2 artifacts, and supplies batch, cache, dataset,
Torch, and PyG workflows. A separate native GearNet-Edge backend produces learned residue and
structure embeddings from a verified external checkpoint.

Eris does not generate sequences, run arbitrary external embedding models, infer missing
scientific identity, or download model checkpoints automatically.

## 3. Architecture

The feature-bearing explicit residue path is:

```text
structure input
    -> parser-confirmed model and chain selection
    -> biological residue selection
    -> residue position
    -> FeatureMode / FeatureSpec registry
    -> node_attr + node_attr_keys
    -> explicit feature-bearing representation
    -> StructureArtifact
    -> cache / batch / pack / dataset / export
```

The main implementation layers are:

- `src/eris/core/io.py`: structure loading, chain/model selection, and atom/residue extraction;
- `src/eris/core/config.py` and `src/eris/core/registry.py`: canonical configuration and explicit
  representation registration;
- `src/eris/features/`: deterministic node, edge, grid, and graph-statistic features;
- `src/eris/encoders/`: explicit representation construction;
- `src/eris/core/artifacts.py`, `artifact_schema_v2.py`, and `cache.py`: validation, persistence,
  provenance, and reuse;
- `src/eris/pipelines/`, `datasets/`, and `export/`: batch, pack, dataset, Torch, and PyG flows;
- `src/eris/core/learned/`: lazy learned-backend registry and GearNet-Edge execution.

Typed canonical configuration is the authority for defaults, validation, serialization, and
fingerprints. Learned GearNet execution is separate from the explicit feature registry.
Position-only matrix representations branch after residue positioning, while the voxel grid uses
the registered grid-channel builder; both share the same Artifact and downstream workflow layers.

## 4. Implemented explicit representations

| Public name | Type | Granularity | Purpose |
| --- | --- | --- | --- |
| `atom_point_cloud` | cloud | atom | Atomic coordinates and node features |
| `residue_point_cloud` | cloud | residue | Residue coordinates and node features |
| `atom_knn_graph` | graph | atom | Atomic k-nearest-neighbor graph |
| `residue_knn_graph` | graph | residue | Residue k-nearest-neighbor graph |
| `residue_contact_graph` | graph | residue | Residue graph from a distance cutoff |
| `residue_distogram` | matrix | residue | Pairwise residue distances |
| `residue_distogram_bins` | matrix | residue | Discretized pairwise distances |
| `residue_contact_map` | matrix | residue | Binary residue contact map |
| `residue_contact_map_bins` | matrix | residue | Distance-binned contact map |
| `residue_voxel_grid` | grid | residue | Multi-channel three-dimensional grid |

The Registry key, encoder name, artifact representation name, CLI identifier, and documentation
identifier are the same. Canonical defaults are listed in
[representation_defaults.md](representation_defaults.md).

## 5. Residue coordinate policies

Residue representations accept `residue_center` values:

- `ca`: the selected alpha-carbon coordinate;
- `mean`: the arithmetic mean of selected residue-atom coordinates;
- `com`: the atomic-mass-weighted center of mass of selected residue atoms.

The default is `ca`. The corresponding artifact `coordinate_source` values are `ca`,
`residue_mean`, and `residue_com`.

Version 0.1 selects standard amino-acid residues, uses the frozen `altloc="first"` policy, and
operates on one selected chain and model. If a requested center cannot be constructed,
`missing_residue_center="skip"` omits that residue and records it in `omitted_residue_ids`; the
explicit `"error"` policy fails with the affected residue IDs. Eris does not fabricate a center.

## 6. Residue features

The residue feature registry currently provides:

- `onehot`: canonical amino-acid identity;
- `aa_index`: the canonical numeric amino-acid index;
- `physchem`: the fixed Eris physicochemical descriptor table;
- `aaindex`: selected real AAIndex1 records;
- `plddt_ca`: explicitly sourced CA pLDDT;
- `bfactor_ca`: raw CA B-factor;
- `flags`: glycine, proline, unknown-residue, and CA-presence flags;
- `external`: a validated source-independent numerical matrix.

Features that accept parameters validate and canonicalize them at configuration construction.
Parameterless features reject unexpected parameters. Numerical feature blocks are emitted as
`float32` at the node-feature boundary.

Physical block order follows canonical `FeatureSpec.order`, not request-list order. Internal
column order follows the selected feature's canonical contract. `node_attr_keys` records the exact
resulting column semantics.

## 7. AAIndex

The public `aaindex` token uses AAIndex1 records from the installed `aaindex` package without
representation-time network access.

Users supply one or more explicit accession IDs. IDs are canonicalized, duplicates and unknown
accessions fail, and requested accession order defines the block's column order. Public keys are
`aaindex.<ACCESSION>`.

The canonical parameters include:

- `ids`;
- `normalization="none"`;
- `unknown`: `nan`, `zero`, or `error` for unknown residue `X`;
- `missing`: `nan`, `zero`, or `error` for unavailable/non-finite record values;
- provider name, provider version, and `AAIndex1` source identity.

Provider identity/version participate in serialized scientific configuration and therefore in
configuration/cache identity. No normalization or scientific value replacement is implicit.

## 8. B-factor and pLDDT

`bfactor_ca` is the raw CA `b_iso` value. It is not clipped, divided by 100, or interpreted from
its numeric range.

`plddt_source="none"` is the generic default and produces no inferred pLDDT. Set
`plddt_source="bfactor"` only when the input B-factor field is explicitly known to carry pLDDT.
That interpretation is serialized in canonical configuration. Finite interpreted values outside
0–100 fail instead of clipping.

The `plddt_ca` feature defaults to `normalization="none"`, preserving the 0–100 scale. The
explicit `normalization="unit_interval"` option divides by 100. Missing-value behavior follows the
feature-builder policy: unavailable extracted pLDDT remains `NaN`, normal explicit encoders fill a
requested `plddt_ca` feature with the builder's default zero policy, and direct builder use may
request `nan`. Generic residue and voxel defaults do not request pLDDT.

## 9. External residue features

The `external` feature consumes a generic `.npz` archive. Required fields are `values (N,D)`,
`residue_ids (N,)`, `source_name`, `chain_id`, and `model_index`; `feature_keys`, `source_version`,
and `source_model` are optional. Values become contiguous `float32` and object/pickle payloads are
forbidden.

Residue IDs use `<chain>:<number>:<insertion>:<resname>`. Rows may be shuffled, but alignment is
only by exact canonical identity. Missing, extra, malformed, or duplicate IDs and chain/model
mismatches fail. Eris does not align sequences, pad, truncate, or match by position.

Scientific identity uses a canonical SHA-256 over values and scientific metadata. It includes
source metadata, keys, dimensions, chain/model, and alignment policy. The invocation path is
excluded from scientific/cache identity and retained in the artifact's
`extensions.org.eris.external_features` provenance. The source is revalidated during resolution,
encoding, and cache reuse so mutation after resolution fails.

See [external_residue_features.md](external_residue_features.md) for the complete schema and API
examples. There is no direct Sylphy, Roxy, Ruddy, or model-library integration.

## 10. Hybrid representations

Built-in and external residue blocks use the same Registry pipeline. A request such as
`onehot + physchem + aaindex + bfactor_ca + external` produces one contiguous:

```text
node_attr (N, F)
```

and one exact ordered `node_attr_keys` list. It does not create a separate external embedding
field or second hybrid encoder. External widths such as 512 and 1024 were validated without a
fixed feature-width limit, padding, or truncation.

## 11. Edge and graph descriptors

The default explicit graph edge block is `dist + rbf + seq + local`:

- raw edge distance;
- 16 radial-basis-function channels over the configured distance range;
- biological sequence separation;
- a deterministic local-sequence flag.

Explicit graph `graph_attr` is a stable vector of deterministic graph statistics: node, edge, and
self-loop counts; simple-undirected density; degree summaries; edge-distance summaries; and
isolated-node fraction. Keys are stored in `graph_attr_keys`.

`graph_attr` is not a learned graph embedding. GearNet-Edge `global_embedding` is a learned
structure-level representation.

## 12. GearNet-Edge

`gearnet_edge` is the single registered v0.1 learned backend. It uses native Eris PyTorch code,
frozen preprocessing, the official Multiview Contrast checkpoint identity, CPU/fp32 execution,
and lazy optional dependencies. The checkpoint is neither bundled nor downloaded automatically.

Supported output modes are:

- `node`: `(N, 3072)` residue embeddings plus `node_ids`;
- `global`: one `(3072,)` structure embedding;
- `both`: both outputs.

Frozen graph and embedding references remain in `tests/data/`. Final certification against the
supported release environment and checkpoint remains Álvarez release-validation work; WU1–WU4 do
not alter GearNet scientific behavior.

## 13. Artifact, provenance, and cache

Every finalized representation uses `StructureArtifact` Schema v2:

```text
artifact/
|-- meta.json
`-- data.npz
```

Metadata records representation identity, encoder/model identity, selected input, source SHA-256,
canonical scientific parameters, configuration fingerprint, output declarations, exact feature
keys, shapes, dtypes, counts, and optional namespaced extensions.

The active explicit cache identity combines Schema/cache contract versions, representation and
encoder implementation/version, configuration fingerprint, input bytes, parser hint, and
chain/model selection. External scientific parameters contain the external content hash and
metadata but not the invocation path. Mutable external resources are verified before reuse.

After an artifact is finalized and saved, its arrays and metadata are self-contained. Loading,
packing, dataset access, and Torch/PyG export do not require the original structure or external
`.npz` file.

## 14. Batch, pack, and dataset

`eris.batch_encode()` processes rows with canonical manifest fields:

```text
id
path
chain
model
```

The Python API's `params_by_id` mapping supplies shallow top-level overrides for individual row
IDs. A row value replaces the corresponding global `params` value; nested dictionaries are not
recursively merged. Every row configuration is resolved and validated before batch execution
starts.

Resume identity is derived from the same resolved configuration passed to `eris.encode`. It checks
scientific parameters, input SHA-256, representation identity, and actual registered encoder
implementation/version, so an artifact from a stale encoder cannot be resumed.

The CLI currently accepts global `--params` only. `eris.pack()` validates artifacts and creates a
metadata index. `ErisFolderDataset` consumes that index without rescanning the artifact tree, and
family-specific collates reject incompatible widths or layouts instead of padding or truncating.

## 15. Torch and PyG

`eris.to_torch()` supports graph, cloud, matrix, grid, and learned artifacts. `eris.to_pyg()`
supports graph artifacts with public `edge_index` topology and maps `node_attr` to `x`; it never
infers topology or reconstructs GearNet internals.

High-dimensional hybrid node features and the PyG adapter contract were validated during WU3.
Native `torch_geometric` is not installed in the current handoff environment, so native PyG
runtime certification remains Álvarez release-validation scope.

## 16. Validation evidence

WU4 validation was run in the available `eris` Conda environment on Python 3.12. The latest
results are:

```text
focused/regression evidence from WU3: 369 passed
full suite: 434 passed, 5 skipped
Ruff: passed
format check (tracked source and WU4 documentation): passed
pip check: passed
lazy import check: passed
git diff --check: passed
```

The full working-tree suite includes the intentionally untracked PDB/mmCIF equivalence test. The
tracked repository contains 18 test modules; `tests/test_residue_features.py` permanently protects
WU1 semantics. Some focused WU2/WU3 validation cases were intentionally temporary and are not
claimed as tracked tests. The recorded temporary coverage included external alignment/failures,
source mutation, path relocation, high-dimensional hybrid flow, per-row batch parameters, resume
identity, artifact independence, and adapter behavior.

Five skips reflect unavailable optional runtime/checkpoint conditions, not observed scientific
test failures. A whole-worktree format scan also sees untracked task-specification documents that
are deliberately excluded from this handoff commit; their code-fence formatting is not part of the
candidate validation.

## 17. Known limitations

- A generic pooled residue representation is not implemented; this remains a non-blocking P1.
- Native `torch_geometric` runtime certification is pending.
- Final GearNet supported-environment/checkpoint frozen-reference certification is pending.
- Row-specific batch overrides are available only through the Python API.
- External features require exact residue identities; there is no automatic sequence alignment.
- There is no direct Sylphy, Roxy, or Ruddy integration.
- No generic GNN, Graph2Vec, GIN, GAT, or GraphSAGE family is included.
- Model checkpoints and external feature payloads are never downloaded automatically.
- Release distribution naming and the final Python support matrix are unresolved.

## 18. Álvarez release and hardening scope

The following work is intentionally outside Fernández's scientific implementation closure:

- reconcile the public distribution name (`eris-lib` versus policy references to `eris`);
- reconcile and certify the final Python support matrix;
- repair and run the complete CI matrix;
- freeze dependency policy and the final public API;
- build and validate wheel/sdist artifacts;
- validate a clean install from the built wheel;
- certify native PyG in the release environment;
- certify GearNet frozen-reference parity in its supported environment;
- add or finalize `CITATION.cff` and changelog/release metadata;
- configure Trusted Publishing and publish to PyPI;
- create the GitHub Release and Zenodo record.

These tasks should not be interpreted as missing Fernández feature implementation. A concrete
reproducible scientific or workflow defect discovered during hardening should be reported against
the frozen candidate rather than hidden by a release-policy change.

## 19. Nicole and Julián independent testing

Use the environment supplied by Álvarez and test from the public user boundaries:

1. Install the built candidate cleanly and confirm `import eris`, `eris --version`, and discovery.
2. Encode representative PDB and mmCIF structures with explicit chain/model selection.
3. Compare residue `ca`, `mean`, and `com` coordinate policies, including a missing-center case.
4. Inspect residue point-cloud, residue kNN, and residue contact-graph artifacts.
5. Inspect contact maps, distograms, binned matrices, and a voxel grid.
6. Request known AAIndex1 accessions and verify values, keys, unknown, and missing policies.
7. Supply shuffled external rows and verify exact realignment; then confirm missing, extra,
   duplicate, wrong-chain, wrong-model, and mutated-source failures.
8. Encode a hybrid graph combining one-hot, physicochemical, AAIndex, and external columns; inspect
   `node_attr`, `node_attr_keys`, edge features, and `graph_attr`.
9. Save/load, pack, dataset-load, and Torch/PyG-export representative artifacts after removing or
   relocating the original external `.npz`.
10. Exercise CLI encode/embed/info/export/batch paths, remembering that per-row batch parameters
    are a Python API feature.
11. Run GearNet `node`, `global`, and `both` modes with the verified checkpoint and inspect residue
    identity, shapes, provenance, and frozen-reference agreement.

Unexpected biological-selection or structural-semantic differences should be escalated rather
than normalized or repaired implicitly.

## 20. Handoff criterion

Fernández's Eris v0.1 feature implementation is complete at the candidate produced by committing
WU4 on top of the implementation baseline named above. Further scientific code changes should be
made only for concrete reproducible defects found during independent or release validation.

Documentation completion does not itself certify publication readiness. Álvarez's release and
hardening gates remain mandatory before any package publication, tag, GitHub Release, or Zenodo
deposit.
