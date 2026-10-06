# Artifact Schema v2

Every persisted Eris representation is a directory containing:

```text
artifact/
|-- meta.json
`-- data.npz
```

`StructureArtifact` in `src/eris/core/artifacts.py` validates, saves, and loads this form.
`src/eris/core/artifact_schema_v2.py` is the executable schema authority. The serialized
`schema_version` is the integer `2`.

## Metadata

No undeclared top-level fields are accepted.

| Field | Contract |
| --- | --- |
| `schema_version` | exactly `2` |
| `representation_name` | non-empty public identifier |
| `representation_family` | `explicit` or `learned` |
| `representation_type` | `cloud`, `graph`, `matrix`, `grid`, or `vector` for explicit; `embedding` for learned |
| `granularity` | `atom`, `residue`, `structure`, or `mixed` |
| `coordinate_source` | `atom`, `ca`, `residue_mean`, `residue_com`, or `model_specific` |
| `encoder` | producer implementation and version |
| `input` | source content and selected model/chain |
| `parameters` | canonical scientific configuration |
| `provenance` | Eris version and configuration fingerprint |
| `outputs` | ordered declarations matching every NPZ array |
| `feature_keys` | authoritative ordered feature semantics |
| `shapes` | exact shape for every output |
| `dtypes` | exact NumPy dtype string for every output |
| `counts` | non-negative integer summaries |
| `model` | required for learned artifacts; forbidden for explicit artifacts |
| `extensions` | optional namespaced metadata |

An explicit atom artifact uses `coordinate_source="atom"`. An explicit residue artifact uses
`ca`, `residue_mean`, or `residue_com`. GearNet learned artifacts use `model_specific`.

## Producer and model identity

`encoder` is required for both families:

```json
{
  "implementation": "eris.encoders.graphs.residue_knn_graph.ResidueKNNGraph",
  "version": "0.1.4"
}
```

Learned artifacts also require:

```json
{
  "architecture": "GearNet-Edge",
  "checkpoint": {
    "identifier": "mc_gearnet_edge.pth",
    "revision": "Zenodo record v2",
    "source": "https://zenodo.org/records/7723075",
    "sha256": "6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec"
  }
}
```

`checkpoint.identifier` is required. Revision, source, and SHA-256 are optional at the generic
schema level.

## Input identity

The required `input` block is:

```json
{
  "source": "protein.pdb",
  "source_sha256": "64-lowercase-hex-characters",
  "format": "pdb",
  "model_index": 0,
  "chain_ids": ["A", "B"]
}
```

`source` is the invocation locator. `source_sha256` identifies the input bytes. `format` is the
parser-confirmed format, not an unchecked filename inference. `model_index` is an integer greater
than or equal to zero. `chain_ids` is a non-empty JSON list of unique strings in selected-model
chain order. Single-chain artifacts use a one-element list. Legacy scalar `chain_id` metadata is
rejected. Residue- and atom-aligned explicit artifacts also carry a Unicode `chain_id` array.

The only optional input keys are string-valued `original_filename`, `structure_id`, and
`source_database`.

## Scientific parameters and invocation provenance

`parameters` contains the canonical scientific configuration that determines representation
content. It accepts JSON mappings, lists, strings, Booleans, integers, finite floats, and `null`.
Mapping keys must be strings. NumPy values, arrays, `Path` objects, NaN, and infinities are
rejected.

Invocation details that do not alter scientific output do not belong in `parameters` or its
fingerprint. External residue features demonstrate this split:

- scientific parameters retain content hash, source metadata, feature keys and dimension,
  chain-set/model identity, and alignment policy;
- the local NPZ path is recorded under `extensions.org.eris.external_features`.

Do not omit values that change scientific output. Requested values remain canonical even when an
encoder computes an effective runtime value; for example, voxel `voxel_size=null` remains in
`parameters` while the effective spacing is stored as an output.

## Configuration fingerprint

`compute_config_fingerprint(representation_name, parameters, chain_ids=...)` hashes canonical
UTF-8 JSON for:

```json
{
  "representation_name": "...",
  "parameters": {},
  "input_selection": {"chain_ids": ["A", "B"]}
}
```

Canonical JSON uses sorted keys, compact separators, UTF-8 characters, and rejects NaN. The
fingerprint changes with the public representation name, scientific parameters, or canonical
resolved chain set. It excludes payload arrays, timestamps, environment, package/encoder
versions, source/model identity, and invocation-only paths.

The fingerprint is not a complete cache key.

## Outputs and arrays

`outputs` is a non-empty ordered list. Each declaration has a unique `name` and a non-empty
`role`; optional fields are `granularity`, `layout`, and `description`.

```json
{
  "name": "node_attr",
  "role": "node_features",
  "granularity": "residue",
  "layout": "NF"
}
```

Output names match `^[a-z][a-z0-9_]*$` and correspond exactly to keys in `data.npz`.
`shapes` and `dtypes` have the same key set. Physical ZIP member order is irrelevant; loaders
restore runtime array order from `outputs`.

Every value is a NumPy array. Allowed dtype kinds are signed integer, unsigned integer, float,
Boolean, and Unicode string. Object, structured, complex, datetime, timedelta, and byte-string
arrays are rejected. Loading always uses `allow_pickle=False`.

## Feature keys

`feature_keys` is always present and never inferred. It may contain:

- `node_attr_keys`;
- `edge_attr_keys`;
- `graph_attr_keys`;
- `grid_channels`; or
- `pooled_attr_keys`.

Each value is an ordered list of strings whose length matches the corresponding feature dimension.
An empty mapping is correct for outputs such as learned embeddings whose dimensions have no
individual semantic labels. Output names and layouts belong in `outputs`, not `feature_keys`.

## Provenance

Required provenance is:

```json
{
  "eris_version": "0.1.0",
  "config_fingerprint": "64-lowercase-hex-characters"
}
```

Optional `environment`, `created_at`, and `invocation` fields may record generation context.
Input hash belongs in `input`, producer version in `encoder`, and learned checkpoint identity in
`model`.

## Counts and extensions

`counts` maps non-empty names to non-negative plain integers, for example `n_nodes`, `n_edges`,
or `n_omitted_residues`. Strict validation checks counts where an output role establishes their
meaning.

`extensions` maps non-empty namespace strings to finite JSON mappings. Extensions may record
producer-specific provenance but cannot replace a required field or alter the core schema.

## Strict validation

`validate_schema_v2(meta, data, strict=True)` verifies:

- exact top-level and nested fields;
- family/type, granularity, coordinate-source, producer, input, and model compatibility;
- exact output/data key coverage, shapes, dtypes, counts, and feature-key cardinality;
- the configuration fingerprint;
- array safety and generic role invariants;
- learned embedding shape, finiteness, node-ID alignment, and optional valid-mask alignment.

Generic role invariants include `edge_index (2,E)`, aligned edge features, `node_features (N,F)`,
square matrices, four-dimensional grids, `node_embedding (N,D)`, and
`global_embedding (D,)`. The additive vector contract validates one-dimensional numeric
`pooled_features` with matching `pooled_attr_keys`. Learned node embeddings require aligned
unique Unicode `node_ids`.
Representation-specific scientific validation remains with the producer.

## Persistence limits

Schema validation provides structural integrity, not a complete numeric payload checksum:

- equivalent NPZ writes need not be byte-identical;
- `data.npz` and `meta.json` are staged and replaced independently, not atomically as a pair;
- different numeric values can pass when all validated metadata and array invariants still match.

Versionless and non-v2 artifacts are rejected rather than migrated.

## Artifact cache identity

The v2 cache key combines:

- `cache_contract=2` and `artifact_schema_version=2`;
- representation and encoder identity;
- configuration fingerprint;
- input SHA-256 and parser-relevant suffix;
- model and chain selection;
- learned model identity and effective precision where applicable.

Source paths, checkpoint paths, checkpoint mirrors, execution device, package version, and
timestamps are excluded. A hit for identical content at another source path refreshes only the
current `input.source`. Learned hits preserve original generation execution and checkpoint-source
provenance.

Cache entries live under `v2/<key>/`. Invalid, corrupt, or older entries are misses; Eris does not
migrate or delete them automatically.

## Schema v2 multi-chain and pooling finalization

Because v0.1 has not been released, the multi-chain changes finalized Schema v2 without a
competing legacy format or schema-version increment.

Collection-valued `input.chain_ids` replaces scalar `input.chain_id` for
every artifact. It is a non-empty JSON list of unique strings in selected-model
chain order. Single-chain artifacts use a one-element list. Residue-aligned
artifacts additionally carry a Unicode `chain_id` array aligned with their
rows or nodes. The resolved selection is included in the configuration fingerprint and cache
identity. Legacy scalar metadata is rejected rather than maintained as a parallel permanent
contract.

Structure-level pooling adds these declarations:

| Field | Value/contract |
| --- | --- |
| `representation_type` | `"vector"` |
| `granularity` | `"structure"` |
| data array | `pooled_attr`, shape `(F,)`, numeric dtype |
| data role | `pooled_features` |
| feature-key field | `feature_keys.pooled_attr_keys`, exactly `F` ordered strings |

A pooled artifact has `representation_name = "residue_pool"`, uses
`representation_family = "explicit"`, and copies the source artifact's
coordinate source.
Its canonical parameters identify the source array and ordered keys, reduction
mode, NaN policy, and a source-artifact fingerprint covering the selected
source data. The source artifact remains immutable.

All other frozen derived features fit Schema v2's ordered node/edge attributes
and declared auxiliary arrays. Exact feature keys, array roles, shapes, units,
provider information, and scientific parameters are recorded as specified in
[`structural_features.md`](structural_features.md). Current multi-chain graphs permit spatial
edges between chains. Cross-chain sequence separation uses the sentinel `-1` and the local flag
is zero. Opt-in `edge.same_chain`, `edge.inter_chain`, relative-direction, and interface-contact
columns annotate existing residue-graph edges without changing topology.
