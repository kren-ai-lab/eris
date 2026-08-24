# Eris Artifact Schema v2

Artifact Schema v2 is the canonical active runtime artifact contract for Eris v0.1. It
describes explicit and learned structural representations with one Registry-independent
metadata and array contract. All ten canonical explicit encoders produce complete, strict-valid
Schema v2 artifacts directly from `BaseEncoder.encode()`.

The canonical on-disk form is `meta.json` plus `data.npz`. `StructureArtifact` validates,
finalizes, saves, and loads Schema v2; Schema v1 is unsupported by these public operations.

## Top-level contract

The following fields are all required, and no other top-level fields are accepted except the
two optional fields listed below:

| field | contract |
| --- | --- |
| `schema_version` | Integer exactly equal to `2`. |
| `representation_name` | Non-empty public representation identifier. |
| `representation_family` | `explicit` or `learned`. |
| `representation_type` | Family-compatible type. |
| `granularity` | Represented biological entity. |
| `coordinate_source` | Source/meaning of structural coordinates. |
| `encoder` | Required producer implementation identity. |
| `input` | Required source identity and selection. |
| `parameters` | Canonical requested configuration. |
| `provenance` | Required Eris/config provenance and optional execution context. |
| `outputs` | Ordered declarations corresponding exactly to array keys. |
| `feature_keys` | Authoritative ordered feature semantics. |
| `shapes` | Exact array shapes by output name. |
| `dtypes` | Exact canonical NumPy dtype strings by output name. |
| `counts` | Generic non-negative integer summaries. |

Optional top-level fields are:

- `model`, required conditionally for learned artifacts and forbidden for explicit artifacts;
- `extensions`, for namespaced metadata which does not alter the core contract.

Unknown top-level fields are rejected. Extensions cannot replace a missing required field.

## Identity vocabularies

`representation_family` has exactly two values:

- `explicit`;
- `learned`.

Family and type must agree:

- explicit: `cloud`, `graph`, `matrix`, or `grid`;
- learned: `embedding`.

`granularity` is one of `atom`, `residue`, `structure`, or `mixed`. It never contains `ca`.

`coordinate_source` is one of:

- `atom`;
- `ca`;
- `residue_mean`;
- `residue_com`;
- `model_specific`.

An explicit atom representation uses `coordinate_source = atom`. An explicit residue
representation records `ca`, `residue_mean`, or `residue_com` according to canonical config.
Learned preprocessing may use `model_specific`. A global learned representation uses
`granularity = structure` and a mixed learned representation uses `granularity = mixed`, both
with `coordinate_source = model_specific` when model preprocessing owns coordinate semantics.

M3.1 does not add speculative `backbone` or `all_atom` coordinate-source values.

## Encoder and model

`encoder` is required for both families and has the exact form:

```json
{
  "implementation": "eris.encoders.graphs.ResidueKNNGraph",
  "version": "0.1.0"
}
```

Both values are non-empty strings. `implementation` is a dotted import-path-like producer
identity. The representation name is not duplicated in this block.

An explicit artifact must not contain `model`. A learned artifact must contain it:

```json
{
  "architecture": "structural-model-name",
  "checkpoint": {
    "identifier": "checkpoint-name",
    "revision": "optional-revision",
    "source": "optional-source-locator",
    "sha256": "optional-lowercase-sha256"
  }
}
```

`architecture` and `checkpoint.identifier` are required non-empty strings. `revision` and
`source` are optional strings. When present, `sha256` is exactly 64 lowercase hexadecimal
characters. Unknown model or checkpoint fields are rejected. This common contract does not
implement GearNet or any learned backend.

## Input

The required input block is:

```json
{
  "source": "locator/to/or/name/of/input",
  "source_sha256": "64-lowercase-hex-characters",
  "format": "pdb",
  "model_index": 0,
  "chain_id": null
}
```

`source` and `format` are non-empty strings. `source_sha256` hashes the input source and is a
lowercase SHA-256 value. `model_index` is an integer greater than or equal to zero; a Boolean is
not an integer for this contract. `chain_id` is a string or `null`.

Only these additional input keys are permitted in v0.1:

- `original_filename`;
- `structure_id`;
- `source_database`.

Their values are strings. Assembly identity and parsed-content hashes are not part of M3.1.

## Parameters: requested versus effective values

`parameters` is a JSON-compatible mapping. The producer contract for later M3 work is:

```text
parameters == encoder.config.to_dict()
```

It records the canonical request, including `null` when the user intentionally delegates an
effective value to runtime computation. It does not duplicate config validation or scientific
parameter semantics. Mapping keys are strings; nested mappings/lists, strings, Booleans,
integers, finite floats, and `null` are supported. NumPy values, NumPy arrays, `Path`, NaN, and
infinities are prohibited in parameters.

For example, a binned representation can retain the request:

```json
{"edges": null, "n_bins": 32}
```

while declaring a computed `bin_edges` array in `outputs`. Similarly, a voxel request can keep
`voxel_size: null` while the effective voxel size is an output or an explicitly specified
extension in a later producer contract. Runtime-effective results must not overwrite the
canonical requested parameters.

## Provenance

The required minimum is exactly the Eris package version and config fingerprint:

```json
{
  "eris_version": "0.1.0",
  "config_fingerprint": "64-lowercase-hex-characters"
}
```

Optional fields are:

- `environment`: a finite JSON mapping; known useful keys include `python_version`, `platform`,
  and `dependencies`, but none is required;
- `created_at`: a UTC ISO-8601 string ending in `Z`;
- `invocation`: a JSON-compatible mapping or string.

Python/platform are not required core provenance. There is no full `pip freeze` or hardware
inventory requirement. Input hash belongs to `input.source_sha256`; encoder implementation
version belongs to `encoder.version`; learned checkpoint identity belongs to `model`.

## Configuration fingerprint

`compute_config_fingerprint(representation_name, parameters)` calculates lowercase SHA-256 over
the UTF-8 canonical JSON representation of exactly:

```json
{
  "representation_name": "...",
  "parameters": {}
}
```

Canonical JSON is produced with:

```python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
)
```

The fingerprint is deterministic and insensitive to mapping insertion order. It changes when
the representation name or canonical requested parameters change. It does not depend on schema
version, encoder/package version, source, paths, arrays, timestamps, environment, or cache
settings.

`config_fingerprint` is not a complete artifact identity and is not itself a cache key. The active
v2 cache boundary combines it with schema/cache contract versions, representation and encoder
identity, source SHA-256, parser hint, and input selection. This independent v2 identity does not
replace `BaseEncoder.fingerprint()`, which remains available as a distinct legacy/Registry
contract but is not used by active encoding cache lookup.

## Outputs

`outputs` is a non-empty ordered list. Every declaration requires `name` and `role`:

```json
[
  {
    "name": "node_attr",
    "role": "node_features",
    "granularity": "residue",
    "layout": "NF",
    "description": "Optional human-readable description"
  }
]
```

Names are unique and match `^[a-z][a-z0-9_]*$`. Each name corresponds to exactly one array, and
the set of output names equals the set of `data` keys. `role` is a non-empty extensible string;
M3.1 intentionally does not define a globally closed role vocabulary. Optional `granularity`
uses the core granularity vocabulary, `layout` is a non-empty string, and `description` is a
string. Shapes and dtypes are not duplicated inside output declarations.

Core roles with generic strict invariants include `edge_index`, `edge_distance`,
`node_features`, `edge_features`, `graph_features`, `matrix`, `grid`, `node_embedding`,
`global_embedding`, `node_ids`, and `valid_mask`. Other role names remain valid and can be
specified by later representation contracts.

## Authoritative feature keys

`feature_keys` is always present and is never inferred. `{}` is valid, including for learned
embeddings whose dimensions have no individual semantic names. The only core-known fields are:

- `node_attr_keys`;
- `edge_attr_keys`;
- `graph_attr_keys`;
- `grid_channels`.

Each value is an ordered `list[str]`. In strict validation, cardinality agrees with the
corresponding `node_features`, `edge_features`, `graph_features`, or `grid` role when the field
is present. Legacy layout fields such as `matrix_key`, `matrix_attr_keys`, `grid_key`, and
`grid_layout` are rejected. Output identity and layout belong to `outputs`, not `feature_keys`.
Embedding dimension labels are not generated.

## Shapes, dtypes, and counts

`shapes` maps every output name to a list of non-negative integer dimensions. A scalar ndarray
has shape `[]`. `dtypes` maps every output name to `str(array.dtype)`. Both mappings have exactly
the same keys as `outputs` and `data`; strict validation checks their values against each array.

`counts` maps non-empty string names to non-negative plain integers. It stays generic rather
than duplicating every shape. Representative values are:

- graph: `n_nodes`, `n_edges`, `n_self_loops`;
- learned node output: `n_nodes`;
- global learned output: `{}`.

Strict validation checks counts only where a generic role establishes their meaning. Scientific
topology, contact, binning, and other representation-specific semantics are outside this module.

## Extensions

`extensions` is an optional mapping of non-empty namespace strings to finite JSON mappings:

```json
{
  "org.example.analysis": {
    "score": 0.5
  }
}
```

The core validator verifies namespace/value structure but does not interpret extension
semantics. None of the ten current explicit representations requires an extension.

## JSON normalization

`normalize_metadata()` creates a new JSON-safe value without mutating the input:

- mapping keys must be strings;
- tuples become lists;
- NumPy scalar values become corresponding Python scalars during controlled normalization;
- `None`, Boolean, integer, finite float, and string values are preserved;
- NumPy arrays and `Path` values are rejected;
- NaN and positive or negative infinity are rejected.

The stricter parameters rule rejects NumPy scalars as well as arrays so canonical config output
cannot silently depend on NumPy types.

## Array and NPZ safety policy

Every data key follows `^[a-z][a-z0-9_]*$`, and every value is a `numpy.ndarray`. Allowed dtype
kinds are signed integer, unsigned integer, float, Boolean, and Unicode string. Object,
structured/record, complex, datetime, timedelta, and byte-string arrays are rejected. Schema v2
therefore has no pickle-dependent array payload.

The same array-safety policy is enforced in memory and at persistence boundaries.

## Loading and disk integrity

Persisted Schema v2 artifacts are always loaded with strict validation, regardless of the
historical `validate` and `strict` arguments retained in the method signature. NPZ data is opened
with `allow_pickle=False`, and its member names must exactly match the names declared by
`outputs`. Schema v1 and versionless provisional v1 artifacts are rejected rather than loaded.

`outputs` is the semantic ordering authority. Physical ZIP member order is not contractual; a
loader reconstructs the runtime data mapping in `outputs` order after confirming exact member
coverage. Shapes, dtypes, feature cardinality, counts, learned alignment, and other generic v2
integrity rules are then checked by the strict schema validator.

This provides structural integrity, not cryptographic generation identity. Persistence has three
deliberate limitations:

- an NPZ archive is not guaranteed to be byte-identical across equivalent writes;
- staging and replacement protect each file independently, but replacing `data.npz` and
  `meta.json` is not a pair-level transactional atomic operation;
- Schema v2 has no checksum for the complete numeric payload, so structural validation cannot
  detect different valid numeric values with identical keys, shapes, dtypes, counts, and other
  validated semantic metadata.

## Schema v2 cache identity

The active Schema v2 cache key hashes a canonical mapping containing separate
`cache_contract = 2` and `artifact_schema_version = 2` fields, representation name, encoder
implementation and version, config fingerprint, input source SHA-256, parser hint, and structured
chain/model selection. An optional canonical model mapping participates when present.

`parser_hint` is derived only from the normalized parser-relevant suffix. It is path-independent:
`.PDB` and `.pdb` are equivalent, while `.pdb`, `.ent`, `.cif`, `.mmcif`, and their compressed
forms remain conservatively distinct. An unknown terminal suffix is preserved in normalized form;
no suffix uses the explicit value `<no-suffix>`.

Input source locator, Eris package version, confirmed input format, representation type,
granularity, coordinate source, and the full parameter mapping are excluded. The source SHA
identifies bytes, selection identifies the requested structure subset, `parser_hint` prevents
cross-parser cache reuse, and `config_fingerprint` identifies canonical requested parameters.

On a cache miss, `input.format` is the format confirmed by Gemmi. On a hit, the cached confirmed
format is preserved; it is never inferred from the suffix. Because the source locator is excluded
from the key, a valid hit rebuilds only `input` with the current invocation's `source`, hash, model,
and chain selection, then validates the refreshed artifact. The persisted cache entry is not
rewritten merely to refresh the locator.

Schema v2 cache entries live under a separate `v2/<key>/` namespace and use canonical
`data.npz`/`meta.json` files. Legacy, stale, corrupt, or invalid entries are cache misses; they are
never migrated to Schema v2 and are not automatically deleted. Active encoding calls only the v2
cache API; legacy flat cache entries are ignored.

## Basic and strict validation

`validate_schema_v2(meta, data, strict=False)` returns a normalized metadata copy. Basic
validation enforces JSON structure, exact fields, identity compatibility, encoder/input/
provenance/model structures, ordered output declarations, output/data equality, array safety,
feature-key structure, and shapes/dtypes/counts mapping structure.

With `strict=True`, it additionally verifies exact shapes and dtypes, feature-key cardinality,
the config fingerprint against `representation_name` plus `parameters`, and generic role
invariants:

- `edge_index`: integer `(2, E)`;
- `edge_distance`: `(E,)` and aligned with `edge_index` when both exist;
- `node_features`: `(N, F)`;
- `edge_features`: `(E, F)` and aligned with `edge_index` when both exist;
- `graph_features`: one-dimensional;
- `matrix`: two-dimensional and square;
- `grid`: four-dimensional;
- `node_embedding`: finite floating `(N, D)` with positive `N` and `D`;
- `global_embedding`: finite floating `(D,)` with positive `D`;
- `node_ids`: one-dimensional Unicode with non-empty values and unique learned-node identity;
- `valid_mask`: one-dimensional Boolean aligned to `node_embedding`.

A learned `node_embedding` requires aligned `node_ids`. The validator does not require explicit
residue identifiers to be globally unique, does not import Registry, and does not perform
scientific validation.

## Learned embedding contract example

A strict-valid mixed learned artifact can declare these arrays:

| output | role | canonical shape/dtype |
| --- | --- | --- |
| `node_embedding` | `node_embedding` | `(N, D)`, floating |
| `global_embedding` | `global_embedding` | `(D,)`, floating |
| `node_ids` | `node_ids` | `(N,)`, Unicode |
| `valid_mask` | `valid_mask` | `(N,)`, Boolean |

It uses `representation_family = learned`, `representation_type = embedding`,
`granularity = mixed`, `coordinate_source = model_specific`, a populated `model`, an empty
`feature_keys` mapping, and `counts.n_nodes = N`.

## Schema v1 hard cut

Schema v1 is a pre-release internal format and has no public v0.1 compatibility guarantee. The
active runtime policy is:

- new encodes and saves produce Schema v2 only;
- Schema v1 load, validate, finalize, and save are explicitly rejected as legacy, including
  `save(validate=False)`;
- no transparent migration or automatic provenance reconstruction is performed;
- versionless provisional v1 metadata is unsupported;
- artifacts should be regenerated from their source and canonical config.

Legacy classification remains solely to produce clear unsupported-format errors. No active Schema
v1 validation, finalization, save, load, or migration implementation remains. Batch processing
validates an existing artifact before skipping it, so a v1, corrupt, or incomplete output cannot
silently satisfy a request.

## Encode CLI boundary

`eris encode single` saves the strict-valid Schema v2 artifact returned by the encoder. Historical
`--node-emb` and `--node-emb-key` options were removed because they mutated payload arrays after
schema construction. Learned embeddings belong to the later `eris embed` learned-representation
path; they are not injected into explicit artifacts.
