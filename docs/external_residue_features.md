# External residue features

The residue feature token `external` adds a validated numerical matrix to the normal
`node_attr` pipeline. It does not run a model, download data, or create a separate embedding
output.

## NPZ schema

The source must be a `.npz` archive readable with `np.load(path, allow_pickle=False)`.

| Field | Required | Contract |
| --- | --- | --- |
| `values` | yes | real numeric matrix `(N,D)` with `N,D >= 1`; converted to contiguous `float32` |
| `residue_ids` | yes | unique string vector `(N,)` |
| `source_name` | yes | non-empty scalar string used in public feature keys |
| `chain_id` | yes | scalar string for the source chain |
| `model_index` | yes | scalar integer greater than or equal to zero |
| `feature_keys` | no | unique non-empty string vector of length `D` |
| `source_version` | no | scalar string |
| `source_model` | no | scalar string |

If `feature_keys` is absent, Eris generates `"0"` through `str(D - 1)`. `source_name` and each
feature key must start with an ASCII letter or digit and may otherwise contain ASCII letters,
digits, underscores, or hyphens. Public keys are:

```text
external.<source_name>.<feature_key>
```

Unknown archive fields, object arrays, pickle payloads, malformed scalars, and Boolean, string,
complex, or out-of-range `values` fail.

## Residue identity and alignment

Each row uses the canonical residue ID:

```text
<chain_id>:<residue_number>:<insertion_code>:<residue_name>
```

Examples:

```text
A:42::GLY
A:42:A:GLY
```

The only alignment policy is `strict_residue_id`. Rows may be shuffled; Eris reorders them by
exact ID. Missing, extra, duplicate, or malformed IDs fail. Chain, model, insertion-code, and
residue-name mismatches also fail.

Alignment uses the exact residue nodes retained by structure extraction. If extraction omits a
residue that remains in the NPZ, that row is extra. Eris never truncates, pads, matches by position,
or performs sequence alignment.

## Source metadata and content hash

Configuration resolution validates the payload and records:

- resolved absolute invocation path;
- source name, version, and model;
- chain and model identity;
- feature dimension and ordered keys;
- alignment policy;
- canonical `content_sha256`.

The hash covers contiguous little-endian `float32` values, ordered residue IDs and feature keys,
source metadata, chain/model identity, dimensions, and alignment policy. It does not hash NPZ
container bytes or the path. Scientifically identical payloads at different paths share a content
hash.

The resolved config has two views:

- `config.to_dict()` retains `path` so the encoder can open and validate the payload;
- the scientific parameter/fingerprint projection removes only `path`.

Artifacts record the actual invocation path and content hash under
`extensions.org.eris.external_features`. The content hash and all other scientific descriptor
fields remain in artifact `parameters`.

Eris reloads and verifies the payload before cache lookup, after a cache hit, and during feature
construction on a miss. Mutation after configuration resolution fails.

## Hybrid use

`external` composes with other residue features:

```python
node_attr_mode = [
    "onehot",
    "physchem",
    {"aaindex": {"ids": ["KYTJ820101"]}},
    "bfactor_ca",
    {"external": {"path": "protein_A.npz"}},
]
```

All selected blocks become one contiguous `node_attr (N,F)` matrix and one ordered
`node_attr_keys` list. Physical block order follows `FeatureSpec.order`; external column order
follows the validated `feature_keys` vector. There is no fixed external width, padding, or
truncation.

External features flow through residue point clouds and graphs, Artifact Schema v2, packing,
datasets, Torch export, and graph-only PyG export. Once saved, an artifact is self-contained;
loading and downstream operations do not reopen the source NPZ.

## Minimal producer and encode example

```python
import numpy as np

from eris import encode

np.savez(
    "protein_A.npz",
    values=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    residue_ids=np.asarray(["A:1::ALA", "A:2::GLY"]),
    source_name=np.asarray("custom_descriptor_v2"),
    source_version=np.asarray("1.0"),
    feature_keys=np.asarray(["score_a", "score_b"]),
    chain_id=np.asarray("A"),
    model_index=np.asarray(0, dtype=np.int64),
)

artifact = encode(
    "protein.pdb",
    representation="residue_point_cloud",
    chain_id="A",
    node_attr_mode=[
        "onehot",
        {"external": {"path": "protein_A.npz"}},
    ],
)
```

## Per-row batch parameters

Python `batch_encode()` accepts shallow overrides keyed by manifest row ID:

```python
from eris import batch_encode

batch_encode(
    rows=rows,
    representation="residue_knn_graph",
    artifact_root="artifacts",
    params={"k": 8},
    params_by_id={
        "protein_A": {
            "node_attr_mode": [
                "onehot",
                {"external": {"path": "A.npz"}},
            ]
        },
        "protein_B": {
            "node_attr_mode": [
                "onehot",
                {"external": {"path": "B.npz"}},
            ]
        },
    },
)
```

Each row starts with a copy of global `params` and receives one top-level `dict.update()`.
Nested feature settings are not recursively merged. Every override key must match a manifest row
ID. The batch CLI exposes only global `--params`.

## Failure conditions

Encoding fails for:

- missing, unreadable, non-NPZ, or pickle-dependent sources;
- missing, unknown, malformed, or incorrectly typed fields;
- non-finite conversion overflow or invalid matrix shape;
- duplicate keys or residue IDs;
- payload/structure chain or model mismatch;
- missing or extra residue identities;
- descriptor or content changes after resolution.
