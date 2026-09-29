# External residue numerical features

Eris accepts externally computed residue-level numerical features through the normal residue
`node_attr` feature pipeline. The v0.1 token is `external`; it does not run a model, download data,
or create a separate embedding tensor.

## NPZ payload

The input must be a `.npz` archive readable with `np.load(path, allow_pickle=False)`. Required
fields are:

| Field | Contract |
| --- | --- |
| `values` | Real numeric matrix `(N, D)`, with `N,D >= 1`; consumed as contiguous `float32`. |
| `residue_ids` | Unique string vector `(N,)` in canonical Eris residue order/identity. |
| `source_name` | Non-empty scalar identifier used in public feature keys. |
| `chain_id` | Scalar string naming the structure chain used to build the payload. |
| `model_index` | Scalar integer naming the structure model used to build the payload. |

Optional scalar strings are `source_version` and `source_model`. Optional `feature_keys` is a
unique, non-empty string vector of length `D`. If omitted, Eris generates `"0"` through
`str(D - 1)`.

`source_name` and source-local feature keys must start with an ASCII letter or digit and may then
contain ASCII letters, digits, underscores, or hyphens. Public keys are:

```text
external.<source_name>.<feature_key>
```

Object arrays, pickle payloads, complex/string/boolean value matrices, malformed scalars, unknown
archive fields, and non-`.npz` files are rejected.

## Residue identity and alignment

Each residue ID must use the existing canonical form:

```text
<chain_id>:<residue_number>:<insertion_code>:<residue_name>
```

For example, `A:42::GLY` and `A:42:A:GLY` are different residues. The only v0.1 alignment policy is
`strict_residue_id`. External rows may be shuffled, but Eris reorders them solely by exact residue
ID. Missing, extra, or duplicate IDs, chain/model mismatches, insertion-code differences, and
residue-name differences fail. Eris does not truncate, pad, match by row position, or perform
sequence alignment.

Alignment uses the exact retained node set. If structure extraction omits a residue but the `.npz`
still contains it, that external row is extra and encoding fails.

## Content identity

At configuration time Eris validates the payload and records its resolved absolute path, source
metadata, chain/model identity, feature dimension and keys, alignment policy, and a canonical
`content_sha256`. The hash covers the canonical contiguous `float32` values and all scientific
metadata, not the `.npz` container bytes or path. Equivalent payloads at different paths therefore
share a content hash.

The resolved descriptor has two serialized views. `config.to_dict()` retains the absolute path for
invocation provenance. The scientific projection used by artifact `parameters`, the configuration
fingerprint, and cache identity excludes only `path`; it retains `content_sha256`, source metadata,
chain/model identity, feature dimension and keys, and alignment. The artifact records the actual
invocation path under `extensions.org.eris.external_features`. Relocating an identical payload can
therefore reuse the same cache entry without losing the path used for that invocation.

Eris reloads and verifies the payload before cache lookup, again after a valid cache hit before it
is returned, and again during feature construction on a cache miss. Changing a file after
configuration resolution therefore fails instead of using stale identity, including mutation
during cache lookup.

## Hybrid and high-dimensional node features

`external` composes with the other residue features in the normal registry. For example,
`onehot + physchem + aaindex + bfactor_ca + external` produces one `node_attr` matrix and one
ordered `node_attr_keys` list; it does not create a separate embedding output. External widths such
as 512 or 1024 are supported without padding, truncation, or a feature-width limit and flow through
residue graphs, artifacts, Torch export, PyG export, and folder datasets.

Physical `node_attr` block ordering follows canonical feature-registry `FeatureSpec.order`, not the
order in which callers write feature tokens. Within each parameterized block, the feature's own
contract preserves its canonical key ordering, including AAIndex accessions and external columns.

Once an artifact has been encoded and saved, its numerical arrays and metadata are self-contained.
Loading, packing, dataset access, and Torch/PyG export do not reopen or revalidate the source
external `.npz`. The source file is required only while resolving or encoding a configuration and
while checking encoder-cache reuse.

## Minimal example

```python
import numpy as np

from eris import encode

np.savez(
    "protein_A.npz",
    values=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    residue_ids=np.asarray(["A:1::ALA", "A:2::GLY"]),
    source_name=np.asarray("custom_descriptor_v2"),
    source_version=np.asarray("1.0"),
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

The Python API accepts explicit per-row overrides without changing the canonical structure
manifest fields (`id`, `path`, `chain`, and `model`):

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
                "physchem",
                {"aaindex": {"ids": ["KYTJ820101"]}},
                {"external": {"path": "A.npz"}},
            ]
        },
        "protein_B": {
            "node_attr_mode": [
                "onehot",
                "physchem",
                {"aaindex": {"ids": ["KYTJ820101"]}},
                {"external": {"path": "B.npz"}},
            ]
        },
    },
)
```

For each row, Eris copies the global `params` and applies that row's mapping with a shallow
top-level update. A row-specific `node_attr_mode` therefore replaces the global value entirely;
nested feature parameters are not recursively merged. Every override key must match a manifest row
ID. The batch CLI continues to expose only global `--params`; `params_by_id` is currently a Python
API feature.

