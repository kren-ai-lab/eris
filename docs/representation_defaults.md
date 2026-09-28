# Eris v0.1 — Canonical Representation Defaults

This document is the approved M2.1 contract for the default configuration of the ten explicit
Eris v0.1 representations. It is the reference for M2.2–M2.8. Where earlier Registry metadata,
runtime behavior, or documentation differs, the values below are the canonical v0.1 decision.

M2.2 encodes these values in typed canonical configuration, and M2.3 makes the Registry derive
its defaults and parameter metadata from that authority. This document is not the Artifact Schema
contract.

## Global rules

### Units

- Structural distances and cutoffs use Å.
- `scale=None` preserves Å-based geometry.

### Common geometry

- `center=True`.
- `scale=None`.
- Centering semantics correspond to one effective centering operation.
- The default residue coordinate source is `ca`.

The implemented voxel path applies `center=True` exactly once, as established by M2.7.

### Residue extraction

The residue extraction defaults are:

- amino acids only: `True`;
- alternate-location policy: `"first"`;
- missing residue-center policy: `"skip"` by default, with `"error"` available for strict
  explicit residue encoding;
- `plddt_source="none"`; use `"bfactor"` only when the CA `b_iso` field is explicitly known to
  contain pLDDT.

With `"skip"`, omitted residues remain traceable through the artifact's
`omitted_residue_ids` output and `n_omitted_residues` count. These policies are represented
explicitly in the typed canonical configuration without changing the default value.

### Atom extraction

The atom extraction defaults are:

- `include_hetero=False`;
- `include_h=False`;
- `drop_water=True`;
- `keep_ions=True`;
- `altloc="first"`;
- `amino_acids_only=False`.

These are the current policies frozen for v0.1. This contract does not redefine their existing
interactions.

### Execution options

The following are not part of canonical representation configuration:

- `use_cache`;
- cache objects or instances;
- output directories;
- implementation identity or name fields.

## Feature defaults

### Atom node features

- Mode: `elem+role+bfactor+occ`.
- Default emitted width: 25.

### Residue node features

- Mode: `onehot+physchem+bfactor_ca`.
- Default emitted width: 37.
- `bfactor_ca` is the raw CA B-factor and is never implicitly normalized.
- `plddt_ca` is opt-in. Its default `normalization="none"` preserves the 0–100 scale;
  `normalization="unit_interval"` explicitly divides by 100.

The parameterized `aaindex` token reads AAIndex1 records from the `aaindex` provider. It requires
one or more accession IDs, preserves their requested order, and serializes the provider name,
provider version, source database, normalization, unknown-residue policy, and missing-value policy.
Its keys are `aaindex.<ACCESSION>`.

### Edge features

- Mode: `dist+rbf+seq+local`.
- `rbf_k=16`.
- `rbf_dmax=20.0 Å`.
- `local_cutoff=4`.
- Default emitted width: 19.

### Voxel channels

- Requested default: `occupancy+density+aa_onehot+bfactor`.
- Default physical width: 24.
- Scalar `plddt` and `bfactor` channels default to `normalization="none"`; the opt-in
  `"unit_interval"` mode divides by 100.

Requested token order is not necessarily physical channel order. Physical grid channel ordering
remains defined by feature Registry/spec ordering.

## Per-representation defaults

### `atom_point_cloud`

| parameter | canonical default |
| --- | --- |
| `center` | `True` |
| `scale` | `None` |
| atom extraction | global atom defaults |
| `node_attr_mode` | `elem+role+bfactor+occ` |
| numerical output | fixed `float32` |

### `residue_point_cloud`

| parameter | canonical default |
| --- | --- |
| `center` | `True` |
| `scale` | `None` |
| `residue_center` | `ca` |
| residue extraction | global residue defaults |
| `node_attr_mode` | `onehot+physchem+bfactor_ca` |
| numerical output | fixed `float32` |

### `atom_knn_graph`

| parameter | canonical default |
| --- | --- |
| `center` | `True` |
| `scale` | `None` |
| atom extraction | global atom defaults |
| `k` | `16` |
| `cutoff` | `12.0 Å` |
| `symmetric` | `True` |
| self-loops | not configurable in v0.1; none emitted |
| `node_attr_mode` | `elem+role+bfactor+occ` |
| `edge_attr_mode` | `dist+rbf+seq+local` |
| `rbf_k` | `16` |
| `rbf_dmax` | `20.0 Å` |
| `local_cutoff` | `4` |
| floating-point arrays | fixed `float32` |
| edge indices | fixed `int64` |

No public `include_self_loops` option exists for `atom_knn_graph` in v0.1.

### `residue_knn_graph`

| parameter | canonical default |
| --- | --- |
| `center` | `True` |
| `scale` | `None` |
| `residue_center` | `ca` |
| residue extraction | global residue defaults |
| `knn_mode` | `"local"` |
| `k` | `16` |
| `cutoff` | `12.0 Å` |
| `symmetric` | `True` |
| `include_self_loops` | `False` |
| `node_attr_mode` | `onehot+physchem+bfactor_ca` |
| `edge_attr_mode` | `dist+rbf+seq+local` |
| `rbf_k` | `16` |
| `rbf_dmax` | `20.0 Å` |
| `local_cutoff` | `4` |
| floating-point arrays | fixed `float32` |
| edge indices | fixed `int64` |

The canonical option name is `symmetric`, never `symmetrize`.

For residue graphs, `seq_sep` is the absolute difference between chain-relative biological
sequence positions. Numeric residue-number gaps are retained; insertion-code residues advance in
their source order. Each explicit residue graph selects one chain, so no cross-chain sequence
separation is represented.

### `residue_contact_graph`

| parameter | canonical default |
| --- | --- |
| `center` | `True` |
| `scale` | `None` |
| `residue_center` | `ca` |
| residue extraction | global residue defaults |
| `cutoff` | `8.0 Å` |
| `include_self_loops` | `False` |
| `node_attr_mode` | `onehot+physchem+bfactor_ca` |
| `edge_attr_mode` | `dist+rbf+seq+local` |
| `rbf_k` | `16` |
| `rbf_dmax` | `20.0 Å` |
| `local_cutoff` | `4` |
| floating-point arrays | fixed `float32` |
| edge indices | fixed `int64` |

### `residue_distogram`

| parameter | canonical default |
| --- | --- |
| `center` | `True` |
| `scale` | `None` |
| `residue_center` | `ca` |
| residue extraction | global residue defaults |
| `include_self` | `True` |
| payload | raw pairwise distances; no contact cutoff |
| payload dtype | fixed `float32` |

### `residue_distogram_bins`

| parameter | canonical default |
| --- | --- |
| `center` | `True` |
| `scale` | `None` |
| `residue_center` | `ca` |
| residue extraction | global residue defaults |
| `edges` | `None` |
| `dmin` | `2.0 Å` |
| `dmax` | `20.0 Å` |
| `num_bins` | `36` |
| `include_self` | `True` |
| `add_no_contact_class` | `True` |
| `dtype` | `uint8` |

The 36 bins produce 37 bin edges.

### `residue_contact_map`

| parameter | canonical default |
| --- | --- |
| `center` | `True` |
| `scale` | `None` |
| `residue_center` | `ca` |
| residue extraction | global residue defaults |
| `cutoff` | `8.0 Å` |
| `include_self` | `False` |
| `dtype` | `uint8` |
| values | binary `0` or `1` |

`bool` is not the canonical v0.1 contact-map dtype.

### `residue_contact_map_bins`

| parameter | canonical default |
| --- | --- |
| `center` | `True` |
| `scale` | `None` |
| `residue_center` | `ca` |
| residue extraction | global residue defaults |
| `edges` | `None` |
| `dmin` | `2.0 Å` |
| `dmax` | `20.0 Å` |
| `num_bins` | `36` |
| `include_self` | `False` |
| `add_no_contact_class` | `True` |
| `dtype` | `uint8` |

The 36 bins produce 37 bin edges.

### `residue_voxel_grid`

| parameter | canonical default |
| --- | --- |
| `center` | `True` |
| `scale` | `None` |
| `residue_center` | `ca` |
| residue extraction | global residue defaults |
| `grid_size` | `64` |
| `voxel_size` | `None` |
| `fit` | `True` |
| `margin` | `4.0 Å` |
| `channels` | `occupancy+density+aa_onehot+bfactor` |
| `sigma` | `1.0` voxel unit |
| `pool` | `"max"` |
| `clamp` | `False` |
| `dtype` | `float32` |

The default physical tensor layout is `(C, X, Y, Z)` with `C=24`. `float16` may remain a
supported non-default option.

## Dtype rules

| payload | canonical dtype |
| --- | --- |
| coordinates and node/edge numerical features | `float32` |
| `edge_index` | `int64` |
| graph statistics | `float32` |
| distogram | `float32` |
| contact map | `uint8` |
| binned matrices | `uint8` |
| voxel grid default | `float32` |

These rules do not create configurable dtype options where the runtime does not expose them.

## Topology semantics

### Symmetric kNN

`symmetric=True` means:

- directed pairs are unique;
- a reverse edge exists for every directed edge;
- no directed pair is duplicated.

### Self-loops

- Atom kNN has no public v0.1 self-loop option and emits no self-loops.
- Residue kNN self-loops are configurable and default to `False`.
- Residue contact-graph self-loops are configurable and default to `False`.

These rules preserve the topology decisions closed in M1.

## Binning semantics

The uniform generated defaults are:

- `dmin=2.0 Å`;
- `dmax=20.0 Å`;
- `num_bins=36`;
- number of edges: 37.

Explicitly supplied `edges` override the generated defaults according to the current
implementation. The no-contact class is enabled by default for both binned matrices.
`residue_distogram_bins` defaults to `include_self=True`, while
`residue_contact_map_bins` defaults to `include_self=False`.

## Known historical Registry drift

This section is non-normative. The following are known pre-M2 Registry drift, not alternate
defaults:

- `symmetrize=False` instead of canonical `symmetric=True`;
- 32 bins instead of canonical 36;
- contact-map `bool` instead of canonical `uint8`;
- voxel `32` / `2.0` / `occupancy` / `sum` / `False` instead of canonical
  `64` / `4.0` / five-token channels / `max` / `True`;
- Registry `level=ca` instead of residue granularity;
- an unsupported atom `include_self_loops` option;
- historical representation identifiers instead of canonical public names.

M2.3 removed these independently maintained Registry values; this list records historical drift only.

## Known implementation follow-ups

### Voxel centering

M2.7 verified that canonical `center=True` produces exactly one effective centering operation.

### Requested versus effective `voxel_size`

The implementation distinguishes configured `voxel_size=None` from the effective spacing derived
when `fit=True`. The canonical default remains `None`.
