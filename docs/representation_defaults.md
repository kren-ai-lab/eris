# Explicit representation defaults

`src/eris/core/config.py` is the executable authority for these defaults. Registry discovery,
artifact parameters, and configuration fingerprints derive from the same typed configs.

## Common geometry and selection

All distances are in ångström. `center=True` subtracts one centroid before representation
construction; `scale=None` preserves ångström-based geometry.

| Scope | Parameter | Default | Meaning |
| --- | --- | --- | --- |
| all | `center` | `True` | center coordinates once |
| all | `scale` | `None` | optional positive coordinate divisor |
| residue | `residue_center` | `"ca"` | alpha-carbon coordinate |
| residue | `amino_acids_only` | `True` | frozen for v0.1 |
| residue | `altloc` | `"first"` | frozen alternate-location policy |
| residue | `missing_residue_center` | `"skip"` | omit and record missing centers; `"error"` is available |
| residue | `plddt_source` | `"none"` | use `"bfactor"` only for inputs known to store pLDDT in CA B-factor |
| atom | `include_hetero` | `False` | exclude hetero residues |
| atom | `include_h` | `False` | exclude hydrogen atoms |
| atom | `drop_water` | `True` | remove water |
| atom | `keep_ions` | `True` | retain ions when otherwise selected |
| atom | `altloc` | `"first"` | select the first encountered location |
| atom | `amino_acids_only` | `False` | do not restrict atom extraction to amino acids |

Residue coordinate sources in artifacts are `ca`, `residue_mean`, and `residue_com` for
`residue_center="ca"`, `"mean"`, and `"com"`. Mean uses selected residue atoms; center of mass
uses their atomic masses. With `missing_residue_center="skip"`, omitted identities are stored in
`omitted_residue_ids` and counted as `n_omitted_residues`.

`use_cache`, cache objects, output paths, encoder names, and implementation identities are
execution concerns, not representation parameters.

## Feature defaults

### Atom node features

| Setting | Default |
| --- | --- |
| `node_attr_mode` | `elem+role+bfactor+occ` |
| emitted width | 25 |
| output dtype | `float32` |

The artifact also retains the raw atom fields, including element index, atom role, B-factor,
occupancy, residue identity, residue name, atom name, and residue sequence position.
The atomic `bfactor` node-feature column is the raw experimental value. It is never implicitly
divided by 100, clipped, or interpreted as pLDDT.

### Residue node features

| Setting | Default |
| --- | --- |
| `node_attr_mode` | `onehot+physchem+bfactor_ca` |
| emitted width | 37 |
| output dtype | `float32` |

Registered residue feature tokens are:

| Token | Semantics |
| --- | --- |
| `onehot` | canonical amino-acid one-hot block |
| `aa_index` | numeric amino-acid index |
| `physchem` | fixed Eris physicochemical descriptors |
| `aaindex` | requested AAIndex1 accessions |
| `plddt_ca` | explicitly sourced CA pLDDT |
| `bfactor_ca` | raw CA B-factor |
| `flags` | glycine, proline, unknown, and CA-presence flags |
| `external` | aligned external numerical values |

Physical blocks follow `FeatureSpec.order`, not request order. `node_attr_keys` is authoritative.
The parameterized `aaindex` token requires one or more accession IDs and preserves their
canonicalized request order. It records the installed `aaindex` provider version, uses
`normalization="none"`, and supports `nan`, `zero`, or `error` policies for unknown residues and
missing record values.

`bfactor_ca` is never normalized or interpreted from its range. `plddt_ca` is opt-in:
`normalization="none"` preserves 0–100 and `"unit_interval"` divides by 100. Non-finite missing
pLDDT remains `NaN` after extraction; the normal explicit encoder feature builder fills a
requested missing `plddt_ca` value with zero. Finite values outside 0–100 fail.

The `external` token follows [the external residue feature contract](external_residue_features.md).

### Graph edge and graph features

| Setting | Default |
| --- | --- |
| `edge_attr_mode` | `dist+rbf+seq+local` |
| `rbf_k` | 16 |
| `rbf_dmax` | 20.0 Å |
| `local_cutoff` | 4 sequence positions |
| emitted edge width | 19 |

Graph encoders also emit deterministic `graph_attr` statistics and ordered
`graph_attr_keys`. Floating graph arrays use `float32` and `edge_index` uses `int64`.

For residue graphs, sequence separation is the absolute difference between chain-relative
biological positions. Numeric residue-number gaps are retained; insertion-code residues advance in
source order. Each explicit residue graph contains one selected chain.

### Voxel channels

| Setting | Default |
| --- | --- |
| `channels` | `occupancy+density+aa_onehot+bfactor` |
| physical width | 24 |
| layout | `(C, X, Y, Z)` |

Physical channel order follows the grid feature registry. Scalar `bfactor` and opt-in `plddt`
channels use `normalization="none"` by default; `"unit_interval"` divides by 100.

## Representation table

Common geometry and selection defaults above apply unless a row says otherwise.

| Registered name | Type | Important defaults | Primary output |
| --- | --- | --- | --- |
| `atom_point_cloud` | cloud | atom feature defaults | `pos (N,3)`, `node_attr (N,25)` |
| `residue_point_cloud` | cloud | residue feature defaults | `pos (N,3)`, `node_attr (N,37)` |
| `atom_knn_graph` | graph | `k=16`, `cutoff=12.0`, `symmetric=True` | `edge_index (2,E)` plus node/edge/graph features |
| `residue_knn_graph` | graph | `knn_mode="local"`, `k=16`, `cutoff=12.0`, `symmetric=True`, `include_self_loops=False` | `edge_index (2,E)` plus node/edge/graph features |
| `residue_contact_graph` | graph | `cutoff=8.0`, `include_self_loops=False` | `edge_index (2,E)` plus node/edge/graph features |
| `residue_distogram` | matrix | `include_self=True` | `dist (N,N)`, `float32` |
| `residue_distogram_bins` | matrix | uniform bin defaults; `include_self=True` | `dist_bins (N,N)`, default `uint8` |
| `residue_contact_map` | matrix | `cutoff=8.0`, `include_self=False` | `contact (N,N)`, default `uint8` |
| `residue_contact_map_bins` | matrix | uniform bin defaults; `include_self=False` | `contact_bins (N,N)`, default `uint8` |
| `residue_voxel_grid` | grid | voxel defaults below | `grid (24,64,64,64)` by default |

### k-nearest-neighbor graphs

`symmetric=True` emits unique directed pairs with a reverse edge for every edge. Atom kNN has no
public self-loop option and emits none. Residue kNN supports `include_self_loops`, defaulting to
`False`.

For residue kNN, `knn_mode="local"` uses the configured cutoff. `knn_mode="global"` sets the
effective cutoff to `None`. The public option is `symmetric`, not `symmetrize`.

### Binned matrices

| Parameter | Default |
| --- | --- |
| `edges` | `None` |
| `dmin` | 2.0 Å |
| `dmax` | 20.0 Å |
| `num_bins` | 36 |
| `add_no_contact_class` | `True` |
| `dtype` | `uint8` |

Generated defaults contain 37 edges for 36 bins. Explicit `edges` replace `dmin`, `dmax`, and
`num_bins`. Supported dtypes are `uint8`, `uint16`, and `int16`; the encoder rejects a dtype that
cannot hold the configured class range.

`residue_distogram_bins` includes the diagonal by default.
`residue_contact_map_bins` excludes it. With the no-contact class enabled, class zero represents
distances beyond the last edge and excluded diagonal positions.

### Residue voxel grid

| Parameter | Default |
| --- | --- |
| `grid_size` | 64 |
| `voxel_size` | `None` |
| `fit` | `True` |
| `margin` | 4.0 Å |
| `channels` | `occupancy+density+aa_onehot+bfactor` |
| `sigma` | 1.0 voxel |
| `pool` | `"max"` |
| `clamp` | `False` |
| `dtype` | `float32` |

With `fit=True` and `voxel_size=None`, the encoder derives effective spacing and stores it in the
`voxel_size` output while artifact parameters retain the requested `None`. `float16` is supported
as a non-default output dtype.

## Canonical dtypes

| Payload | Dtype |
| --- | --- |
| coordinates and numerical node/edge features | `float32` |
| `edge_index` | `int64` |
| graph statistics | `float32` |
| raw distogram | `float32` |
| contact map | `uint8` by default; `bool` is optional |
| binned matrices | `uint8` by default |
| voxel grid | `float32` by default; `float16` is optional |

## Frozen final-v0.1 defaults not yet implemented

The following targets are frozen for the final v0.1 implementation but do not
claim current runtime support. Their complete scientific definitions, feature
orders, missing-value behavior, and provenance requirements live in
[`structural_features.md`](structural_features.md).

| Setting | Frozen default |
| --- | --- |
| `chain_id` | `None`, meaning all relevant protein/polymer chains in the selected model |
| `model_index` | explicit selected model; no averaging |
| `max_peptide_bond_distance` | `2.0` A, inclusive |
| `backbone_torsion_mode` | `"sincos"` |
| `sidechain_torsion_mode` | `"sincos"` |
| local-frame epsilon | `1e-6` A |
| secondary-structure provider | Biotite 1.6 P-SEA (`biotite==1.6.0`) |
| SASA provider | Biotite 1.6 Shrake-Rupley |
| SASA probe radius | `1.4` A |
| SASA sphere points | `1000`, Fibonacci distribution |
| SASA radii/context | ProtOr; selected-chain polymer heavy atoms; no ions |
| RSA normalization | Tien et al. theoretical maximum ASA; no clipping |
| interface cutoff | `5.0` A minimum inter-chain polymer-heavy-atom distance, inclusive |
| pooling mode | `mean` |
| pooling NaN policy | `omit`, with all-NaN output remaining `NaN` |
| pooling standard deviation | population (`ddof=0`) |
| alternate-location policy | highest finite occupancy; ties blank, then `A`, then lexicographic |

These values are part of canonical configuration and cache identity wherever
they affect scientific output. Atom-level retention of all alternate locations
requires explicit `altloc="all"`; it is not the default.
