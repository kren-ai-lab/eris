# Explicit representation defaults

`eris/core/config.py` is the executable authority for these defaults. Registry discovery,
artifact parameters, and configuration fingerprints derive from the same typed configs.

## Common geometry and selection

All distances are in ångström. `center=True` subtracts one centroid before representation
construction; `scale=None` preserves ångström-based geometry.

| Scope | Parameter | Default | Meaning |
| --- | --- | --- | --- |
| all | `center` | `True` | center coordinates once |
| all | `scale` | `None` | optional positive coordinate divisor |
| residue | `residue_center` | `"ca"` | alpha-carbon coordinate |
| residue | `amino_acids_only` | `True` | parser-classified polymer amino acids, including noncanonical residues |
| residue | `altloc` | `"max_occupancy"` | highest finite occupancy; ties blank, `A`, then lexicographic |
| residue | `missing_residue_center` | `"skip"` | omit and record missing centers; `"error"` is available |
| residue | `plddt_source` | `"none"` | use `"bfactor"` only for inputs known to store pLDDT in CA B-factor |
| atom | `include_hetero` | `False` | exclude hetero residues |
| atom | `include_h` | `False` | exclude hydrogen atoms |
| atom | `drop_water` | `True` | remove water |
| atom | `keep_ions` | `True` | retain ions when otherwise selected |
| atom | `altloc` | `"max_occupancy"` | highest finite occupancy; ties blank, `A`, then lexicographic |
| atom | `amino_acids_only` | `False` | do not restrict atom extraction to amino acids |

Atom representations also support explicit nondefault alternate-location modes. `first` selects
the first encountered conformer per atom name, `all` retains every alternate-conformer atom
record, and an identifier such as `A` requests that conformer using the documented fallback.

Residue coordinate sources in artifacts are `ca`, `residue_mean`, and `residue_com` for
`residue_center="ca"`, `"mean"`, and `"com"`. Mean uses selected residue atoms; center of mass
uses their element atomic masses and fails when an element has no valid mass. With
`missing_residue_center="skip"`, omitted identities are stored in
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
| `backbone_geometry` | physical-continuity-aware phi, psi, and omega torsions |
| `sidechain_geometry` | standard residue-specific chi1–chi4 torsions |
| `external` | aligned external numerical values |
| `interface` | opt-in inter-chain polymer-heavy-atom interface flag for residue graphs |
| `secondary_structure` | Biotite P-SEA helix/sheet/coil/unassigned one-hot block |
| `sasa` | Biotite Shrake-Rupley residue SASA in square angstroms |
| `rsa` | SASA divided by the Tien theoretical maximum-ASA reference |

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
`backbone_geometry` defaults to `mode="sincos"`; `raw` emits signed radians and `both` emits
the six periodic columns followed by the three raw columns. Undefined values remain `NaN`, and
physical continuity uses an inclusive `max_peptide_bond_distance=2.0` A cutoff.
`sidechain_geometry` uses the same `sincos` (default), `raw`, and `both` modes with a fixed
chi1–chi4 schema. Chemically undefined angles, missing atoms, unsupported residues, and
degenerate geometry remain `NaN`; side chains are not reconstructed.

`secondary_structure` runs Biotite P-SEA independently for each selected chain and physical
continuity segment. Its four columns are ordered helix, sheet, coil, and unassigned. `sasa` and
`rsa` share one selected-complex Shrake-Rupley calculation over selected polymer heavy atoms with
a 1.4 A probe, 1000 Fibonacci points, and ProtOr radii. Incomplete canonical residues receive
`NaN`; present atoms still occlude other residues. RSA uses the frozen Tien table and is not
clipped. Biotite is loaded only when one of these features executes.

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

Registered edge-feature tokens are:

| Token | Semantics |
| --- | --- |
| `dist` | edge distance |
| `rbf` | radial-basis expansion of distance |
| `seq` | chain-relative sequence separation, with `-1` across chains |
| `local` | sequence-local flag |
| `same_chain` | `edge.same_chain`, the binary chain-identity relation |
| `inter_chain` | `edge.inter_chain`, the exact complement of `same_chain` |
| `relative_direction` | directed unit CA displacement in the source N-CA-C frame |
| `interface_contact` | whether an existing edge joins an inter-chain interface-contact pair |

The last four tokens are residue-graph-only and opt-in, so the default edge width remains 19.
`relative_direction` records the frozen right-handed frame convention, `1e-6` A degeneracy
tolerance, source-frame convention, and `NaN` missing-frame policy in canonical parameters.
Frames are internal and are not emitted as global-axis node attributes.
The `interface` and `interface_contact` tokens are enabled as a matched pair with the same
configurable cutoff, defaulting to an inclusive `5.0` A minimum selected polymer-heavy-atom
distance. They add `interface.is_interface`, `edge.interface_contact`, and the ordered
`interface_residue_ids` subset without changing graph topology or the contact-graph cutoff.

For residue graphs, sequence separation is the absolute difference between chain-relative
biological positions. Numeric residue-number gaps are retained; insertion-code residues advance in
source order. Cross-chain spatial edges retain `seq_sep=-1` and a zero locality flag.

### Structure-level pooling

`eris.pool(artifact, source="node_attr", mode="mean", nan_policy="omit")` creates a new
`residue_pool` vector artifact with `float64` `pooled_attr`. Modes are `mean`, `sum`, `max`, and
`mean_std`; the last uses population standard deviation and emits all mean keys before all
standard-deviation keys. NaN policies are `omit`, `propagate`, and `error`. Under `omit`, reduction
is column-wise and an all-NaN column remains NaN, including for `sum`.

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

## Structural defaults

The multi-chain, selected-model, alternate-location, physical-continuity, backbone/side-chain
torsion, local-frame, secondary-structure, solvent-accessibility, interface, and pooling behavior
below is implemented. Complete scientific definitions, feature orders, missing-value behavior,
and provenance requirements live in
[`structural_features.md`](structural_features.md).

| Setting | Frozen default |
| --- | --- |
| `chain_id` | `None`, meaning all relevant protein/polymer chains in the selected model |
| `model_index` | explicit selected model; no averaging |
| `max_peptide_bond_distance` | `2.0` A, inclusive |
| `backbone_torsion_mode` | `"sincos"` |
| `sidechain_torsion_mode` | `"sincos"` |
| local-frame epsilon | `1e-6` A |
| secondary-structure provider | Biotite P-SEA (`biotite>=1.6,<2`; installed version recorded) |
| SASA provider | Biotite Shrake-Rupley |
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
