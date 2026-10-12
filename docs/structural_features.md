# Structural feature contracts

This document is the authoritative scientific and persistence contract for the
Eris structural-feature implementation. The contracts are implemented; they are
not a roadmap of planned behavior.

The existing architecture remains authoritative: public APIs build canonical
configuration, resolve registered implementations, run feature-local optional
dependencies lazily, and persist immutable `StructureArtifact` objects. The
CLI must delegate to the same public Python APIs. These contracts do not expose
registries, runners, backends, cache internals, or third-party provider objects
as public API.

## Contract status

All contract groups of the structural-feature implementation are frozen and
implemented.

| Contract group | Status | Frozen decision |
| --- | --- | --- |
| Multi-chain selection and identity | FROZEN | Canonical model-order chain tuple; `None` means all relevant protein/polymer chains |
| Polymer continuity and backbone torsions | FROZEN | Physical peptide-bond continuity; signed radians; fixed missing-value behavior |
| Side-chain torsions | FROZEN | Standard chi definitions through chi4; fixed-width outputs |
| Local frames and relative direction | FROZEN | Right-handed N-CA-C frame; directed source-frame displacement |
| Secondary structure | FROZEN | Biotite P-SEA three-state assignment plus unassigned |
| SASA and RSA | FROZEN | Biotite Shrake-Rupley SASA and Tien maximum-ASA normalization |
| Interface features | FROZEN | Selected-chain polymer-heavy-atom contacts at 5 A; no topology mutation |
| Structure-level pooling | FROZEN | Immutable vector artifact; four modes; explicit NaN policy |
| Parsing and missing-data policy | FROZEN | Deterministic altloc selection; retained polymer noncanonicals; explicit model |

## Selection, identity, and ordering

### Model selection

Exactly one coordinate model is selected by an explicit zero-based
`model_index`. There is no model averaging. An out-of-range model index fails
clearly. The selected model index is part of canonical configuration,
provenance, the configuration fingerprint, and scientific cache identity.

### Chain selection

The public `chain_id` argument accepts:

- `None`: all relevant protein/polymer chains in the selected model;
- one string such as `"A"`: exactly one chain; or
- a sequence such as `["A", "B"]`: exactly those chains.

This is an intentional change from the pre-freeze behavior in which `None`
selected the first chain. A relevant chain is a chain containing at least one
polymer amino-acid residue according to the parser's polymer/entity
classification. Water-only, ligand-only, and free-ion chains are not relevant.
Noncanonical polymer amino acids do make a chain relevant.

The selection is resolved before cache lookup. Its canonical internal form is
an ordered tuple of unique chain identifiers in their order in the selected
coordinate model. JSON metadata serializes it as a list; NPZ fields serialize
it as a one-dimensional Unicode array. Caller list order never changes
scientific identity. An empty selection, a duplicate identifier, or an unknown
identifier fails clearly; duplicates are not silently deduplicated.

The resolved tuple is recorded as `input.chain_ids`, in canonical selection
parameters, in the configuration fingerprint, and in cache identity. A single
chain is represented as a one-element collection everywhere; there is no
single-chain scalar special case.

No biological-assembly generation is performed. Selection is limited to
chains present in the explicitly selected coordinate model.

### Residue identity and order

Residue identifiers retain the exact form:

```text
<chain>:<number>:<insertion>:<resname>
```

Consequently `A:42::GLY` and `B:42::GLY` are different nodes, as are residue
number 42 and insertion variants 42A and 42B within one chain.

Every residue-level artifact carries a one-dimensional `chain_id` string
array aligned with its nodes or rows. Residues are ordered first by chain order
in the selected model and then by structural residue order within each chain.
Exact residue names are preserved.

Residue-graph nodes are polymer amino-acid residues only. Waters, ligands,
cofactors, and free ions are excluded. Atom-level representations continue to
apply their explicitly configured atom filters.

### External residue features

For external residue-feature input, `residue_ids` remain authoritative.
Required `chain_ids` is a one-dimensional Unicode array containing the
selected chains in canonical model order. Its values must be unique and must
exactly match the resolved selected-chain tuple. A single-chain file therefore
contains, for example, `chain_ids = ["A"]`. The obsolete scalar `chain_id`
field is rejected. Array rows are aligned by exact residue ID as specified in
[`external_residue_features.md`](external_residue_features.md).

## Polymer continuity and chain breaks

Polymer continuity is a scientific relation, not a residue-numbering
heuristic. Residue `i` is peptide-connected to residue `i+1` only when all of
the following hold:

1. both are consecutive polymer residues in structural order within the same
   selected chain;
2. `C(i)` and `N(i+1)` are present after alternate-location selection; and
3. `distance(C(i), N(i+1)) <= max_peptide_bond_distance`.

`max_peptide_bond_distance` defaults to `2.0` A and is part of feature
parameters, provenance, the configuration fingerprint, and cache identity.
Residue numbers, insertion codes, and adjacency in an input table do not by
themselves establish continuity.

A failed continuity test begins a new segment. Backbone torsions and local
sequential relations never cross segments or chains. Spatial, radius, contact,
or k-nearest-neighbor graph edges may cross a physical break when their own
topology contract selects them; doing so does not restore polymer continuity.

For an edge between different chains:

- `seq_sep` is `-1`;
- `flag.is_local_le_4` is `0`;
- `edge.same_chain` is `0`; and
- `edge.inter_chain` is `1`.

For residues in different continuity segments of the same chain, `seq_sep`
retains the existing nonnegative chain-relative sequence separation, while
`flag.is_local_le_4` is forced to `0`, `edge.same_chain = 1`, and
`edge.inter_chain = 0`. The existing same-chain separation preserves numeric
residue-number gaps and treats insertion-code variants in structural order as
adjacent. The continuity relation, not the magnitude of `seq_sep`, controls
whether an edge is locally sequential. `edge.same_chain` and
`edge.inter_chain` are exact binary complements.

## Backbone torsions

Backbone dihedrals use selected coordinates before any representation-level
centering or scaling:

- `phi(i) = dihedral(C(i-1), N(i), CA(i), C(i))`;
- `psi(i) = dihedral(N(i), CA(i), C(i), N(i+1))`; and
- `omega(i) = dihedral(CA(i), C(i), N(i+1), CA(i+1))`.

The dihedral is the standard signed `atan2` result in radians, normalized to
`[-pi, pi)`. `phi` requires continuity from the previous residue; `psi` and
`omega` require continuity to the next residue. A missing required atom,
broken continuity, or degenerate dihedral produces `NaN`, never an inferred or
zero value.

`backbone_torsion_mode` is one of:

- `"sincos"` (default): `backbone.phi_sin`, `backbone.phi_cos`,
  `backbone.psi_sin`, `backbone.psi_cos`, `backbone.omega_sin`,
  `backbone.omega_cos`;
- `"raw"`: `backbone.phi`, `backbone.psi`, `backbone.omega`; or
- `"both"`: the six sine/cosine keys followed by the three raw keys.

When an angle is undefined, every value derived from it is `NaN`. The mode,
continuity cutoff, key order, units, and convention are recorded in canonical
feature parameters and provenance.

The implemented residue feature token is `backbone_geometry`. Its canonical
parameters are `mode`, `max_peptide_bond_distance`, `torsion_convention`,
`units`, and `missing_value_policy`. The latter three record the fixed signed
`atan2` convention, radians, and `NaN` policy rather than offering competing
scientific behaviors.

## Side-chain torsions

Side-chain torsions use the following standard atom definitions. Definitions
not listed for a residue are structurally absent, not zero.

| Residue | chi1 | chi2 | chi3 | chi4 |
| --- | --- | --- | --- | --- |
| ARG | N-CA-CB-CG | CA-CB-CG-CD | CB-CG-CD-NE | CG-CD-NE-CZ |
| ASN | N-CA-CB-CG | CA-CB-CG-OD1 | - | - |
| ASP | N-CA-CB-CG | CA-CB-CG-OD1 | - | - |
| CYS | N-CA-CB-SG | - | - | - |
| GLN | N-CA-CB-CG | CA-CB-CG-CD | CB-CG-CD-OE1 | - |
| GLU | N-CA-CB-CG | CA-CB-CG-CD | CB-CG-CD-OE1 | - |
| HIS | N-CA-CB-CG | CA-CB-CG-ND1 | - | - |
| ILE | N-CA-CB-CG1 | CA-CB-CG1-CD1 | - | - |
| LEU | N-CA-CB-CG | CA-CB-CG-CD1 | - | - |
| LYS | N-CA-CB-CG | CA-CB-CG-CD | CB-CG-CD-CE | CG-CD-CE-NZ |
| MET | N-CA-CB-CG | CA-CB-CG-SD | CB-CG-SD-CE | - |
| PHE | N-CA-CB-CG | CA-CB-CG-CD1 | - | - |
| PRO | N-CA-CB-CG | CA-CB-CG-CD | - | - |
| SER | N-CA-CB-OG | - | - | - |
| THR | N-CA-CB-OG1 | - | - | - |
| TRP | N-CA-CB-CG | CA-CB-CG-CD1 | - | - |
| TYR | N-CA-CB-CG | CA-CB-CG-CD1 | - | - |
| VAL | N-CA-CB-CG1 | - | - | - |

ALA and GLY have no chi angles. Canonical residues always occupy the fixed
chi1-through-chi4 schema. A chemically absent angle, a missing required atom,
an unsupported exact residue name, or a degenerate dihedral is `NaN`.

`sidechain_torsion_mode` is one of:

- `"sincos"` (default): `sidechain.chi1_sin`, `sidechain.chi1_cos`, through
  `sidechain.chi4_sin`, `sidechain.chi4_cos`;
- `"raw"`: `sidechain.chi1`, `sidechain.chi2`, `sidechain.chi3`,
  `sidechain.chi4`; or
- `"both"`: the eight sine/cosine keys followed by the four raw keys.

Raw angles use the same signed-radian convention as backbone torsions.
No rotamer classification, residue rebuilding, symmetry remapping, or arbitrary
noncanonical-to-canonical mapping is performed.

## Local frames and relative direction

For a residue with finite `N`, `CA`, and `C` coordinates, define a local
right-handed frame with origin at `CA`:

```text
x = normalize(C - CA)
y = normalize((N - CA) - dot(N - CA, x) * x)
z = normalize(cross(x, y))
R = [x y z]
```

The vectors are columns of `R` in global coordinates. A norm below
`frame_epsilon = 1e-6 A`, a non-finite coordinate, or a missing required atom
invalidates the frame. Invalid frames are stored as `NaN` with a zero validity
mask; valid frames have a one mask. The epsilon, atom convention, handedness,
and matrix layout are frozen provenance.

Under a rigid global rotation the frame rotates equivariantly, while vectors
expressed through `R.T` remain invariant to that global rotation.

For each directed residue-graph edge `i -> j`, let
`u_ij = normalize(CA(j) - CA(i))`. Relative direction is
`R_i.T @ u_ij`, emitted in order as:

- `edge.relative_direction_x`;
- `edge.relative_direction_y`; and
- `edge.relative_direction_z`.

The reverse edge is computed independently in the reverse source frame. A
missing CA, zero-length displacement, non-finite coordinate, or invalid source
frame produces three `NaN` values. This feature is defined only for
residue-level graph edges and neither exposes nor reconstructs GearNet-internal
topology.

The implemented residue-edge tokens are `relative_direction`, `same_chain`,
and `inter_chain`. They are opt-in for residue kNN and contact graphs and use
the edge-feature registry's deterministic physical ordering. Atom graphs
reject these residue-only tokens. Local frames remain internal infrastructure:
Eris does not expose their global axes as node attributes or persist
`local_frame` arrays.

Implementations may persist aligned auxiliary arrays `local_frame` with shape
`(N, 3, 3)` and `local_frame_valid` with shape `(N,)`; when persisted, their
roles and exact shapes must be declared. The three relative-direction values
are ordinary ordered edge attributes.

## Secondary structure

The selected provider is Biotite's `annotate_sse`, a P-SEA-style assignment
based on CA geometry. It is an optional, feature-local dependency and must be
imported lazily; requesting unrelated Eris functionality must not import it.
There is no external executable and no network access. It is selected because
it implements a published method without making Torch, GearNet, or a separate
system executable mandatory.

Assignments use each selected chain and continuity segment independently so a
physical break cannot create a secondary-structure assignment across the gap.
The public categorical states are:

- `secondary_structure.helix`;
- `secondary_structure.sheet`;
- `secondary_structure.coil`; and
- `secondary_structure.unassigned`.

Provider states `a`, `b`, and `c` map to helix, sheet, and coil respectively.
Missing/invalid CA input or an empty provider assignment maps to unassigned.
Categorical encoding order, provider package/version, provider algorithm,
segment handling, and mapping are recorded in provenance. Secondary structure
is not silently substituted with DSSP or another algorithm.

The dependency is `biotite>=1.6,<2`. The installed Biotite version is recorded as
`provider_version` in the feature parameters, so it is part of the config fingerprint and cache
identity: artifacts computed with different Biotite releases never share a cache entry. Biotite
documents the P-SEA basis, CA-only inputs, state mapping, and deviations from the original
program in its
[`annotate_sse` reference](https://www.biotite-python.org/latest/apidoc/biotite.structure.annotate_sse.html).
The dependency is declared in package metadata and remains lazily imported by these feature
paths.

## SASA and RSA

The selected provider is Biotite's Shrake-Rupley implementation with:

- probe radius `1.4` A;
- `1000` sphere points;
- Fibonacci point distribution;
- ProtOr van der Waals radii;
- hydrogen atoms excluded; and
- ions excluded.

These parameters match the documented Biotite
[`sasa` interface](https://www.biotite-python.org/latest/apidoc/biotite.structure.sasa.html),
which also identifies ProtOr as the hydrogen-free crystal-structure radii set.

The calculation context is exactly the selected chains' polymer heavy atoms in
the selected model after deterministic alternate-location resolution. Thus a
multi-chain selection measures the selected complex, while a single-chain
selection measures that selected chain alone. Waters, ligands, cofactors,
unselected chains, and free ions are neither targets nor occluders.

Per-atom areas are summed by residue to produce `sasa` in square angstroms.
For a canonical residue known to be incomplete after parsing, its own `sasa`
and `rsa` are `NaN`; atoms that are present still participate as occluders for
other residues. If the provider cannot assign a required radius to any atom in
the selected context, the request fails clearly rather than dropping that atom.
A noncanonical polymer residue may have `sasa` when all its present atoms have
supported radii, but its `rsa` is `NaN` unless its exact residue name has an
explicit normalization reference.

`rsa` is `sasa / maximum_accessible_area` using the Tien et al.
[theoretical maximum-ASA table](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0080635)
for the 20 canonical amino acids. The frozen values in square angstroms are:

| Residue | ALA | ARG | ASN | ASP | CYS | GLN | GLU | GLY | HIS | ILE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maximum ASA | 129 | 274 | 195 | 193 | 167 | 225 | 223 | 104 | 224 | 197 |

| Residue | LEU | LYS | MET | PHE | PRO | SER | THR | TRP | TYR | VAL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maximum ASA | 201 | 236 | 224 | 240 | 159 | 155 | 172 | 285 | 263 | 174 |

Values are not clipped to `[0, 1]`. The exact reference table and citation,
provider package/version, probe, point count/distribution, radii set,
hydrogen/ion handling, calculation context, residue-completeness rule, and
units are canonical parameters or provenance as appropriate. Biotite is
selected here because the same lazily loaded, supported provider implements the
published Shrake-Rupley method without an external executable or Torch.

## Inter-chain interface features

An interface contact exists between residues in different selected chains when
the minimum distance between any of their selected polymer heavy atoms is at
most `5.0` A. The cutoff is inclusive. Hydrogens, waters, ligands, cofactors,
and free ions are excluded. A residue with no usable heavy atom cannot be an
interface residue.

The derived outputs are:

- a binary residue attribute `interface.is_interface`;
- a binary existing-edge attribute `edge.interface_contact`; and
- an ordered `interface_residue_ids` subset in ordinary residue order.

The residue flag and subset are computed over all selected inter-chain residue
pairs. The edge flag is `1` only when an edge already present in the public
residue graph joins a contacting pair. Interface computation never creates,
deletes, or rewires representation topology. It remains distinct from
sequential, radius, contact, and k-nearest-neighbor topology construction.

The cutoff, atom/context filters, inclusivity, alternate-location policy, and
selected chain tuple are part of canonical parameters and provenance. This is
not a docking interface, biological-assembly inference, or water-mediated
contact definition.

The implemented paired feature tokens are residue-node `interface` and
residue-edge `interface_contact`. They must use the same cutoff. Enabling the
pair also emits `interface_residue_ids`; default graph feature modes remain
unchanged.

## Structure-level pooling

Pooling is a dedicated public operation:

```python
eris.pool(
    artifact,
    source="node_attr",
    mode="mean",
    nan_policy="omit",
)
```

It consumes a
persisted or in-memory structure artifact and produces a new immutable
structure-level artifact; it never mutates the source artifact.

Supported modes and ordered output keys are:

- `mean`: `pool.mean.<key>`;
- `sum`: `pool.sum.<key>`;
- `max`: `pool.max.<key>`; and
- `mean_std`: all `pool.mean.<key>` values followed by all
  `pool.std.<key>` values.

The source is one explicitly named residue-aligned numeric array of shape
`(N, D)` plus its `D` ordered feature keys, with `N > 0`. Pooling across
ambiguous arrays or inventing feature names is not allowed. Reduction is
column-wise across the source's node axis. Output length is `D` except for
`mean_std`, whose output length is `2D`.

`nan_policy` is one of:

- `omit` (default): ignore NaNs, but any all-NaN column remains NaN for every
  mode, including `sum`;
- `propagate`: a NaN in a column makes that pooled column NaN; or
- `error`: any NaN fails clearly.

Standard deviation uses population normalization (`ddof=0`). Empty input fails
clearly. Positive and negative infinity are not treated as missing values and
must already satisfy the source artifact's finite-value contract.

The pooled artifact uses:

- `representation_name = "residue_pool"`;
- `representation_family = "explicit"`;
- `representation_type = "vector"`;
- `granularity = "structure"`;
- `pooled_attr` with shape `(F,)` and role `pooled_features`; and
- `feature_keys.pooled_attr_keys` aligned one-to-one with `pooled_attr`.

The coordinate source is copied from the source artifact. Canonical pooling
parameters include the mode, NaN policy, source array name, source feature keys
and their order, and a source-artifact fingerprint. That fingerprint covers
the source schema version, representation identity, encoder identity, input
source hash/model/chain selection, source configuration fingerprint, selected
array name, ordered source keys, and selected array dtype, shape, and canonical
byte-content hash. This makes pooling scientifically cache-safe without
changing the source artifact.

The implemented public operation is `eris.pool()`. It emits `float64`
`pooled_attr`, supports in-memory artifacts and persisted artifact directories,
and can optionally save the new artifact. Torch export preserves the vector as
`pooled_attr`; pooling is not an embedding operation.

## Parsing and missing-data policy

### Alternate locations

The default is one conformer per atom name, chosen by highest finite occupancy.
Occupancy ties are resolved by blank alternate-location identifier first,
then `A`, then lexicographic identifier order. A missing or non-finite occupancy
ranks below every finite occupancy and participates in the same deterministic
identifier tie-break. Input record order is never the scientific tie-break.

Atom-level representations may retain alternate conformers only through an
explicit `altloc="all"` mode. An explicit conformer identifier remains a valid
selection mode. The resolved policy is part of configuration, provenance,
fingerprinting, and cache identity.

### Noncanonical polymer residues

Noncanonical polymer amino-acid residues remain nodes with their exact residue
names and identifiers. Geometry that is well-defined from present atoms is
computed. Sequence-derived categorical properties are unknown/unassigned, and
canonical-residue-only normalization or chi definitions are `NaN`. Eris never
silently maps a noncanonical residue to a canonical amino acid.

### Missing atoms and invalid geometry

Missing center atoms obey the representation's explicit `skip` or `error`
policy because they determine whether a sample/node can exist. Optional numeric
features use `NaN`; optional categorical features use their documented
unassigned state. No silent padding, truncation, coordinate reconstruction,
residue rebuilding, topology inference, or replacement with zero is allowed.

The edge-case policy is summarized here; feature-specific details above remain
authoritative.

| Case | Frozen behavior |
| --- | --- |
| Missing CA | Residue centers follow the configured `skip` or `error` policy. Features requiring CA use their documented missing value; secondary structure is unassigned. |
| Missing N or C | Backbone torsions or N-CA-C local frames that require the atom are undefined (`NaN` values/invalid frame); nodes are not removed for optional geometry. |
| Incomplete side chain | Each chemically defined chi angle with a missing required atom is `NaN`; no side-chain reconstruction is performed. |
| Alternate locations | Residue atoms use the deterministic maximum-finite-occupancy policy and tie-break defined above. Atom representations retain alternatives only by explicit configuration. |
| Insertion codes | Included in exact residue identity and preserved in structural order; numbering alone does not establish peptide continuity. |
| Noncanonical residues | Parser-classified polymer amino acids remain with exact names/IDs. Canonical-only chi definitions and normalization are `NaN`; no residue remapping occurs. |
| Heteroatoms, water, ions, ligands, cofactors | Not residue-graph nodes or interface/SASA occluders. Atom representations follow their explicit atom filters; polymer-classified noncanonical amino acids are retained as residues. |
| Multiple models | Exactly one explicit zero-based `model_index` is selected; models are not averaged. |
| Physical chain breaks | Continuity requires consecutive polymer residues and a present C-N distance within the configured inclusive cutoff; sequential torsions/relations do not cross a break. Spatial graph edges may still cross it. |
| Multiple selected chains | Chains are unique, resolved in model order, and have chain-qualified residue IDs. Chain-local continuity never crosses chains; spatial edges and interface analysis may span selected chains. |

## Artifact and schema assessment

The following records the artifact schema contract. Validators, serializers,
fixtures, and contract tests use this one canonical contract.

| Capability | Artifact schema contract | Data/key contract |
| --- | --- | --- |
| Multi-chain identity | Canonical `input.chain_ids`, selection parameters, and provenance | Aligned residue `chain_id` |
| External multi-chain features | Canonical chain/model identity and content fingerprint | External `chain_ids` and aligned feature keys |
| Backbone and side-chain torsions | Modes, atom convention, units, continuity/missing policy, definitions, and provenance | Fixed ordered node-feature keys, including chi1–chi4 |
| Local frames/directions | Frame convention, epsilon, and missing policy | Optional declared frame/mask arrays; ordered edge keys |
| Secondary structure | Provider, version, algorithm, segment handling, and state mapping | Ordered node-feature keys |
| SASA/RSA | Calculation context, algorithm, reference table, and units | Ordered node-feature keys |
| Interface features | Contact definition, cutoff, selection, and context | Node/edge keys and ordered residue-ID subset |
| Structure pooling | Mode, NaN policy, source identity, and fingerprint | `pooled_attr` plus ordered keys |
| Altloc/noncanonical/model policy | Canonical selection parameters and provenance | No new required array |

Arrays use explicit dtypes and exact shapes. String arrays are Unicode, not
object arrays. Feature key order remains scientifically significant. The
serializer must not pickle. Unknown required roles, types, shapes, enum values,
or contradictory single-/multi-chain metadata fail validation clearly.

## Implementation boundaries

These contracts do not change GearNet scientific behavior or expose
GearNet-internal topology. Learned embeddings remain unable to reveal or
reconstruct internal graph edges. PyG export may use only topology already
public in a source artifact.

Every implementation must reuse canonical configuration, validators,
serializers, hashes, and registries. Provider imports stay lazy and
feature-local. Runtime network access is forbidden except for automatic
acquisition of official model checkpoints whose HTTPS URL and SHA-256 are
registered by Eris; arbitrary user-supplied download URLs are forbidden. No
parallel permanent schema, silent fallback, or cache-identity shortcut
is permitted.

All listed behaviors are implemented. Future changes to scientific behavior
must be treated as explicit contract changes.
