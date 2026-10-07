# Eris v0.1 developer handoff

## Start here

Scientific feature development for v0.1 is frozen. This document is the concise
handoff summary; the following permanent contracts are authoritative for
implementation details:

1. [`structural_features.md`](structural_features.md): scientific behavior,
   selection, missing values, and edge cases.
2. [`representation_defaults.md`](representation_defaults.md): canonical
   representation and feature defaults.
3. [`architecture.md`](architecture.md): code map and public data flow.
4. [`artifact_schema.md`](artifact_schema.md): Schema v2 persistence,
   provenance, and identity.
5. [`external_residue_features.md`](external_residue_features.md): external
   feature producer and alignment contract.
6. [`gearnet_edge_backend.md`](gearnet_edge_backend.md): frozen learned
   behavior and checkpoint requirements.
7. [`../DEVELOPMENT.md`](../DEVELOPMENT.md): development environments and
   validation commands.

The executable authorities remain `pyproject.toml`, canonical configuration,
registries, validators, and implementation. This handoff records a development
state, not production or public-release readiness.

## Final implementation state

The repository is on `dev`; the package version is `0.1.0`, distribution
metadata name is `eris-lib`, the import/CLI name is `eris`, and the declared
Python range is `>=3.11,<3.13`. Ten explicit representations, one learned
`gearnet_edge` backend, and `StructureArtifact` Schema v2 are implemented.

| Area | Implemented v0.1 behavior |
| --- | --- |
| Explicit representations | `atom_point_cloud`, `residue_point_cloud`, `atom_knn_graph`, `residue_knn_graph`, `residue_contact_graph`, `residue_distogram`, `residue_distogram_bins`, `residue_contact_map`, `residue_contact_map_bins`, `residue_voxel_grid` |
| Selection and coordinates | Explicit model selection; one or multiple selected polymer chains in canonical model order; CA, residue-atom mean, or center-of-mass residue centers |
| Residue features | Onehot, index, physchem, AAIndex1, raw B-factor, explicitly interpreted pLDDT, flags, aligned external features and hybrid composition |
| Structural geometry | Physical-continuity-aware `phi/psi/omega` and standard residue-specific `chi1–chi4`; internal N-CA-C local frames and directed relative edge direction |
| Other structural features | Same/inter-chain edge identity; inter-chain interface flags/edge annotations; Biotite P-SEA secondary structure; Shrake–Rupley SASA and Tien RSA |
| Pooling and workflows | `mean`, `sum`, `max`, and `mean_std` residue-feature pooling; batch encoding, artifact packing/datasets, Torch export, graph-only PyG export |
| Learned output | GearNet-Edge `node`, `global`, or `both` outputs; frozen to one selected chain in v0.1 |
| Identity and persistence | Schema v2 `meta.json` plus `data.npz`; ordered feature keys; canonical configuration fingerprints, content identity, and provenance |

The detailed residue/atom selection and missing-value policy is centralized in
[`structural_features.md`](structural_features.md). Scientific behavior must
not be expanded or redesigned as part of handoff hardening.

## Definition of Ready

This table audits the final roadmap's P0 checklist against the implemented
repository and WU8A evidence. The roadmap remains an external, untracked
checklist; this table is the permanent handoff summary.

| Checklist area | Status | Evidence or remaining boundary |
| --- | --- | --- |
| Existing explicit clouds, graphs, matrices, and grids | PASS | Registered implementations remain covered by the full suite |
| Onehot, physchem, AAIndex, external and hybrid features | PASS | Feature and integration tests; external alignment and fingerprint coverage |
| B-factor and explicit pLDDT semantics | PASS | Raw B-factor and opt-in pLDDT paths are covered |
| Backbone geometry and residue orientation | PASS | `phi/psi/omega`, `chi1–chi4`, local frames and relative direction implemented |
| Secondary structure and SASA/RSA | PASS | Frozen Biotite P-SEA and Shrake–Rupley/Tien contracts implemented and tested |
| Side-chain torsions (roadmap P1 target) | PASS | Standard residue-specific chi definitions are implemented; no rotamer classification or side-chain reconstruction |
| Distance, RBF, sequence separation, relative direction, chain relations | PASS | Edge features and rigid-transform behavior covered |
| Multichain identity, inter-chain edges, interfaces | PASS | Canonical chain order, chain-qualified IDs, edge relations, and interface features implemented |
| `mean`, `sum`, `max`, `mean_std` pooling | PASS | Public `eris.pool()` and persisted-vector artifact contract implemented |
| Missing atoms, altloc, insertion codes, noncanonical residues, breaks, models | PASS | Frozen policies and regression coverage; see structural edge-case table in the contract |
| Translation/rotation expectations and validation panel | PASS | WU8A focused regression tests and existing structural fixtures; SASA/RSA uses numerical tolerance |
| Feature, architecture, provenance, limitations, handoff documentation | PASS | Updated permanent contracts and this handoff |
| GearNet node/global execution against official checkpoint | known limitation | Checkpoint-dependent tests skipped because `ERIS_GEARNET_CHECKPOINT` was not configured |
| Native PyG export in this environment | known limitation | `torch_geometric` was unavailable; existing fake-module conversion tests are not native certification |
| CI, clean-install, supported-environment and release certification | post-handoff item | Required before release; not a scientific implementation blocker |

## WU8A validation

Final WU8A result: **463 passed, 4 skipped, 0 failed**. The four skips are
checkpoint-dependent GearNet tests and require `ERIS_GEARNET_CHECKPOINT`.
There was no production defect found during WU8A; changes for that work were
validation tests.

The WU8A regression additions/updates are in:

- [`../tests/test_explicit_geometry.py`](../tests/test_explicit_geometry.py):
  rigid transforms for geometric matrices and experimental monomer/multimer
  fixture coverage.
- [`../tests/test_explicit_graphs.py`](../tests/test_explicit_graphs.py):
  graph/RBF transform behavior and multichain edge/interface integration.
- [`../tests/test_residue_features.py`](../tests/test_residue_features.py):
  rich feature composition, pooling, artifact round-trip, external identity,
  interface/SASA behavior, and structural cases.
- [`../tests/test_packaging_boundaries.py`](../tests/test_packaging_boundaries.py):
  lazy optional-import boundary coverage.

The validation panel uses existing experimental structure fixtures for a
monomer and multimer, supplemented by focused generated structures for
controlled geometry and missing-atom, alternate-location, insertion-code,
noncanonical, chain, and interface/context behavior. It is a small regression
panel, not broad independent biological certification.

Rigid-transform checks cover expected invariance/equivariance for distance and
contact matrices, distograms, residue graph topology, distance/RBF edge
features, backbone and side-chain torsions, relative direction, interface
detection, secondary structure, AAIndex, physicochemical features, and pooling
of non-coordinate features. Cartesian coordinate arrays are not expected to
remain invariant.

WU8A checks passed Ruff lint, `pip check`, and `git diff --check`. Ruff format
passed for changed test files. The project-wide format check still reports five
existing notebooks as unformatted; they were intentionally not changed. The
test run emitted a non-fatal pytest cache permission warning in the local
environment.

## Dependency boundaries and known limitations

- `import eris` does not eagerly import Torch, TorchDrug, PyG, RDKit, or GearNet
  numerical code. Feature-local providers are loaded on use. No checkpoint is
  bundled. GearNet can acquire only its registered official HTTPS checkpoint,
  verifies the registered SHA-256, and otherwise supports an explicit local
  checkpoint for offline use.
- GearNet remains single-chain in v0.1. Official-checkpoint certification
  remains post-handoff; the checkpoint-dependent tests were skipped in WU8A.
- Native PyG export was not exercised in the WU8A environment because
  `torch_geometric` was unavailable.
- SASA/RSA uses a deterministic 1,000-point finite Fibonacci sphere. Its
  finite sampling has a small expected orientation-dependent approximation;
  WU8A observed about 0.107% total-SASA change on 1AQ2 after rotation and used
  scientifically justified numerical tolerances rather than requiring exact
  per-residue invariance. This is not a P0 scientific blocker.
- External features require exact canonical residue-ID alignment; sequence
  alignment is not implemented.
- GearNet, PyG, clean installations, and release
  artifacts have not been certified across the supported environment matrix.

## Post-handoff validation, hardening, and release work

These are separate from the frozen scientific implementation. Do not implement
them as feature work under this handoff:

1. Obtain independent scientific/user validation on representative PDB and
   mmCIF inputs and independently verify edge-case semantics.
2. Validate Python 3.11 and 3.12 in clean supported environments.
3. Repair and harden CI, then run the supported Python/dependency matrix.
4. Build and inspect wheel and sdist; test clean installation and the full suite
   against installed distributions.
5. Freeze dependency policy and public API/distribution metadata.
6. Install native PyG and certify export in supported environments.
7. Certify registered GearNet checkpoint acquisition, cache reuse,
   node/global/both outputs, frozen references, and repeatability across the
   supported matrix.
8. Finalize release metadata, `CHANGELOG`/release notes, and `CITATION.cff`.
9. Only after those gates, plan PyPI publication, GitHub Release, and Zenodo
   archival.

**Handoff classification:** the scientific feature implementation is ready for
developer handoff, with the stated validation limitations. This is not a
production or public-release readiness claim.
