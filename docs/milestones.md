# Eris v0.1 Milestones

This document is the authoritative internal tracker for Eris v0.1 development. GitHub
Milestones are not an operational requirement. Historical partial code does not complete a
work unit or milestone: completion requires its acceptance criteria and milestone gate to pass.

Tracking conventions:

- `[ ]` means pending.
- `[x]` means completed.
- Only one milestone is active unless an explicitly recorded external task runs in parallel.
- Each work unit should close through one coherent, testable, auditable functional commit.
- No milestone may depend on a later milestone.
- Target contracts listed here are not claims that the implementation already exists.

## M0 — Project freeze and release contract

**Status:** `DONE`

**Objective:** Freeze the architectural, naming, operating, tracking, and release direction for
v0.1.

### Completed internal work

- [x] **M0.1 — Freeze branch policy.** `jr_revision` is the frozen historical baseline and
  `dev` is the only active v0.1 development line.
- [x] **M0.2 — Freeze v0.1 scope.** v0.1 is consolidation and release-hardening, not a redesign
  or expansion into new representation families.
- [x] **M0.3 — Freeze framework boundaries.** Sylphy owns learned sequence/molecular encoders,
  Roxy owns engineered descriptors, Eris owns 3D structural representations, and Rudeus owns
  EDA/visualization.
- [x] **M0.4 — Freeze representation families.** Eris contains first-class `explicit` and
  `learned` structural representations.
- [x] **M0.5 — Freeze canonical explicit representation names.** The ten public identifiers are
  recorded in `docs/architecture.md`.
- [x] **M0.6 — Separate granularity from coordinate source.** In particular,
  `granularity = residue` and `coordinate_source = ca` are distinct concepts.
- [x] **M0.7 — Freeze conceptual Artifact Schema v2.** Schema v2 is the first stable public
  artifact contract; Schema v1 remains pre-release.
- [x] **M0.8 — Freeze core Python and dependency policy.** Core targets Python 3.10–3.12 and is
  independent of Roxy, Sylphy, Torch, PyG, and TorchDrug/GearNet dependencies.
- [x] **M0.9 — Freeze configuration authority.** Typed canonical configuration will drive
  defaults, Registry metadata, CLI help, artifact parameters, fingerprints, and documentation.
- [x] **M0.10 — Freeze target Python API and CLI.** The target hierarchy is recorded in
  `docs/architecture.md` and is not presented as current behavior before implementation.
- [x] **M0.11 — Document the M1–M9 roadmap.** This file records work units, gates,
  dependencies, and decision owners.

### Completed distribution-name decision

- [x] **M0.X — Freeze distribution name as `eris`**
  - **Frozen identity:** Project **Eris**; repository **Eris**; Python import `eris`; CLI command
    `eris`; desired and frozen distribution name `eris`.
  - **Acceptance:** The project owner has frozen `eris` as the distribution name for normal v0.1
    development. No alternative distribution name is considered or adopted automatically.
  - **Depends on:** Explicit project-owner decision; completed.

### External PyPI release condition

Ownership and availability of the PyPI `eris` namespace must be verified before publication.
This external feasibility check does not keep M0 open and does not reopen the frozen naming
decision by itself. If publication as `eris` proves impossible, release is blocked and M0 must
be explicitly reopened before any different distribution name can be considered.

### M0 gate

**Satisfied.** Branch policy, v0.1 scope, public naming, conceptual Artifact Schema v2, Python
policy, dependency boundary, configuration authority, target API/CLI, the M1–M9 tracker, and the
distribution name `eris` are frozen and versioned. External PyPI publication feasibility remains
a release condition, not an open M0 decision.

## M1 — Core correctness and release baseline

**Status:** `DONE`

**Objective:** Stabilize the existing explicit-representation core before Registry/config v2.

### Work units

- [x] **M1.1 — Declare core runtime dependencies**
  - **Scope:** Declare `gemmi` and `platformdirs`, audit mandatory runtime imports, and apply
    `>=3.10,<3.13`.
  - **Acceptance:** Core runtime dependencies are complete and consistent; this unit alone does
    not require Roxy removal.
  - **Depends on:** M0 internal contract.
- [x] **M1.2 — Establish a single package-version authority**
  - **Scope:** Define one source shared by distribution metadata and `eris.__version__`, ready
    for later provenance and CLI use.
  - **Acceptance:** Installed metadata and `eris.__version__` agree without a misleading
    fallback.
  - **Depends on:** M1.1.
- [x] **M1.3 — Internalize residue property constants and remove Roxy**
  - **Scope:** Preserve current values within Eris, document their provenance, and remove the
    runtime Roxy import.
  - **Acceptance:** Regression tests prove value equivalence and core imports without Roxy.
  - **Depends on:** M1.1.
- [x] **M1.4 — Fix B-factor propagation**
  - **Scope:** Correct residue/CA B-factor propagation and fallback in current feature building.
  - **Acceptance:** Present and missing B-factor cases produce the intended tested values.
  - **Depends on:** M1.3 where residue features use the internalized constants.
- [x] **M1.5 — Harden feature parsing and builder calls**
  - **Scope:** Correct unsafe parsing calls and incompatible builder arguments found in the
    baseline while preserving valid current feature modes.
  - **Acceptance:** Supported modes build and invalid inputs fail with actionable errors.
  - **Depends on:** M1.3.
- [x] **M1.6 — Preserve authoritative `feature_keys`**
  - **Scope:** Fix pre-v2 finalization so explicit feature metadata is preserved and inference is
    only a non-destructive fallback; do not implement Schema v2.
  - **Acceptance:** Encoder keys survive in exact order and align with feature dimensions.
  - **Depends on:** M1.5.
- [x] **M1.7 — Correct continuous contact-map metadata**
  - **Scope:** Describe continuous contact values correctly and keep them distinct from the
    binned contact-map representation.
  - **Acceptance:** Keys, dtype, shape, and semantics agree for continuous and binned variants.
  - **Depends on:** M1.6.
- [x] **M1.8 — Deduplicate symmetric kNN topology**
  - **Scope:** Enforce unique directed pairs, one reverse edge when symmetric, no self-loops when
    disabled, and deterministic ordering.
  - **Acceptance:** Focused tests prove these frozen invariants without a new scientific decision.
  - **Depends on:** M0 graph semantics.
- [x] **M1.9 — Keep edge features aligned after topology normalization**
  - **Scope:** Reindex edge index, distances, and features together through deduplication,
    reversal, and ordering.
  - **Acceptance:** Every feature row maps to its edge and all edge-array lengths agree.
  - **Depends on:** M1.8.
- [x] **M1.10 — Correct graph statistics and self-loop accounting**
  - **Scope:** Calculate counts, degree, density, and self-loops against the final topology
    without double-counting symmetric directed edges.
  - **Acceptance:** Directed, symmetric, self-loop, and empty examples match manual expectations.
  - **Depends on:** M1.8 and M1.9.
- [x] **M1.11 — Separate granularity from coordinate source in current core paths**
  - **Scope:** Correct current uses of `ca` as granularity; full consolidation remains in M2.
  - **Acceptance:** Current metadata can express `granularity = residue` and
    `coordinate_source = ca` independently.
  - **Depends on:** M1.5 and the M0 terminology contract.
- [x] **M1.12 — Propagate atom-selection filters**
  - **Scope:** Pass current water, heteroatom, hydrogen, ion, altloc, and amino-acid options into
    structural loading, provisionally preserving current defaults until M2.
  - **Acceptance:** Explicit filter values affect loaded atoms and have focused tests.
  - **Depends on:** M1.5.
  - **Decision note:** Conflicting defaults go to M2.1 rather than being silently resolved.
- [x] **M1.13 — Handle empty and undersized graph inputs safely**
  - **Scope:** Cover `N=0`, `N=1`, `N<k`, and cutoff graphs without neighbors.
  - **Acceptance:** Each case yields a strict-valid result or documented validation error, with no
    out-of-range indices.
  - **Depends on:** M1.8, M1.10, and M1.12.
- [x] **M1.14 — Correct voxel channel metadata**
  - **Scope:** Align voxel channel names/order with actual tensor axes and values.
  - **Acceptance:** Channel keys, shape, dtype, and order are deterministic and consistent.
  - **Depends on:** M1.6.

### M1 gate

- Clean installation succeeds in at least one supported Python environment.
- `python -c "import eris; print(eris.__version__)"` succeeds without Roxy, Sylphy, Torch, PyG,
  or TorchDrug.
- Core dependencies are correctly declared.
- Focused scientific/regression tests and graph invariants are green.
- Every current explicit representation produces a strict-valid artifact under its corresponding
  pre-v2 contract.

**Result:** `M1 GATE: PASS` — 2026-08-19.

Gate evidence:

- Python 3.12 completed the full gate: 172 tests passed, 8 optional Torch/PyG tests skipped,
  and the focused M1 regression suite passed all 167 tests.
- Python 3.10 and 3.11 passed clean install, import, version, and `pip check` smoke validation;
  the full Python 3.10–3.12 CI matrix remains assigned to M8.6.
- Core dependencies are declared, and base import loads none of Roxy, Sylphy, Torch, PyG, or
  TorchDrug.
- All ten current explicit representations executed and produced strict-valid pre-v2 artifacts.
- Tracked `out/` cleanup and the Ruff baseline remain assigned to M8.3 and M8.5 respectively;
  neither blocks the completed M1 gate.

## M2 — Registry/config consolidation

**Status:** `DONE`

**Objective:** Make typed canonical configuration the single source of truth for explicit
representations.

### Work units

- [x] **M2.1 — Freeze canonical explicit-representation defaults**
  - **Scope:** Resolve 32/36 bins; public `bool`/`uint8`; `k`; cutoffs;
    symmetrization/self-loops; voxel size, resolution, margin, channels, pooling and clamp;
    atom-selection/altloc defaults; and missing-residue policy where ambiguous.
  - **Acceptance:** A reviewed table records type, unit, semantics, and one default per public
    parameter, distinguishing scientific decisions from software/API choices.
  - **Depends on:** M1 gate.
  - **Decision note:** Structural/biological ambiguity goes to Nicole/Julián; software/API/release
    choices go to David/project owner.
- [x] **M2.2 — Define typed canonical configuration primitives**
  - **Scope:** Implement typed validation, deterministic serialization, and independent
    granularity/coordinate-source concepts.
  - **Acceptance:** Invalid combinations fail early and equivalent configs serialize identically.
  - **Depends on:** M2.1.
- [x] **M2.3 — Consolidate canonical explicit Representation Registry**
  - **Scope:** Register exactly `atom_point_cloud`, `residue_point_cloud`, `atom_knn_graph`,
    `residue_knn_graph`, `residue_contact_graph`, `residue_distogram`,
    `residue_distogram_bins`, `residue_contact_map`, `residue_contact_map_bins`, and
    `residue_voxel_grid`.
  - **Acceptance:** Registry key, encoder name, family, type, granularity, and coordinate source
    satisfy the frozen invariant; no extra public names appear.
  - **Depends on:** M2.2.
- [x] **M2.4 — Migrate point-cloud encoders**
  - **Scope:** Move atom/residue point clouds to canonical typed configuration.
  - **Acceptance:** Direct and Registry construction share defaults and validation.
  - **Depends on:** M2.3.
- [x] **M2.5 — Migrate graph encoders**
  - **Scope:** Move atom kNN, residue kNN, and residue contact graphs while preserving M1
    invariants.
  - **Acceptance:** Config, Registry metadata, encoder behavior, and defaults agree.
  - **Depends on:** M2.3 and the M1 graph units.
- [x] **M2.6 — Migrate matrix encoders**
  - **Scope:** Move continuous/binned residue distograms and contact maps.
  - **Acceptance:** Each variant preserves canonical identity, dtype, semantics, and defaults.
  - **Depends on:** M2.3 and M1.7.
- [x] **M2.7 — Migrate residue voxel grid**
  - **Scope:** Move voxel geometry/channels to canonical configuration.
  - **Acceptance:** Validation/defaults come from config and metadata preserves M1.14 alignment.
  - **Depends on:** M2.3 and M1.14.
  - **Decision note:** Voxel coordinates are centered exactly once. Scientific review confirmed
    historical double-centering was incorrect; the existing M2.7 single-centering correction is
    authoritative.
- [x] **M2.8 — Derive parameter schemas, help metadata and fingerprints from canonical config**
  - **Scope:** Generate Registry schemas, help, artifact parameters, and fingerprints from config.
  - **Acceptance:** Effective defaults do not drift and fingerprints change only for effective
    config changes.
  - **Depends on:** M2.4–M2.7.

### M2 gate

Canonical public names are identical everywhere; typed config is the single authority;
Registry/config/encoder defaults agree; granularity and coordinate source are correct; and
fingerprints are deterministic.

**Result:** `M2 GATE: PASS` — 2026-08-20.

Gate evidence:

- Registry exposes exactly the ten canonical explicit representations with aligned config and
  encoder identities, deterministic ordering, and no runtime aliases.
- Typed config drives defaults, runtime parameters, artifact metadata, fingerprints, and derived
  Registry parameter introspection without duplicating validation rules.
- All ten default representations produced strict-valid pre-v2 artifacts; the focused M2 gate
  passed 414 tests and the full Python 3.12 suite passed 546 tests with 8 optional tests skipped.
- Historical CLI parameter mutation remains deferred to M6.5; the clean Python 3.10–3.12 CI
  matrix remains assigned to M8.6.

## M3 — Artifact Schema v2

**Status:** `DONE`

**Objective:** Implement the first stable public artifact contract.

### Work units

- [x] **M3.1 — Implement Schema v2 models and validation**
  - **Scope:** Implement all conceptual fields, vocabularies, validation, and extensions frozen in
    `docs/architecture.md`.
  - **Acceptance:** Valid artifacts pass; missing/incompatible contract values fail clearly.
  - **Depends on:** M2 gate.
- [x] **M3.2 — Implement reproducible input and provenance records**
  - **Scope:** Record input identity/hash, selection, effective config, Eris version, producer,
    and relevant environment provenance.
  - **Acceptance:** Deterministic provenance is separated from volatile data and uses M1.2.
  - **Depends on:** M3.1 and M1.2.
- [x] **M3.3 — Implement outputs and authoritative feature semantics**
  - **Scope:** Separate output layout from ordered feature semantics and validate keys, shapes,
    dtypes, and counts against payloads.
  - **Acceptance:** Explicit metadata is never destructively re-inferred; mismatches fail early.
  - **Depends on:** M3.1 and M1.6.
- [x] **M3.4 — Map explicit representations to Schema v2**
  - **Scope:** Define v2 metadata/outputs for all ten explicit representations.
  - **Acceptance:** Every encoder emits strict-valid family/type/granularity/source metadata.
  - **Depends on:** M3.2, M3.3, and M2 gate.
- [x] **M3.5 — Implement deterministic artifact serialization**
  - **Scope:** Persist canonical `data.npz` and `meta.json` safely and deterministically.
  - **Acceptance:** Equivalent artifacts serialize reproducibly; incomplete writes are invalid.
  - **Depends on:** M3.3.
- [x] **M3.6 — Implement loading and integrity validation**
  - **Scope:** Validate version, files, outputs, shapes, dtypes, keys, and counts on load.
  - **Acceptance:** Valid artifacts round-trip; corruption/unknown versions fail clearly.
  - **Depends on:** M3.5.
- [x] **M3.7 — Define explicit Schema v1 pre-release behavior**
  - **Scope:** Detect v1 and implement reviewed rejection or limited internal handling without a
    public compatibility promise.
  - **Acceptance:** v1 behavior is explicit/tested and never silently treated as v2.
  - **Depends on:** M3.6.
  - **Decision note:** Additional schema policy belongs to David/project owner.
- [x] **M3.8 — Add Schema v2 conformance suite for all explicit representations**
  - **Scope:** Test valid/invalid cases, strict validation, round-trip, semantics, and extensions.
  - **Acceptance:** All ten representations pass without historical `out/` as a golden corpus.
  - **Depends on:** M3.4, M3.6, and M3.7.

### M3 gate

All ten explicit representations produce Schema v2, validate strictly, round-trip, preserve
exact semantic metadata, and do not depend on historical `out/`.

## M4 — Learned representation infrastructure

**Status:** `DONE`

**Objective:** Build framework-independent learned structural infrastructure before a real
backend.

### Work units

- [x] **M4.1 — Define generic structural embedder protocol/config**
  - **Scope:** Define availability, preprocessing, checkpoint loading, inference, local/global
    output, and validated config interfaces.
  - **Acceptance:** A backend implements the protocol without GearNet-specific generic concepts.
  - **Depends on:** M3 gate.
- [x] **M4.2 — Implement learned Registry and lazy availability detection**
  - **Scope:** Keep learned registration separate from explicit Registry and load frameworks lazily.
  - **Acceptance:** Missing optional deps do not break core import or explicit discovery.
  - **Depends on:** M4.1.
- [x] **M4.3 — Define checkpoint metadata and SHA-256 verification**
  - **Scope:** Represent model/backend ID, source, license/citation, checkpoint, and SHA-256.
  - **Acceptance:** Correct files validate before load; hash mismatches are rejected.
  - **Depends on:** M4.1 and M3.2.
- [x] **M4.4 — Implement backend-agnostic model weight cache**
  - **Scope:** Use `platformdirs` for local/cached resolution, safe insertion, verification, and
    partial/collision handling.
  - **Acceptance:** Fake checkpoints resolve reproducibly and corrupt files are not reused.
  - **Depends on:** M4.3 and M1.1.
- [x] **M4.5 — Define canonical local/global learned outputs and node mapping**
  - **Scope:** Implement `node_embedding`, `global_embedding`, `node_ids`, optional
    `valid_mask`, and unambiguous biological mapping.
  - **Acceptance:** Every local row maps to a node ID and outputs validate against v2.
  - **Depends on:** M4.1 and M3.3.
- [x] **M4.6 — Implement device and precision abstraction**
  - **Scope:** Expose `auto`, `cpu`, supported `cuda`, and `fp32` default with provenance.
  - **Acceptance:** Fake CPU works; unavailable devices follow documented behavior; core remains
    framework-independent.
  - **Depends on:** M4.1 and M3.2.
- [x] **M4.7 — Validate the complete learned path using a fake backend**
  - **Scope:** Exercise input → preprocessing → checkpoint/cache → fake model → local/global
    outputs → mapping → Artifact v2 → save/load.
  - **Acceptance:** The whole path passes without TorchDrug/GearNet.
  - **Depends on:** M4.2, M4.4, M4.5, M4.6, and M3.6.

### M4 gate

A fake backend completes the whole learned architecture without TorchDrug/GearNet becoming core
dependencies.

## M5 — GearNet / GearNet-Edge

**Status:** `DONE`

**Objective:** Integrate the frozen pretrained GearNet-Edge backend. A second model/variant is not
required for v0.1.

### Work units

- [x] **M5.1 — Select and freeze the official backend/checkpoint**
  - **Scope:** Record architecture, exact checkpoint, source, license, citation, SHA-256,
    preprocessing, output dimensions, and pooling semantics.
  - **Acceptance:** GearNet-Edge with `mc_gearnet_edge.pth` has approved, verifiable provenance
    and the full frozen contract in `docs/gearnet_edge_backend.md`.
  - **Depends on:** M4 gate and required stakeholder decisions.
- [x] **M5.2 — Establish optional backend dependency boundary**
  - **Scope:** Package backend deps optionally with lazy availability/version diagnostics.
  - **Acceptance:** Core-only works; missing backend guidance is actionable.
  - **Depends on:** M5.1 and M4.2.
- [x] **M5.3 — Implement model-specific structural preprocessing and residue mapping**
  - **Scope:** Reproduce checkpoint preprocessing/features/edges/selection/mapping; do not reuse
    generic Eris graphs without demonstrated equivalence.
  - **Acceptance:** Preprocessing is deterministic and every model node maps to input.
  - **Depends on:** M5.1 and M4.5.
  - **Completion:** The frozen canonical-20/CA biological policy, fail-fast missing-CA contract,
    explicit sparse residue mapping, sanitized single-chain transport, exact GearNet
    GraphConstruction, and policy/mapping/equivalence tests are complete.
- [x] **M5.4 — Implement checkpoint loading and deterministic inference**
  - **Scope:** Verify/load checkpoint; use eval, inference/no-grad, fp32 default, explicit device;
    emit local/global embeddings.
  - **Acceptance:** Results are deterministic within tolerance and match frozen output shapes.
  - **Depends on:** M5.2, M5.3, M4.4, and M4.6.
  - **Completion:** Exact verified-checkpoint loading and the exact upstream forward now run under
    `torch.inference_mode()` on the frozen CPU/fp32 stack. Strict pre-forward and output validation,
    direct node/global NumPy conversion, M4 `StructuralEmbeddingResult` modes, actionable errors,
    and bitwise repeatability with the official checkpoint are validated.
- [x] **M5.5 — Emit complete learned Artifact v2 provenance**
  - **Scope:** Persist model/checkpoint/hash, preprocessing, device/precision, IDs/masks, outputs.
  - **Acceptance:** Artifacts validate/round-trip without losing alignment or provenance.
  - **Depends on:** M5.4 and M3.6.
  - **Completion:** The canonical `gearnet_edge` adapter, official model/checkpoint descriptor,
    frozen preprocessing parameters, CPU/fp32 generation provenance, output metadata, and cache
    identity/source-refresh behavior are implemented and validated through the generic M4 runner
    and Artifact Schema v2 contracts without path leakage or a parallel schema.
- [x] **M5.6 — Add real-backend integration smoke test**
  - **Scope:** Cover real preprocessing, checkpoint hash/load, finite embeddings, mapping,
    repeatability/tolerance, and save/load.
  - **Acceptance:** The frozen backend passes reproducibly in its documented environment.
  - **Depends on:** M5.5.
  - **Completion:** The production Registry and generic runner complete the verified offline
    GearNet-Edge lifecycle from real structure input through strict Artifact v2 persistence and
    cache reuse. Cache-hit short-circuiting, source refresh, independent bitwise determinism,
    output modes, lazy core-only failure, and fail-fast missing-CA behavior are validated.

### M5 gate

**Satisfied:** One frozen pretrained structural backend generates reproducible valid residue/node
and global embeddings through the production Registry and runner. A second GearNet-family backend
is not required.

## M6 — Python API and CLI

**Status:** `DONE`

**Objective:** Expose a coherent public interface over stable explicit and learned contracts.

### Work units

- [x] **M6.1 — Implement stable representation/embedder discovery APIs**
  - **Scope:** Implement `eris.list_representations()`, `eris.get_representation_info()`,
    `eris.list_embedders()`, and `eris.get_embedder_info()`.
  - **Acceptance:** Discovery reports canonical metadata and lazy learned availability.
  - **Depends on:** M5 gate and M2.8.
- [x] **M6.2 — Implement `eris.encode()`**
  - **Scope:** Execute canonical explicit representations with typed config and v2 output.
  - **Acceptance:** All ten names produce strict-valid round-trippable artifacts.
  - **Depends on:** M3.8 and M2.8.
- [x] **M6.3 — Implement `eris.embed()`**
  - **Scope:** Execute learned backend/checkpoint/device/precision with v2 output.
  - **Acceptance:** Official-backend behavior/provenance matches M5; unavailable backends fail clearly.
  - **Depends on:** M5.5.
- [x] **M6.4 — Implement CLI version and discovery hierarchy**
  - **Scope:** Implement exactly `eris --version`, `eris representations list`,
    `eris representations info <name>`, `eris embedders list`, and
    `eris embedders info <name>`.
  - **Acceptance:** Version/discovery match public APIs; no generic `eris list` is introduced.
  - **Depends on:** M6.1 and M1.2.
- [x] **M6.5 — Implement CLI explicit encoding**
  - **Scope:** Implement `eris encode ...` from the Python API and canonical help/config.
  - **Acceptance:** CLI/API equivalent requests produce equivalent artifacts.
  - **Depends on:** M6.2 and M6.4.
- [x] **M6.6 — Implement CLI learned embedding**
  - **Scope:** Implement `eris embed ...` with availability/checkpoint/device/precision.
  - **Acceptance:** Behavior matches `eris.embed()` and provenance is complete.
  - **Depends on:** M6.3 and M6.4.
- [x] **M6.7 — Implement artifact info/validation and auxiliary commands**
  - **Scope:** Implement artifact inspection/validation plus `eris models ...`,
    `eris cache ...`, and `eris info`, exposing only implemented operations.
  - **Acceptance:** Commands consume public contracts and validate v2 artifacts.
  - **Depends on:** M6.4, M3.6, and M4.4.
- [x] **M6.8 — Add Python API/CLI contract tests**
  - **Scope:** Cover discovery, encode/embed, hierarchy, version, validation, and optional failures.
  - **Acceptance:** README-level operations execute and API/CLI outputs agree.
  - **Depends on:** M6.5, M6.6, and M6.7.

### M6 gate

**Satisfied:** README-level Python and CLI operations work over explicit and learned
representations without internal implementation knowledge and use exactly the frozen command
hierarchy.

## M7 — Batch, pack, datasets and export

**Status:** `DONE`

**Objective:** Make multi-sample production, indexing, dataset consumption, and exports reliable.

### Work units

- [x] **M7.1 — Standardize structure/batch manifests**
  - **Scope:** Structure fields: `id`, `path`, `chain`, `model`. Batch result fields:
    `id`, `representation_name`, `representation_family`, `artifact_dir`, `status`,
    `error_type`, `error_message`.
  - **Acceptance:** Every input has one auditable deterministic result with explicit status.
  - **Depends on:** M6 gate.
- [x] **M7.2 — Implement batch encode/embed and deterministic resume**
  - **Scope:** Run manifests, preserve partial failures, write safely, and resume only equivalent
    completed tasks.
  - **Acceptance:** Failures do not corrupt outputs and config changes prevent invalid reuse.
  - **Depends on:** M7.1, M6.2, and M6.3.
- [x] **M7.3 — Reconcile historical batch/index consumers with v2**
  - **Scope:** Move internal readers/writers to canonical manifest/v2 contracts; reject ambiguity.
  - **Acceptance:** Consumers share one contract without public pre-release compatibility promises.
  - **Depends on:** M7.1 and M3.7.
- [x] **M7.4 — Implement canonical `eris pack`**
  - **Scope:** Implement `eris pack <artifact-root> --out index.csv`: recursively discover
    artifact dirs, validate `meta.json`, and emit one row per artifact with `id`,
    `artifact_dir`, `meta_path`, `representation_name`, `representation_family`,
    `representation_type`, `granularity`, `n_nodes`, `n_edges`, `shape_summary`; learned
    rows may add `node_embedding_dim`, `global_embedding_dim`, and `model_name`.
  - **Acceptance:** Index is deterministic; `pack` never rewrites `data.npz`, stacks arrays,
    creates aggregate payloads, or pads/truncates data.
  - **Depends on:** M7.3 and M3.6.
  - **Completion:** `artifact_dir` and `meta_path` are serialized as portable POSIX paths
    relative to the `index.csv` parent so downstream consumers can resolve the index without
    hidden artifact-root state.
- [x] **M7.5 — Align `ErisFolderDataset` with the canonical packed index**
  - **Scope:** Load samples from M7.4 while preserving paths, IDs, metadata, and learned mappings.
  - **Acceptance:** Indexing is stable; duplicate/missing IDs fail; samples remain traceable.
  - **Depends on:** M7.4.
- [x] **M7.6 — Stabilize dataset/collate behavior**
  - **Scope:** Define explicit behavior for graph, matrix, cloud, grid, and learned outputs.
  - **Acceptance:** Compatible samples collate deterministically; incompatibilities fail clearly.
  - **Depends on:** M7.5.
- [x] **M7.7 — Complete Torch export**
  - **Scope:** Export graph, matrix, cloud, grid, and learned embeddings downstream to Torch while
    canonical storage remains NumPy `data.npz` plus JSON `meta.json`.
  - **Acceptance:** Values, dtypes, feature order, IDs, masks, and mappings are preserved.
  - **Depends on:** M7.5 and M7.6.
- [x] **M7.8 — Complete PyTorch Geometric graph export**
  - **Scope:** With optional PyG, map graph `pos`, `node_attr` or learned node representation,
    `edge_index`, `edge_attr`, applicable `graph_attr`, and needed node IDs.
  - **Acceptance:** Topology/attributes align; missing PyG is optional; matrix/grid are never
    converted artificially into graphs.
  - **Depends on:** M7.5 and M7.6.
- [x] **M7.9 — Complete batch/pack/export CLI workflow**
  - **Scope:** Connect batch, canonical `eris pack`, and `eris export ...` to public APIs/CLI.
  - **Acceptance:** Workflow needs no manual metadata edits and never hides failed tasks.
  - **Depends on:** M7.2, M7.4, M7.7, M7.8, and M6.8.
- [x] **M7.10 — Add end-to-end workflow tests**
  - **Scope:** Test manifest → encode/embed → artifacts → pack/index → dataset → Torch/PyG export.
  - **Acceptance:** IDs, metadata, alignment, failures/resume, and learned mappings are preserved
    without generated repository output as a golden corpus.
  - **Depends on:** M7.9.

### M7 gate

**Satisfied:** The full workflow operates without manual file editing or silently dropped
failures. `pack` is metadata-index generation only; Torch/PyG are downstream exports from
NumPy/JSON artifacts.

## M8 — Packaging, testing, CI, docs and hygiene

**Status:** `DONE`

**Objective:** Make the implemented project reproducibly installable, testable, documented, and
clean.

### Work units

- [x] **M8.1 — Audit final dependency boundaries**
  - **Scope:** Audit final real imports and core/optional installs; this is not the first
    declaration of `gemmi`/`platformdirs`.
  - **Acceptance:** Fresh core/learned installs have no missing/accidental mandatory deps and pass
    `pip check`.
  - **Depends on:** M7 gate, M1.1, and M5.2.
- [x] **M8.2 — Finalize packaging metadata and optional extras**
  - **Scope:** Correct metadata, classifiers, URLs, license/readme, Python range, content, extras;
    use the distribution name `eris` frozen by M0.X.
  - **Acceptance:** Wheel/sdist describes Eris accurately and uses distribution name `eris`.
  - **Depends on:** M8.1 and the completed M0.X naming contract.
- [x] **M8.3 — Remove tracked generated outputs and enforce repository hygiene**
  - **Scope:** Untrack generated `out/`, preserve fixtures, enforce ignores/clean worktree.
  - **Acceptance:** Tests/examples do not leave expected output tracked or dirty the repository.
  - **Depends on:** M7.10.
- [x] **M8.4 — Harden unit/integration/regression suite**
  - **Scope:** Close contract coverage gaps across M1–M7.
  - **Acceptance:** Public contracts/regressions have proportionate positive/negative coverage.
  - **Depends on:** M7 gate.
- [x] **M8.5 — Establish lint/static-check baseline**
  - **Scope:** Finalize reproducible Ruff, formatting, diff, and justified static checks.
  - **Acceptance:** Checks pass locally/CI with narrow justified exclusions.
  - **Depends on:** M8.4.
- [x] **M8.6 — Add core CI matrix for Python 3.10–3.12**
  - **Scope:** Clean install, import, `pip check`, core/API/CLI tests, and build smoke.
  - **Acceptance:** Python 3.10, 3.11, and 3.12 pass without learned deps.
  - **Depends on:** M8.1, M8.4, and M8.5.
- [x] **M8.7 — Add learned/backend and wheel/sdist CI lanes**
  - **Scope:** Validate learned optional install, fake backend, controlled real smoke, built wheels.
  - **Acceptance:** Optional failures are isolated and compatible learned environment documented.
  - **Depends on:** M8.2, M8.6, and M5.6.
- [x] **M8.8 — Complete README and reference/development documentation**
  - **Scope:** Document only implemented installation, names, API/CLI, v2, backend, workflows,
    architecture, and contribution.
  - **Acceptance:** Public examples/text match executable contracts; v1 remains pre-release.
  - **Depends on:** M6.8, M7.10, and M8.2.
- [x] **M8.9 — Audit/rebuild examples and notebooks against public API**
  - **Scope:** Align explicit, v2, batch, and optional backend examples with API/output policy.
  - **Acceptance:** Examples validate reproducibly, avoid internals/history, and create no tracked
    output.
  - **Depends on:** M8.3 and M8.8.

### M8 gate

CI is green; core passes on Python 3.10–3.12; learned passes in its compatible environment;
clean wheel install passes; docs/examples execute; no generated execution output is tracked.

## M9 — Release candidate and PyPI

**Status:** `NOT STARTED`

**Objective:** Produce, audit, and—only with explicit authorization—publish Eris v0.1.0.

### Work units

- [ ] **M9.1 — Verify/secure the PyPI `eris` namespace before release**
  - **Scope:** Verify that the already-frozen distribution name `eris` can be used for PyPI
    publication and secure the required ownership/access before release.
  - **Acceptance:** Publication as `eris` is demonstrably available and authorized. If it is not,
    release remains blocked and M0 is explicitly reopened before any different name is considered;
    M9.1 never selects a replacement name.
  - **Depends on:** Completed M0.X naming contract and M8 gate.
- [ ] **M9.2 — Freeze v0.1.0 version and CHANGELOG**
  - **Scope:** Set version through M1.2 and document delivered capabilities/limitations.
  - **Acceptance:** Version, changelog, and release contract agree without unimplemented promises.
  - **Depends on:** M9.1.
- [ ] **M9.3 — Build and inspect wheel/sdist release candidate**
  - **Scope:** Build cleanly, inspect content/metadata, record hashes.
  - **Acceptance:** Wheel/sdist share version and exclude generated/cached/unrelated content.
  - **Depends on:** M9.2.
- [ ] **M9.4 — Validate clean installations from built artifacts**
  - **Scope:** Test core on 3.10–3.12 and learned in its environment: import, version, CLI,
    `pip check`, explicit/learned smoke.
  - **Acceptance:** Tests run against built artifacts without relying on source tree.
  - **Depends on:** M9.3.
- [ ] **M9.5 — Perform final release-contract audit**
  - **Scope:** Audit M0 and all M1–M8 gates; functional failures reopen their owning milestone.
  - **Acceptance:** No release blockers remain.
  - **Depends on:** M9.4.
- [ ] **M9.6 — Publish the approved Eris v0.1.0 artifacts**
  - **Scope:** Only with explicit human authorization, publish the audited artifacts and matching
    tag/release record.
  - **Acceptance:** Public install works and distribution/import/CLI/version/tag/notes/hashes agree.
  - **Depends on:** M9.5 and explicit human authorization.

### M9 gate

A new user can install the authorized public distribution and reproduce documented quick-start
behavior from public instructions.

## Cross-milestone dependency order

```text
M1
→ M2
→ M3
→ M4
→ M5
→ M6
→ M7
→ M8
→ M9
```

M0.X is complete: the distribution name is frozen as `eris`. External ownership/availability of
that PyPI namespace is verified in M9.1 as a release condition. Failure blocks release and requires
explicitly reopening M0; it never authorizes a silent rename. No milestone may depend on a later
milestone.

## Stakeholder decisions

### Structural/biological decisions

Escalate to Nicole/Julián only when current implementation/contracts genuinely leave unresolved:

- atom/residue selection semantics;
- missing-coordinate/residue policy;
- scientific representation defaults;
- voxel semantics;
- GearNet structural preprocessing/mapping.

Do not reopen:

- the unique symmetric directed-edge invariant;
- Roxy core independence;
- granularity versus coordinate source.

### Architecture/release decisions

Escalate to David/project owner when required for:

- public defaults that are software/API decisions;
- schema policy;
- optional dependency packaging;
- final GearNet checkpoint/backend choice when licensing/release constraints matter;
- PyPI namespace access/ownership verification and release authorization.
