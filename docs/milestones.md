# Eris v0.1 Milestones

This document tracks the approved implementation and release sequence. Historical partial code
does not make a milestone complete; each milestone must satisfy its stated gate.

## M0 — Project freeze and release contract

**Objective:** Freeze the architectural, naming, operating, and release direction for v0.1.

**Main required outcomes:** Document branch policy, framework boundaries, canonical
representation identifiers and terminology, core Python/dependency targets, the conceptual
Artifact Schema v2 contract, output policy, version authority, and the v0.1 milestone sequence.

**Gate:** The approved decisions are versioned in the repository, contradictions are surfaced,
and remaining external decisions are explicitly recorded.

**Status:** In progress. Decisions are frozen, but repository documentation is only now being
materialized and the desired PyPI distribution name remains externally unresolved.

## M1 — Core correctness

**Objective:** Establish a correct, importable, deterministic explicit-representation core.

**Main required outcomes:** Resolve verified core correctness, dependency-boundary, IO,
validation, and deterministic feature issues without redesigning the representation system.

**Gate:** Core behavior is covered by focused and subsystem tests across the supported Python
range.

**Status:** Not started.

## M2 — Registry/config consolidation

**Objective:** Make typed representation configuration the single source of truth.

**Main required outcomes:** Align canonical identifiers, runtime defaults, Registry metadata,
CLI descriptions, fingerprints, artifact parameters, and documentation from one authoritative
definition.

**Gate:** Every explicit representation satisfies the canonical naming invariant and has one
authoritative configuration contract.

**Status:** Not started.

## M3 — Artifact Schema v2

**Objective:** Implement the first stable public artifact contract.

**Main required outcomes:** Implement `data.npz`/`meta.json` Schema v2 persistence, authoritative
feature semantics, outputs, provenance, validation, counts, shapes, dtypes, and explicit-versus-
learned metadata rules.

**Gate:** Schema v2 artifacts validate and round-trip without semantic re-inference; Schema v1 is
treated as pre-release rather than a required public compatibility contract.

**Status:** Not started.

## M4 — Learned representation infrastructure

**Objective:** Add framework-independent infrastructure for learned structural representations.

**Main required outcomes:** Define embedder/model registration, local-to-biological-entity
mapping, canonical learned arrays, backend boundaries, and learned provenance without narrowing
the Eris core environment.

**Gate:** A learned backend can integrate through stable configuration and Artifact Schema v2
contracts without becoming a core dependency.

**Status:** Not started.

## M5 — GearNet/GearNet-Edge

**Objective:** Deliver the mandatory learned structural backend for v0.1.

**Main required outcomes:** Integrate GearNet/GearNet-Edge behind the learned-backend boundary,
including model and checkpoint provenance and unambiguous node mapping.

**Gate:** GearNet/GearNet-Edge produces validated learned artifacts through the M4 and M3
contracts with appropriate tests.

**Status:** Not started.

## M6 — Python API and CLI

**Objective:** Expose a coherent public interface over explicit and learned representations.

**Main required outcomes:** Implement the approved high-level Python API and conceptual CLI,
including representation/embedder discovery, encode/embed, information, cache, model, export,
and pack entry points as applicable.

**Gate:** Public Python and CLI identifiers agree with the canonical contracts and have
executable interface tests.

**Status:** Not started.

## M7 — Batch, pack, datasets and export

**Objective:** Make artifact production and consumption reliable for real multi-sample use.

**Main required outcomes:** Implement the frozen manifest, task-result, packed-index, dataset,
collation, and export contracts for explicit and learned artifacts.

**Gate:** An end-to-end manifest-to-artifact-to-pack-to-dataset/export flow passes integration
tests without depending on generated repository output as a golden corpus.

**Status:** Not started.

## M8 — Packaging, testing, CI and documentation

**Objective:** Make the project reproducibly installable, testable, and understandable.

**Main required outcomes:** Align package metadata and dependency boundaries, validate Python
3.10–3.12, establish CI, complete release-relevant tests, and publish accurate user and developer
documentation.

**Gate:** Supported environments pass the full suite, build/install checks, CLI checks, and
documentation verification.

**Status:** Not started.

## M9 — Release candidate and PyPI

**Objective:** Produce and approve the Eris v0.1 release candidate and publish the distribution.

**Main required outcomes:** Resolve PyPI namespace ownership, complete release metadata and
changelog, build final artifacts, perform clean-install validation, and complete manual release
review.

**Gate:** The approved release candidate passes all release checks and the intended PyPI
distribution can be published under an authorized name.

**Status:** Not started.

## Current external/open M0 item

- The desired PyPI distribution name is `eris`; its external namespace availability and
  ownership remain unresolved.
