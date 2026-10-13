# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

**Eris** is a Python library that turns **protein structures** (PDB/mmCIF) into explicit
numerical representations (point clouds, graphs, contact maps, distograms, voxel grids) and
learned GearNet-Edge embeddings. It sits alongside sibling libraries from the same lab, each with a
distinct scope:

- **Roxy** — classical protein sequence descriptors
- **Sylphy** — sequence encoders, embeddings and dimensionality reductions
- **Ruddy** — statistical exploratory data analysis (EDA)
- **Saber** — classical supervised machine learning on prepared numerical features

Eris does **not** train models or compute sequence features. Its output is a validated
`StructureArtifact` (`meta.json` plus `data.npz`) that downstream tools consume, optionally through
`to_torch`, `to_pyg` or `ErisFolderDataset`.

This project uses `uv` and `taskipy` — see `DEVELOPMENT.md` for setup and task commands.

## Package Layout

The package is a **flat layout** at `eris/`. `tests/` is flat (`tests/test_*.py`), with structure
fixtures and GearNet reference data under `tests/data/`. Permanent contracts live in `docs/`:
`architecture.md`, `structural_features.md`, `artifact_schema.md`, `representation_defaults.md`,
`external_residue_features.md` and `gearnet_edge_backend.md`. Read the relevant one before
changing the behavior it describes.

## Development Workflow

Package management is `uv`; task running is `taskipy`. See `DEVELOPMENT.md` for the full command
list (`uv sync --all-extras`, `uv run task format|lint|lint-fix|test|test-v|test-cov|pyrefly`).

## Scientific Invariants (read before touching any scientific module)

- **No silent changes to data.** Never pad, truncate, resize, normalize, clip, infer topology,
  invent IDs, drop failed samples or coerce invalid scientific values. Incompatible inputs fail
  with a clear error.
- **Deterministic identity.** Preserve sample identity, array order, feature order, exact keys and
  deterministic behavior.
- **One artifact schema.** Persistence is the `StructureArtifact` schema. Do not change it, the
  cache identity or GearNet's scientific behavior unless the task explicitly requires it.
- **Public topology only.** PyG export uses only topology already public in an artifact; learned
  embeddings never expose or reconstruct GearNet-internal topology.
- **Escalate ambiguous semantics.** Residue/atom selection, chains, heteroatoms, water, alternate
  locations, topology and structural-feature semantics need independent scientific review before
  implementation.

A change that alters numerical results needs a test that pins the new behavior.

## Contracts

- `import eris` must not import Torch, TorchDrug, PyG, RDKit or GearNet numerical code. Optional
  dependencies stay lazy and feature-local (`tests/test_packaging_boundaries.py` checks this).
- CLI commands delegate to the equivalent public Python API, and map failures to the exit codes
  documented in the README.
- Registries, runners, backends, descriptors and cache internals are not public API.
- No runtime network access, except downloading official model checkpoints whose HTTPS URL and
  SHA-256 are registered in Eris. Arbitrary user-supplied download URLs are not allowed.
- Reuse the canonical configs, validators, serializers, hashes and registries, and raise the
  exception types in `eris.exceptions` (all derive from `ErisError`).
