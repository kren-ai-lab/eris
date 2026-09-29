# AGENTS.md — Eris development rules

## Repository authority

- Work on `dev`, the active development branch.
- `jr_revision` is a frozen historical baseline. Never modify it or maintain parity with `dev`.
- Do not create or switch branches unless explicitly requested.
- `pyproject.toml` is the development authority for package metadata, Python compatibility, and
  dependency groups.
- Permanent technical contracts live in `docs/architecture.md`, `docs/artifact_schema.md`,
  `docs/representation_defaults.md`, `docs/external_residue_features.md`, and
  `docs/gearnet_edge_backend.md`.
- Prefer focused changes; do not expand a requested task into unrelated cleanup or features.

## Git safety

- Never stage, commit, push, tag, publish, release, merge, or rewrite history unless explicitly
  requested.
- At the start of a change, verify branch, HEAD, and the expected worktree state.
- Preserve unrelated tracked and untracked user work.
- Never use destructive history or broad filesystem commands to discard local changes.
- At handoff, report `git diff --stat` and `git status --short`.

## Contract guardrails

- Preserve optional-dependency isolation. `import eris` must not eagerly import Torch, TorchDrug,
  PyG, RDKit, or the GearNet numerical runtime.
- Artifact persistence is `StructureArtifact` Schema v2 with `meta.json` and `data.npz`.
- Do not change Artifact Schema v2, scientific cache identity, or GearNet scientific behavior
  unless the task explicitly requires it.
- CLI operations should delegate to the equivalent public Python API.
- Do not expose internal registries, runners, backends, descriptors, or cache internals as public
  API without an explicit contract change.
- Keep optional dependencies lazy and feature-local. Do not add runtime network access or
  automatic checkpoint downloads.

## Scientific safety

- Preserve biological/sample identity, array order, deterministic feature order, and exact keys.
- Do not silently pad, truncate, resize, normalize, clip, infer topology, invent IDs, drop failed
  samples, or coerce invalid scientific values.
- Incompatible scientific inputs must fail clearly rather than be repaired implicitly.
- PyG export may use only public topology already present in an artifact.
- Learned embeddings must not expose or reconstruct GearNet-internal topology.
- Escalate unresolved residue/atom selection, chain, heteroatom, water, alternate-location,
  topology, or structural-feature semantics to Nicole and/or Julián before implementation.

## Implementation and validation

- Reuse canonical configs, validators, serializers, hashes, and registries rather than creating
  parallel mechanisms.
- Preserve deterministic behavior and use stable project exception types where available.
- Run focused tests for the changed subsystem and relevant regression tests.
- For cross-cutting, gate-level, or pre-commit work, run the full suite in the available supported
  environment.
- Run Ruff check and format checks for affected files, plus `git diff --check`.
- Run `pip check` when dependencies, packaging, or a handoff/release gate is involved.
- Keep reports concise: changed files, important decisions, blockers, validation results, diff,
  worktree status, readiness, and a recommended commit message.
