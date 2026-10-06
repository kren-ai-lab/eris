# AGENTS.md — Eris development rules

## Repository authority

- Work on `dev`; `jr_revision` is a frozen historical baseline.
- Do not create or switch branches unless explicitly requested.
- `pyproject.toml` controls package metadata, Python compatibility, and dependency groups.
- Permanent contracts live in `docs/architecture.md`, `docs/structural_features.md`,
  `docs/artifact_schema.md`, `docs/representation_defaults.md`,
  `docs/external_residue_features.md`, and `docs/gearnet_edge_backend.md`.
- Keep changes focused; do not add adjacent features or cleanup.

## Git safety

- Never stage, commit, push, tag, publish, release, merge, or rewrite history unless requested.
- Before editing, verify the branch, HEAD, and worktree.
- Preserve unrelated tracked and untracked work; never discard it with destructive commands.
- At handoff, report `git diff --stat` and `git status --short`.

## Contract guardrails

- `import eris` must not eagerly import Torch, TorchDrug, PyG, RDKit, or GearNet numerical code.
- Persistence is `StructureArtifact` Schema v2: `meta.json` plus `data.npz`.
- Do not change Schema v2, scientific cache identity, or GearNet scientific behavior unless the
  task explicitly requires it.
- CLI operations must delegate to the equivalent public Python API.
- Do not expose registries, runners, backends, descriptors, or cache internals as public API
  without an explicit contract change.
- Keep optional dependencies lazy and feature-local. Do not add runtime network access or
  automatic checkpoint downloads.

## Scientific safety

- Preserve sample identity, array order, feature order, exact keys, and deterministic behavior.
- Never silently pad, truncate, resize, normalize, clip, infer topology, invent IDs, drop failed
  samples, or coerce invalid scientific values.
- Fail clearly on incompatible scientific inputs.
- PyG export may use only topology already public in an artifact.
- Learned embeddings must not expose or reconstruct GearNet-internal topology.
- Escalate unresolved residue/atom selection, chain, heteroatom, water, alternate-location,
  topology, or structural-feature semantics for independent scientific review before
  implementation.

## Implementation and validation

- Reuse canonical configs, validators, serializers, hashes, and registries.
- Use stable project exception types where available.
- Run focused tests and relevant regressions; run the full suite for cross-cutting or gate work.
- Run Ruff check and format checks for affected files plus `git diff --check`.
- Run `pip check` for dependency, packaging, handoff, or release work.
- Report changed files, decisions, blockers, validation, diff, worktree status, readiness, and a
  recommended commit message.
