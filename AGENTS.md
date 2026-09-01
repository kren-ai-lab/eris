# AGENTS.md — Eris development rules

## Authority and scope

- Work on branch `dev`.
- `dev` is the only active development line for Eris v0.1.
- `jr_revision` is the frozen historical/reference baseline for Eris v0.1.
- Never modify `jr_revision`.
- Never create or switch branches unless explicitly requested.
- Never maintain parity or synchronization between `dev` and `jr_revision`, and never copy changes into `jr_revision`.
- `docs/milestones.md` is the authoritative implementation tracker and wins over historical code or stale documentation.
- Implement only the requested milestone/work unit. Do not opportunistically start later milestones.
- If a requested change exposes a genuine contract/design blocker, STOP before changing the frozen contract and report the blocker concisely.
- Prefer focused changes over broad refactors.

## Git safety

- Never commit, push, tag, publish, release, merge, or rewrite history unless explicitly requested.
- Do not stage files unless explicitly asked.
- At the start of implementation tasks, verify current branch, HEAD subject, and clean/expected worktree.
- At the end, report `git diff --stat` and `git status --short`.

## Frozen project contracts

- Package/import/CLI name: `eris`.
- Core Python: `>=3.10,<3.13`.
- Preserve optional-dependency isolation.
- `import eris` must not eagerly import Torch, TorchDrug, PyG, RDKit, or GearNet runtime.
- Artifact persistence is `StructureArtifact` Schema v2 with `meta.json` + `data.npz`.
- Schema v1 is not a supported public artifact format.
- Do not change Artifact Schema v2, scientific cache identity, or GearNet scientific behavior unless the requested milestone explicitly requires it.
- CLI should delegate to the public Python API when an equivalent public API exists.
- Do not expose internal registries, runners, backends, descriptors, or cache implementation details as public API unless a milestone requires it.

## Scientific safety

- Do not silently pad, truncate, resize, normalize, infer topology, invent IDs, drop failed samples, or coerce invalid scientific values.
- Preserve array order and biological/sample identity.
- Incompatibilities should fail clearly rather than be repaired implicitly.
- PyG export may only use public graph topology already present in the Artifact.
- Learned embeddings must never leak or reconstruct GearNet internal topology.

## Scientific decision escalation

- Do not silently invent unresolved biological or structural semantics; escalate them to Nicole and/or Julián before implementation.
- Escalation is required for residue or atom selection, heteroatom inclusion, water handling, chain semantics, alternate-location handling, biologically meaningful graph construction, and structural feature interpretation.

## Implementation style

- Reuse canonical contracts/helpers instead of creating parallel parsers, serializers, hashes, or validators.
- Avoid large framework abstractions for one milestone.
- Preserve deterministic ordering.
- Keep optional dependencies lazy and feature-local.
- Do not add network access unless explicitly required.
- Do not download model checkpoints automatically.
- Do not use broad `except BaseException`.
- Avoid broad `except Exception` when stable project exception types can be used.

## Testing strategy

For a normal work unit:

1. Run focused tests for the subsystem changed.
2. Run relevant regression tests.
3. Run `ruff check` on changed Python files.
4. Run `ruff format --check` on changed Python files.
5. Run `git diff --check`.

Run the full Python 3.10 suite when:
- the task explicitly asks for it;
- the change crosses subsystem boundaries;
- a milestone gate is being closed;
- or before a commit when the diff is medium/high risk.

Run Python 3.12 focused/core checks when the change should work across the full core Python range.

Use `pip check` at milestone/commit gates or when dependencies/package metadata changed.

## Output/reporting

Keep reports short. Return only:

1. Files changed.
2. Important implementation decisions.
3. Deviations or blockers.
4. Focused/regression test results.
5. Full-suite result only if it was run.
6. `git diff --stat`.
7. `git status --short`.
8. Ready to commit: yes/no.
9. Recommended commit message.

Do not restate the whole prompt or milestone.
Do not include long inventories unless explicitly requested.
