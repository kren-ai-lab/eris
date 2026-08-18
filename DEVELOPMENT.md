# Eris Development Workflow

This document defines the development process for Eris. Scientific and structural contracts
are documented separately in `docs/architecture.md`.

## Development philosophy

The intended workflow is:

> ChatGPT/specification → Codex inspection/implementation → tests → diff review → manual commit

The repository, its tests, and its versioned contracts are the source of truth. Conversational
context can explain intent, but it does not replace repository evidence or an approved
versioned contract.

Development should begin with inspection, proceed through a narrowly scoped implementation,
and end with evidence that the requested behavior and surrounding contracts remain coherent.

## Branches

- `jr_revision` is the frozen historical baseline for Eris v0.1.
- `dev` is the only active development branch for Eris v0.1.
- `jr_revision` must not be modified.
- Feature parity must not be maintained between `jr_revision` and `dev`.
- Agents must not create or switch branches unless explicitly requested.

This is a deliberate adaptation of the original baseline directive. `dev` was created exactly
from `jr_revision` and is now the sole active v0.1 development line.

## Environment target

The official Eris core target is:

- Python 3.10
- Python 3.11
- Python 3.12

The conceptual packaging constraint is:

```text
>=3.10,<3.13
```

The current package metadata does not yet express the upper bound; aligning it is future work.
Learned backends may support a narrower Python range when their frameworks require it, but
that must not narrow the supported Python range of Eris core.

## Expected core dependency boundary

Eris core is intended to remain independent from:

- Roxy;
- Sylphy;
- Torch;
- PyTorch Geometric;
- TorchDrug and GearNet dependencies.

Known intended core runtime dependencies include:

- `numpy`;
- `scipy`;
- `pandas`;
- `tqdm`;
- `typer`;
- `rich`;
- `gemmi`;
- `platformdirs`.

This is the target dependency contract. The current package metadata has not yet been aligned
with it, and dependency changes must be handled in a separate authorized task.

## Validation workflow

Use the validation ladder in increasing scope:

1. **Focused:** reproduce the behavior and run the directly relevant test module, for example
   `python -m pytest -q tests/test_registry.py`.
2. **Subsystem:** run the tests covering the affected package or contract.
3. **Integration:** exercise the relevant cross-package flow, such as encoding through artifact
   persistence or artifact loading through export.
4. **Full suite:** run `python -m pytest -q` after the focused and integration checks pass.
5. **Distribution/release:** when applicable, validate build, installation, dependency
   consistency, package import, and CLI entry points.

The repository currently declares `python -m pytest -q`, `python -m build`, the `eris` entry
point, and pre-commit Ruff/mypy hooks. A complete clean environment and release validation
procedure is still pending; do not report those checks as working until the repository provides
and passes them.

## Git workflow

- Start from a known branch and a clean worktree.
- Inspect the relevant baseline before editing.
- Inspect `git diff` and `git diff --check` before handoff.
- Keep each change and eventual commit functionally coherent.
- Treat commits as manual review boundaries.
- Agents do not commit, push, tag, merge, or release unless explicitly instructed.
