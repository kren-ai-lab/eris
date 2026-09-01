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
- Python 3.13

The conceptual packaging constraint is:

```text
>=3.10,<3.14
```

The package metadata enforces this range. Learned backends may support a narrower Python range
when their frameworks require it, but that does not narrow the supported Python range of Eris core.

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

These dependencies form the declared core runtime contract. GearNet/TorchDrug/PyG remain optional
and are isolated from core imports.

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

The repository declares `python -m pytest -q`, `python -m build`, the `eris` entry point, and
Ruff pre-commit hooks. The core CI matrix covers Python 3.10–3.13; a separate Python 3.10 lane
validates the optional GearNet environment and wheel/sdist installation. There is no project-wide
mypy gate.

The canonical local static checks are:

```console
python -m ruff check .
python -m ruff format --check .
```

Use focused tests during ordinary development. Run the full Python 3.10 suite for broad or
gate-level changes; Python 3.12 and 3.13 core checks and `pip check` are expected at
milestone/packaging boundaries. The learned GearNet environment is intentionally Python 3.10-only.

## Git workflow

- Start from a known branch and a clean worktree.
- Inspect the relevant baseline before editing.
- Inspect `git diff` and `git diff --check` before handoff.
- Keep each change and eventual commit functionally coherent.
- Treat commits as manual review boundaries.
- Agents do not commit, push, tag, merge, or release unless explicitly instructed.
