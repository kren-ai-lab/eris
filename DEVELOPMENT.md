# Eris development

## Environment

`pyproject.toml` is the authority for package metadata, Python compatibility, and dependencies.
The supported development range is Python `>=3.11,<3.13`.

Create and activate an isolated Python 3.11 or 3.12 environment, then install Eris and its
development tools:

```console
python -m pip install -e ".[dev]"
python -m pip check
```

The distribution name is `eris-lib`; the import package and CLI command are `eris`. Public naming
is still subject to release hardening.

## Optional environments

Install only the integration being developed or tested:

```console
python -m pip install -e ".[gearnet]"
python -m pip install -e ".[pyg]"
```

- `gearnet` installs the pinned PyTorch runtime. Supply the verified checkpoint separately; Eris
  never downloads it.
- `pyg` installs PyTorch and PyTorch Geometric for graph export.
- `import eris` must work without either optional group and must not import their runtimes.

The GearNet CPU installation and checkpoint contract are in
[`docs/gearnet_edge_backend.md`](docs/gearnet_edge_backend.md).

## Tests and checks

Run focused tests while developing:

```console
python -m pytest -q tests/test_residue_features.py
```

For cross-cutting changes or a handoff gate, run the tracked suite with a writable pytest base:

```bash
pt="${HOME}/eris-pytest-temp"
rm -rf "$pt"
mkdir -p "$pt"

python -m pytest -q \
  --basetemp "$pt/base" \
  -p no:cacheprovider
```

Run the static and environment checks:

```console
python -m ruff check .
python -m ruff format --check .
python -m pip check
git diff --check
```

Optional dependency or checkpoint tests may skip when their documented runtime is unavailable. A
skip is not certification of that integration. `tests/test_packaging_boundaries.py` verifies the
core lazy-import boundary.

## Workflow

Development happens on `dev`; `jr_revision` is a frozen historical branch. Do not switch branches
or modify history as part of ordinary changes. Before editing, inspect the branch, HEAD, and
worktree. Preserve unrelated tracked and untracked work.

Read the relevant contract under `docs/`, make a focused change, run proportionate tests and
checks, then review `git diff`, `git diff --check`, and `git status --short`. Local validation does
not replace clean-install, built-distribution, CI-matrix, or release-environment certification.
