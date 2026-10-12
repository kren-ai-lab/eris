# Eris development

## Environment

`pyproject.toml` is the authority for package metadata, Python compatibility, and dependencies.
The supported range is Python `>=3.11,<3.15`. Development uses [uv](https://docs.astral.sh/uv/)
and the committed `uv.lock`:

```console
uv sync --all-extras
```

The distribution name is `erislib`; the import package and CLI command are `eris`.

## Optional environments

- `gearnet` installs PyTorch. On a checkpoint-cache miss, Eris can acquire the registered official
  checkpoint over HTTPS and verifies its registered SHA-256 before use. Pass
  `checkpoint_path` / `--checkpoint` to seed the same cache from a local file or to work offline.
- `pyg` installs PyTorch and PyTorch Geometric for graph export.
- `import eris` must work without either optional group and must not import their runtimes.

`uv sync` without `--all-extras` (or with `--extra gearnet` / `--extra pyg`) gives a narrower
environment. The GearNet CPU installation and checkpoint contract are in
[`docs/gearnet_edge_backend.md`](docs/gearnet_edge_backend.md).

## Tests and checks

Common tasks run through taskipy:

```console
uv run task test        # pytest -q
uv run task format      # sort imports + ruff format
uv run task lint        # ruff check
uv run prek run --all-files
git diff --check
```

Run focused tests while developing:

```console
uv run pytest -q tests/test_residue_features.py
```

The GearNet numerical-parity test runs only when `ERIS_GEARNET_CHECKPOINT` points at the official
checkpoint; otherwise it skips. Optional dependency or checkpoint tests may skip when their
runtime is unavailable. A skip is not certification of that integration.
`tests/test_packaging_boundaries.py` verifies the core lazy-import boundary.

CI (`.github/workflows/tests.yml`) runs the suite on Python 3.11–3.14, once against the lowest
declared direct dependencies. Pushing a `v*` tag publishes to
PyPI through `publish-pypi.yml`.

## Workflow

Development happens on `dev`; `jr_revision` is a frozen historical branch. Do not switch branches
or modify history as part of ordinary changes. Before editing, inspect the branch, HEAD, and
worktree. Preserve unrelated tracked and untracked work.

Read the relevant contract under `docs/`, make a focused change, run proportionate tests and
checks, then review `git diff`, `git diff --check`, and `git status --short`.
