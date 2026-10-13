# Development Guide

## Prerequisites

- Python 3.11–3.14
- [uv](https://docs.astral.sh/uv/) (package manager)

## Dependencies

Core dependencies are managed via `uv`:
- **NumPy** holds every artifact array; **gemmi** reads structures, **Biotite** computes
  structural annotations and **aaindex** provides the AAIndex1 residue descriptors.
- **Polars** (`polars>=1.44,<2`) reads manifests and writes batch results and packed indexes.
- **Typer** and **Rich** power the CLI; **platformdirs** locates the per-user cache.
- Optional extras: `gearnet` (PyTorch, for GearNet-Edge embeddings and `to_torch`) and `pyg`
  (PyTorch and PyTorch Geometric, for `to_pyg`). `import eris` must work without them and must not
  import their runtimes. The `examples` group installs marimo/matplotlib for the notebooks.

## Setup

```bash
git clone https://github.com/kren-ai-lab/eris
cd eris
uv sync --all-extras
# optional: dependencies for the examples
uv sync --all-extras --group examples
```

## Common Tasks

```bash
uv run task format    # sort imports + ruff format
uv run task lint      # ruff check (no fixes)
uv run task lint-fix  # ruff check --fix
uv run task test      # pytest -q
uv run task test-v    # pytest -v
uv run task test-cov  # pytest + HTML coverage report
uv run task pyrefly   # pyrefly check
```

## Running the CLI

```bash
uv run eris --version
uv run eris --help
```

## Examples

The examples under `examples/` are marimo notebooks stored as plain Python files (requires the
`examples` group):

```bash
uv sync --all-extras --group examples
bash examples/run_ci_examples.sh                              # run them all, as CI does
uv run marimo edit examples/01_explicit_representations.py    # open one interactively
```

## Project Structure

The package uses a flat layout:

```text
eris/        # Flat package layout
tests/       # Flat test suite; structure fixtures and GearNet references in tests/data/
examples/    # marimo examples (plain .py)
docs/        # Technical documentation and permanent contracts
```

## Making changes

Read [AGENTS.md](AGENTS.md) for the scientific invariants and the contract that matches the
change under `docs/`. A change to numerical results needs a regression test that pins the new
behavior. The Python API and the CLI must agree for the same inputs and options.

## Choosing checks

Run `uv run task lint`, `uv run task pyrefly`, and `uv run task test` before submitting code
changes. Use the test file nearest the affected behavior while iterating:

| Location | What it checks |
| --- | --- |
| `tests/test_<area>.py` | Public behavior of the corresponding area |
| `tests/test_packaging_boundaries.py` | Dependencies and the lazy-import boundary of `import eris` |
| `tests/test_gearnet_native.py` | GearNet-Edge numerical parity; the parity test runs only when `ERIS_GEARNET_CHECKPOINT` points at the official checkpoint |

Tests that need an optional dependency or the checkpoint skip when it is unavailable; a skip does
not certify that integration.

CI runs the suite on Python 3.11–3.14 and once against the lowest declared direct dependencies,
runs the examples, and checks prek hooks. Pushing a `v*` tag publishes to PyPI.

When changing examples or the API they consume, run `bash examples/run_ci_examples.sh`.

For documentation-only changes, verify links, referenced public names, CLI options and runnable
snippets. The [documentation index](docs/README.md) is the user entry point; examples are
catalogued only in [examples/README.md](examples/README.md).
