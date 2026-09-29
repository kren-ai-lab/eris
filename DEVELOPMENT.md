# Eris development

This guide covers local contributor setup and validation. Scientific and persistence contracts
are documented under `docs/`.

## Environment

`pyproject.toml` is the authority for the current development Python range and dependencies. It
currently declares Python `>=3.11,<3.13`.

Create and activate an isolated Python 3.11 or 3.12 environment, then install the project with
development tools:

```console
python -m pip install -e ".[dev]"
python -m pip check
```

The import package and CLI command are both `eris`. The distribution name currently present in
package metadata is `eris-lib`; final public naming is release-policy work, not a local development
assumption.

## Optional dependency groups

Install only the optional integration being developed or tested:

```console
python -m pip install -e ".[gearnet]"
python -m pip install -e ".[pyg]"
```

- `gearnet` provides the frozen native GearNet-Edge PyTorch runtime. The verified checkpoint must
  be supplied separately and is never downloaded automatically.
- `pyg` provides PyTorch and PyTorch Geometric for graph export.
- A core import must remain usable without either optional group.

See `docs/gearnet_edge_backend.md` for the exact learned-runtime contract.

## Tests

Run the smallest relevant scope first:

```console
python -m pytest -q tests/test_residue_features.py
```

Then run related regression modules. For cross-cutting changes or a handoff gate, run the full
suite:

```console
python -m pytest -q
```

Optional dependency or checkpoint tests may skip when their documented runtime is unavailable.
Do not treat a skip as certification of that integration.

## Static checks

```console
python -m ruff check .
python -m ruff format --check .
python -m pip check
git diff --check
```

Keep `import eris` free of eager Torch, TorchDrug, PyG, RDKit, or GearNet numerical-runtime
imports. Packaging-boundary tests enforce this contract.

## Normal workflow

1. Verify the active branch, HEAD, and expected worktree.
2. Read the permanent contract relevant to the change.
3. Implement one focused change without altering unrelated behavior.
4. Run focused tests, then proportionate regression/static checks.
5. Review `git diff`, `git diff --check`, and `git status --short`.
6. Commit or publish only when explicitly authorized.

The repository does not claim release certification merely because local checks pass. CI matrices,
built-distribution installation, publishing, and supported-environment certification remain
separate release-hardening gates.
