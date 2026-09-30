# Eris v0.1 developer handoff

## Start here

Read these files in order:

1. [`README.md`](../README.md) for installation and public usage.
2. [`architecture.md`](architecture.md) for package layout, data flow, and extension points.
3. [`representation_defaults.md`](representation_defaults.md) for explicit scientific defaults.
4. [`artifact_schema.md`](artifact_schema.md) for persistence and identity contracts.
5. [`external_residue_features.md`](external_residue_features.md) for the generic residue-feature
   producer contract.
6. [`gearnet_edge_backend.md`](gearnet_edge_backend.md) for frozen learned behavior and
   release-environment certification.
7. [`DEVELOPMENT.md`](../DEVELOPMENT.md) for setup and validation commands.

The main implementation paths are:

- `src/eris/api.py` and `src/eris/discovery.py`: public Python API;
- `src/eris/core/config.py` and `src/eris/core/registry.py`: explicit configs and registration;
- `src/eris/core/io.py`: structure selection and extraction;
- `src/eris/features/` and `src/eris/encoders/`: explicit features and representations;
- `src/eris/core/artifacts.py`, `artifact_schema_v2.py`, and `cache.py`: persistence and identity;
- `src/eris/core/learned/`: learned runner, checkpoints, artifacts, and GearNet backend;
- `src/eris/pipelines/`, `datasets/`, and `export/`: batch, pack, dataset, Torch, and PyG workflows;
- `src/eris/cli/`: CLI adapters.

## State received

- Branch: `dev`.
- Version: `0.1.0`.
- Distribution metadata name: `eris-lib`.
- Import package and CLI command: `eris`.
- Python range in `pyproject.toml`: `>=3.11,<3.13`.
- Artifact persistence: `StructureArtifact` Schema v2.
- Explicit representations: ten registered encoders.
- Learned representations: one registered `gearnet_edge` backend.
- Checkpoints and external feature files are caller-supplied and never downloaded.

The scientific implementation is feature-complete for the v0.1 candidate. Publication and
supported-environment certification are not complete.

## Implemented surface

### Explicit representations

| Name | Type | Granularity |
| --- | --- | --- |
| `atom_point_cloud` | cloud | atom |
| `residue_point_cloud` | cloud | residue |
| `atom_knn_graph` | graph | atom |
| `residue_knn_graph` | graph | residue |
| `residue_contact_graph` | graph | residue |
| `residue_distogram` | matrix | residue |
| `residue_distogram_bins` | matrix | residue |
| `residue_contact_map` | matrix | residue |
| `residue_contact_map_bins` | matrix | residue |
| `residue_voxel_grid` | grid | residue |

Residue coordinates support CA, residue-atom mean, and center of mass with explicit missing-center
handling. The feature registry includes amino-acid identity, fixed physicochemical values,
AAIndex1, raw B-factor, explicitly sourced pLDDT, flags, and generic external matrices. Exact
defaults and dtypes are in [`representation_defaults.md`](representation_defaults.md).

### Persistence and workflows

- Every encoder returns a strict-valid Schema v2 artifact.
- Cache identity separates scientific content from invocation locators.
- External residue NPZs align by exact canonical residue identity and use a content hash.
- `batch_encode()` supports shallow `params_by_id` overrides; the CLI supports global parameters.
- Batch resume checks config, input, representation, and producer identity.
- Packing writes a metadata index consumed without rescanning the artifact root.
- Family-specific collates reject incompatible samples rather than repairing them.
- Torch export supports all explicit families and learned embeddings.
- PyG export accepts only explicit graph artifacts with public topology.

### GearNet-Edge

The backend implements frozen canonical-20 CA selection, native graph construction, strict
official-checkpoint loading, CPU/fp32 inference, biological node IDs, node/global/both outputs,
Schema v2 provenance, persistence, and cache reuse.

Frozen scientific behavior is specified in
[`gearnet_edge_backend.md`](gearnet_edge_backend.md). Do not change selection, topology,
checkpoint identity, model architecture, inference, mapping, or cache identity during release
hardening.

## Validation at handoff

Validation was run on 2026-09-29 in the local `eris` Conda environment:

- Python `3.12.14`;
- tracked suite: `377 passed, 4 skipped`, exit 0;
- collected tracked tests: `381`;
- Ruff check: passed;
- Ruff format check: `150 files already formatted`;
- `pip check`: no broken requirements;
- lazy-import boundary: `3 passed`;
- `git diff --check`: passed, with expected LF-to-CRLF working-copy warnings only.

The four skips require `ERIS_GEARNET_CHECKPOINT`, which was not configured. PyTorch was installed;
`torch_geometric` was not installed. These results do not certify official-checkpoint inference or
native PyG in a release environment.

Reproduce the tracked suite in Bash:

```bash
pt="${HOME}/eris-pytest-temp"
rm -rf "$pt"
mkdir -p "$pt"

python -m pytest -q \
  --basetemp "$pt/base" \
  -p no:cacheprovider
python -m ruff check .
python -m ruff format --check .
python -m pip check
git diff --check
```

Run `tests/test_packaging_boundaries.py` separately when checking optional-import isolation.

## Known environment and repository caveats

- `.github/workflows/ci.yml` references test modules absent from this commit:
  `tests/test_m8_release_boundaries.py`, `tests/test_gearnet_dependencies.py`,
  `tests/test_gearnet_native_graph.py`, `tests/test_gearnet_native_model.py`, and
  `tests/test_gearnet_inference.py`. The workflow must be repaired before it can serve as a release
  gate.
- The local validation environment lacks `torch_geometric`.
- The official GearNet checkpoint was not configured for the handoff validation.
- Row-specific batch overrides are Python-only.
- External features require exact residue identity; no sequence alignment is implemented.
- A generic pooled residue representation is not implemented.
- No generic GNN, Graph2Vec, GIN, GAT, or GraphSAGE family is included.
- Distribution name, final public metadata, citation file, and release notes remain release work.

## Release-hardening scope

1. Repair the CI workflow so every job invokes tests that exist, then run its Python 3.11/3.12
   matrix.
2. Decide and verify the public distribution name; update package URLs and release metadata as
   needed without changing import or CLI names.
3. Build wheel and sdist artifacts, inspect their contents, and test clean installations.
4. Run the full suite against installed distributions, not only an editable checkout.
5. Install native PyG and certify graph export in each supported release environment.
6. Supply the official GearNet checkpoint, verify its SHA-256, and certify frozen graph/embedding
   references and repeated CPU/fp32 inference across the supported matrix.
7. Freeze dependency policy and confirm the final public API and Python support statement.
8. Add `CITATION.cff` and real release notes/changelog only when release metadata is known.
9. Configure publishing, publish the chosen distribution, and create the GitHub release and
   associated archival record.

Scientific defects found during hardening should be reported against the candidate. Do not hide a
selection, identity, topology, or numerical mismatch by changing release policy or coercing data.

## Independent scientific check

Independent scientific and user validation should use a clean environment and exercise public
boundaries:

1. Encode representative PDB and mmCIF files with explicit chain/model selection.
2. Compare CA, mean, and center-of-mass residue positions, including missing-center failure/skip.
3. Inspect representative cloud, graph, matrix, grid, AAIndex, B-factor, and pLDDT outputs.
4. Test external row reordering plus missing, extra, duplicate, wrong-chain, wrong-model, and
   mutated-source failures.
5. Save/load, pack, dataset-load, and Torch/PyG-export artifacts after relocating source files.
6. Run GearNet `node`, `global`, and `both` modes and compare frozen references.

Escalate unexpected biological selection, alternate-location, structural-feature, or topology
semantics instead of normalizing them.

## Handoff criterion

The repository is ready for developer handoff. Publication still requires the release-hardening
work listed above.

The next maintainer should first read the permanent contracts, repair the stale CI test paths, and
reproduce the tracked validation in clean Python 3.11 and 3.12 environments. That establishes a
trustworthy release baseline before package naming, distribution builds, GearNet/PyG
certification, or publishing changes begin.
