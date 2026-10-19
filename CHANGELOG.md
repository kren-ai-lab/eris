# Changelog

All notable changes to Eris are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## 0.2.0

- `--config PATH` on `encode`, `embed` and the `batch` commands reads option values from a YAML or JSON file; precedence is defaults < file < command-line flags. Keys are option names, and the representation parameters of `--params` go under a nested `params:` mapping (a `--params` flag replaces it as a whole). Unknown keys and wrongly typed values exit with code `2`.
- `eris validate-config COMMAND CONFIG [--json]` checks a config file, including `params` against the representation's own configuration, and reports every problem with a dotted path. It exits `0` if valid and `2` otherwise, and reads no structures.
- `eris.validate_config(config, target)` and `eris.ConfigIssue` expose the same check in Python (`config` is a mapping or a YAML/JSON path, `target` as on the CLI). It does not raise for config problems and does not import model code.
- A text or path option given a non-string in a config (`chain: 5`) is reported instead of being coerced. The config key for `--model` is `model`.
- A missing manifest (`batch encode`, `batch embed`) or folder (`batch from-folder`) exits `3`, as a missing structure does; a missing config file stays `2`.
- Successful `encode`, `embed` and `batch` runs write `<output>.provenance.json` next to the output, following the shared ecosystem CLI contract.
- Batch artifacts record the manifest `id` as `input.structure_id` and the run time as `provenance.created_at` in `meta.json`; the artifact schema version is unchanged.

## 0.1.0

First public release, published on PyPI as `erislib`.

- Ten explicit representations from PDB/mmCIF: atom and residue point clouds, atom and residue
  k-NN graphs, residue contact graphs, distograms, contact maps (plain and binned) and residue
  voxel grids, with configurable residue, edge and graph features.
- GearNet-Edge residue and structure embeddings (`eris.embed`, optional extra `erislib[gearnet]`)
  with automatic, SHA-256-verified download of the official checkpoint.
- `StructureArtifact` persistence (`meta.json` plus `data.npz`) with schema validation,
  provenance and a content-addressed cache.
- `batch_encode` and `batch_embed` over CSV/JSONL manifests or Polars DataFrames, with resume and
  per-row failure records; `pack` indexes artifacts for `ErisFolderDataset`.
- `feature_table` collects structure-level vectors (GearNet global embeddings, pooled features,
  graph statistics) into an `id` + features Polars DataFrame for tabular models.
- Export with `to_torch` and `to_pyg` (optional extra `erislib[pyg]`), and residue pooling with
  `pool`.
- The `eris` CLI, with documented exit codes.
