# Changelog

All notable changes to Eris are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

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
- The `eris` CLI, with documented exit codes. `--config PATH` on `encode`, `embed`, `batch`, `features` and `export` reads option values from a YAML or JSON file (defaults < file < flags); `eris validate-config` and `eris.validate_config` check such a file without reading structures.
- Every successful CLI run writes `<output>.provenance.json` next to the output, following the shared ecosystem CLI contract.
