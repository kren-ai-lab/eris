# Eris documentation

Eris converts PDB and mmCIF structures into reproducible explicit structural representations and
GearNet-Edge learned embeddings. The explicit surface covers atom and residue point clouds,
nearest-neighbor and contact graphs, distance/contact matrices, and residue voxel grids. Outputs
are persisted as strict `StructureArtifact` Schema v2 directories and can flow through cache,
batch, pack, dataset, Torch, and PyG boundaries.

## Start here

- [Architecture](architecture.md) describes the explicit and learned representation families,
  canonical data flow, public boundaries, and workflow integration.
- [Representation defaults](representation_defaults.md) records coordinate, selection, feature,
  topology, binning, voxel, and dtype defaults for every explicit representation.
- [Artifact Schema](artifact_schema.md) defines `meta.json`, `data.npz`, scientific parameters,
  provenance, validation, persistence, and cache identity.
- [External residue features](external_residue_features.md) defines the strict generic `.npz`
  interface, residue alignment, content identity, hybrid composition, and per-row batch usage.
- [GearNet-Edge backend](gearnet_edge_backend.md) defines the learned backend, checkpoint,
  preprocessing, runtime, output, and reproducibility contracts.

## Workflows and development

Batch, pack, dataset, and export contracts are summarized in
[Architecture: Batch and pack contract](architecture.md#batch-and-pack-contract). Row-specific
external-feature examples and shallow `params_by_id` semantics are documented in
[External residue features: Per-row batch parameters](external_residue_features.md#per-row-batch-parameters).

For contributor setup and validation commands, see [DEVELOPMENT.md](../DEVELOPMENT.md).
Repository-specific agent constraints are in [AGENTS.md](../AGENTS.md).

## Internal handoff

[HANDOFF_V0.1.md](HANDOFF_V0.1.md) records the feature-complete transfer state, validation
evidence, limitations, and release-hardening ownership for Álvarez, Nicole, and Julián. It is an
internal handoff record rather than public API documentation.

The README remains the concise installation and public-API introduction; these documents provide
the detailed scientific and persistence contracts.
