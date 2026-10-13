# Eris documentation

Start with the [quickstart](../README.md) to install Eris and encode a structure.

| I want to… | Read |
| --- | --- |
| Understand the package layout, data flow and public API | [Architecture](architecture.md) |
| Look up a representation's default parameters and semantics | [Representation defaults](representation_defaults.md) |
| Select chains, derive geometry or interfaces, or pool features | [Structural features](structural_features.md) |
| Read or validate a persisted artifact | [Artifact schema](artifact_schema.md) |
| Attach my own per-residue values | [External residue features](external_residue_features.md) |
| Compute GearNet-Edge embeddings | [GearNet-Edge backend](gearnet_edge_backend.md) |
| Explore complete workflows | [Example notebooks](../examples/README.md) |
| Set up a checkout, change code and run checks | [Development guide](../DEVELOPMENT.md) |

## Python API help

The public functions live in the package root: `eris.encode`, `eris.embed`, `eris.pool`,
`eris.batch_encode`, `eris.batch_embed`, `eris.pack`, `eris.feature_table`, `eris.to_torch` and
`eris.to_pyg`, plus the
discovery helpers `list_representations`, `get_representation_info`, `list_embedders` and
`get_embedder_info`. Use Python help for the signatures and defaults of your installed version:

```python
import eris

help(eris.encode)
eris.get_representation_info("residue_knn_graph")
```
