# Eris

[![PyPI](https://img.shields.io/pypi/v/erislib?style=flat-square)](https://pypi.org/project/erislib/)
[![PyVersions](https://img.shields.io/pypi/pyversions/erislib?style=flat-square)](https://github.com/kren-ai-lab/eris)
[![Tests](https://img.shields.io/github/actions/workflow/status/kren-ai-lab/eris/tests.yml?style=flat-square)](https://github.com/kren-ai-lab/eris/actions/workflows/tests.yml)
![License](https://img.shields.io/github/license/kren-ai-lab/eris?style=flat-square)

Eris is a Python library that turns protein structures (PDB and mmCIF) into numerical
representations for machine learning. It builds point clouds, graphs, contact maps, distograms and
voxel grids, and GearNet-Edge embeddings. Every result is a validated `StructureArtifact` that keeps
the arrays together with the structure selection, parameters and provenance that produced them.

Eris doesn't train models. Its artifacts feed PyTorch, PyTorch Geometric or any NumPy-based
pipeline.

## Installation

Eris supports Python 3.11 to 3.14.

```bash
pip install erislib
# or with uv:
uv add erislib
```

Optional extras:

```bash
pip install "erislib[gearnet]"   # GearNet-Edge embeddings (PyTorch)
pip install "erislib[pyg]"       # PyTorch Geometric export
```

For development setup with `uv`, see
[DEVELOPMENT.md](https://github.com/kren-ai-lab/eris/blob/main/DEVELOPMENT.md).

## Encode a structure

```python
import eris

artifact = eris.encode(
    "protein.pdb",
    representation="residue_knn_graph",
    chain_id="A",
    k=8,
    output="artifacts/protein_knn",
)
artifact.data["edge_index"].shape   # (2, n_edges)
artifact.meta["parameters"]         # canonical parameters, including defaults
```

Representation parameters are passed as keyword arguments and validated against the
representation's config. `eris.list_representations()` lists the representations and
`eris.get_representation_info(name)` shows one with its parameters and defaults. `output=` writes
the artifact as a directory (`meta.json` plus `data.npz`) that
`eris.core.artifacts.StructureArtifact.load(path)` reads back and validates.

| Name | Type | Granularity | Primary output |
| --- | --- | --- | --- |
| `atom_point_cloud` | cloud | atom | atom coordinates and features |
| `residue_point_cloud` | cloud | residue | residue coordinates and features |
| `atom_knn_graph` | graph | atom | atom k-nearest-neighbor graph |
| `residue_knn_graph` | graph | residue | residue k-nearest-neighbor graph |
| `residue_contact_graph` | graph | residue | distance-cutoff residue graph |
| `residue_distogram` | matrix | residue | pairwise distances |
| `residue_distogram_bins` | matrix | residue | distance-bin classes |
| `residue_contact_map` | matrix | residue | binary contacts |
| `residue_contact_map_bins` | matrix | residue | distance-bin contact classes |
| `residue_voxel_grid` | grid | residue | multi-channel 3D grid |

Residue coordinates can use the alpha carbon, the residue-atom mean or the center of mass. Residue
features can combine amino-acid identity, physicochemical descriptors, AAIndex1 records, raw
B-factor, explicitly interpreted pLDDT, backbone and side-chain torsions, local structural features,
and aligned external values. Residue graphs can add chain relations, relative directions and
interface annotations. `eris.pool(...)` turns residue features into one structure-level vector.

## Learned embeddings

```python
artifact = eris.embed("protein.pdb", representation="gearnet_edge", output_mode="both")
```

`gearnet_edge` emits residue embeddings (`node`), one structure embedding (`global`), or both. It
runs the official `mc_gearnet_edge.pth` checkpoint on CPU in fp32. The checkpoint isn't bundled: on
first use, Eris downloads it over HTTPS, checks its registered SHA-256 and caches it. To work
offline, pass a local copy with `checkpoint_path=` (`--checkpoint` in the CLI).

## Batches, datasets and export

```python
import polars as pl
import eris
from eris.datasets import ErisFolderDataset

manifest = pl.DataFrame({"id": ["p1", "p2"], "path": ["p1.pdb", "p2.cif"], "chain": ["A", None]})
results = eris.batch_encode(manifest, representation="residue_knn_graph", artifact_root="artifacts")
index = eris.pack("artifacts", "artifacts/index.csv")
dataset = ErisFolderDataset("artifacts/index.csv")   # loads each artifact on access
graph = eris.to_pyg(eris.encode("p1.pdb", representation="residue_knn_graph"))
```

A manifest is a CSV or JSONL file or a Polars DataFrame with `id` and `path` columns, and optional
`chain` and `model` (default: all chains of the first model). `batch_encode` and `batch_embed`
return one result row per input, record failures without stopping (unless `fail_fast=True`) and
can resume an interrupted run. `pack` writes a CSV index over the artifact directories.
`to_torch` converts any artifact to tensors; `to_pyg` only accepts graphs with an explicit
`edge_index`.

### Feature tables

`feature_table` turns one structure-level vector per artifact into a Polars DataFrame with an `id`
column and one column per feature, ready for tabular models such as
[Saber](https://github.com/kren-ai-lab/saber):

```python
embeddings = eris.batch_embed(manifest, representation="gearnet_edge", artifact_root="emb", output_mode="global")
eris.pack("emb", "emb/index.csv")
features = eris.feature_table("emb/index.csv")   # id, p_0 ... p_3071

dataset = saber.DatasetBundle(X=features.drop("id"), y=labels, sample_ids=features["id"])
```

It reads GearNet global embeddings (`p_0`, `p_1`, …) or `eris.pool(...)` vectors (named after
their features), or graph statistics with `source="graph_attr"`. It also accepts a mapping from id
to artifact. Every row must come from the same representation, parameters and model; mixing them
is an error.

## Command line

```bash
eris representations list
eris encode protein.pdb --representation residue_knn_graph --params '{"k": 8}' --out artifact_knn
eris embed protein.pdb --representation gearnet_edge --out embedding
eris batch encode manifest.csv --representation residue_contact_map --out artifacts
eris pack artifacts --out artifacts/index.csv
eris features emb/index.csv --out features.csv
eris info validate artifact_knn
```

Each command calls the matching Python function.

`encode`, `embed` and the `batch` commands accept `--config PATH`, a YAML or JSON file whose keys are the option names (`representation`, `device`, ...). Representation parameters go under `params:`. Precedence is defaults < file < flags, and a `--params` flag replaces the file's `params` as a whole. Paths (input, `out`, `results`, `checkpoint`) stay on the command line.

```yaml
representation: residue_knn_graph
params:
  k: 8
```

`eris validate COMMAND CONFIG [--json]` checks such a file without reading structures (`COMMAND` is `encode`, `embed`, `batch-encode`, `batch-embed` or `batch-from-folder`). It prints one `path: message` line per problem to stderr, or with `--json` a JSON array of `{path, message}` on stdout, and exits `0` if valid or `2` if not.

Every successful run also writes `<out>.provenance.json` next to the output (shared ecosystem CLI contract): tool version, effective parameters and their fingerprint, input and output hashes, and the environment. The artifact's own `meta.json` is unchanged in format.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success (batch commands also exit `0` when some rows fail; see the results file) |
| `2` | Invalid usage or configuration (missing argument or file, unknown representation, invalid parameters) |
| `3` | The operation failed (unreadable structure, invalid artifact, backend or cache error) |
| `4` | Unexpected internal error |

## Design principles

- Eris never silently pads, truncates, resizes, normalizes or clips values, and never invents
  identifiers or drops failed samples. Graph edges come only from the documented neighbor or
  cutoff rule, and exports never add topology that the artifact doesn't already contain.
- Inputs that don't make scientific sense fail with a clear error instead of being coerced.
- Sample identity, array order, feature order and keys are deterministic.
- An artifact records the structure selection, canonical parameters and provenance, and is validated
  every time it is saved or loaded.
- The cache is keyed by the input file's content and the full configuration, so a changed input or
  parameter never returns a stale result.
- `import eris` doesn't load PyTorch, PyTorch Geometric or the GearNet runtime; they load only when
  a feature needs them.
- At runtime, the only network access is the download of the registered GearNet checkpoint.

## Documentation and examples

The [documentation index](https://github.com/kren-ai-lab/eris/blob/main/docs/README.md) links the
technical references: representation defaults, structural features, artifact schema, external
residue features and the GearNet-Edge backend. The
[example notebooks](https://github.com/kren-ai-lab/eris/blob/main/examples/README.md) are written
in marimo. To open one:

```bash
uv sync --all-extras --group examples
uv run marimo edit examples/01_explicit_representations.py
```

## Citing and license

If you use Eris in published work, please cite it using
[`CITATION.cff`](https://github.com/kren-ai-lab/eris/blob/main/CITATION.cff), or the "Cite this
repository" button on GitHub. Eris is released under the
[MIT license](https://github.com/kren-ai-lab/eris/blob/main/LICENSE).

GearNet-Edge uses the official Multiview Contrast checkpoint from
[Zenodo record 10.5281/zenodo.7723075](https://doi.org/10.5281/zenodo.7723075). Eris provides a
native runtime for that checkpoint and did not train it.
