# MORPHEO Full Definitive Checklist

## 0) Release hygiene, packaging y repo publicable

* **Build**

  * `python -m build` genera wheel + sdist
* **CI**

  * ruff + tests (Py 3.10–3.12)
  * matrix con/ sin `torch_geometric` (skips correctos)
* **DoD global**

  * `pip install -e .`
  * `morpheo --help`
  * `pytest -q`
  * CI verde + build OK

---

## 2) Contrato único de artefactos (StructureArtifact)

* **Formato obligatorio**

  * `data.npz` + `meta.json` siempre
* **Meta mínima obligatoria**

  * `schema_version`
  * `encoder_name`, `encoder_version` (o git hash), `encoder_fingerprint`
  * `type` (graph/matrix/cloud/grid/mesh)
  * `level` (residue/ca/atom)
  * `family` (residue_graph, atom_graph, matrix, cloud, grid, mesh)
  * `params` (config del encoder)
  * `input` (source, chain_id, model_index, format)
  * `shapes`, `dtypes`
  * `counts` (`n_nodes`, `n_edges`, etc.)
  * `feature_keys` (`node_attr_keys`, `edge_attr_keys`, `graph_attr_keys`, etc.)
* **Validación**

  * `validate_artifact(...)` + errores accionables
* **Inspección**

  * `morpheo info <artifact_dir>` (imprime + valida)
* **DoD**

  * round-trip exacto `save/load`
  * exporters/datasets no usan heurísticas

---

## 3) Registry y taxonomía de encoders

* **Registro único**

  * `EncoderInfo`: `name`, `type`, `level`, `family`, `description`, `params_schema`
* **Fingerprint**

  * `fingerprint()` obligatorio (hash params + versión)
* **CLI**

  * `morpheo encoders` tabla (y `--verbose`)
* **DoD**

  * lo que dice el registry coincide con lo que se escribe en `meta.json`

---

## 4) Features determinísticos completos (residue + atom + graph)

### 4.1 Residue node features

* `aa_onehot`, `aa_index`
* `physchem` (set fijo documentado)
* `aaindex_basic` (subset fijo)
* `plddt_ca` / `bfactor_ca` (missing-policy clara)
* flags: `is_gly`, `is_pro`, `is_unk`, `has_ca`
* **DoD**: `build_node_attr(mode=...)` con keys estables + tests

### 4.2 Atom node features

* `elem_onehot`
* `role_onehot` (backbone/sidechain/ligand/water/ion/unknown)
* `occ`, `bfactor`
* flags: `is_hetero`, `is_h`
* `residue_link` (res_index/res_id) opcional pero recomendado
* **DoD**: `build_atom_node_attr(mode=...)` + tests

### 4.3 Edge features (graphs)

* `dist`
* `rbf` (K + params en meta)
* `seq_sep` (si hay mapping a residuo)
* flags: `same_residue`, `is_local`, `is_long_range`, `is_contact` (cuando aplique)
* `E=0` soportado
* **DoD**: `build_edge_attr` valida shapes/dtypes

### 4.4 Graph attributes (determinísticos)

* `n_nodes`, `n_edges`, `density`
* degree stats (min/mean/max/std)
* edge length stats (min/mean/max/std)
* component proxy (ej. giant component ratio) si aplica
* **DoD**: `graph_attr` + `graph_attr_keys` presentes en `*_ns`

---

## 5) Encoders FULL (por tipo, incluyendo nuevos)

### 5.1 Residue-level Graph encoders

* `residue_contact_graph`
* `residue_knn_graph`
* `residue_contact_graph_ns`
* `residue_knn_graph_ns`
* **Nuevos**

  * `residue_radius_graph_ns`
  * `residue_sequential_graph_ns` (baseline secuencial)
* **DoD**

  * todos con `node_attr`, `edge_attr`, `graph_attr`
  * export torch + pyg sin hacks

### 5.2 Atom-level Graph encoders (pendiente fuerte, “final”)

* `atom_knn_graph`
* `atom_knn_graph_ns`
* **Nuevos**

  * `atom_radius_graph_ns`
  * `atom_contact_graph_ns` (opcional)
* flags estándar:

  * `include_hetero`, `include_h`, `drop_water`, `keep_ions`
* **DoD**

  * corre en PDB real sin quedarse vacío
  * tests con combinaciones de flags
  * export torch + pyg

### 5.3 Matrices (CA)

* `distogram_ca`, `distogram_bins_ca`
* `contact_map_ca`, `contact_map_bins_ca`
* `bin_edges` en meta cuando corresponda
* **DoD**

  * dataset `prefer="matrix"` + `collate_matrix` estable

### 5.4 Clouds (point representations)

* `residue_point_cloud_ca` (con `cloud_x_mode="auto"`)
* **Nuevo**

  * `atom_point_cloud` (con filtros)
* **DoD**

  * `prefer="cloud"` + `collate_cloud_pad` estable

### 5.5 Grids (voxel)

* `voxel_grid_ca` finalizado con schema estable:

  * `origin`, `voxel_size`, `channels`, `centroid`, `grid_shape`
* **Opcional pro**

  * `voxel_grid_atoms` (por canales de elemento/ocupación)
* **DoD**

  * `prefer="grid"` + `collate_grid_pad`

### 5.6 Mesh (si “full final” incluye mesh sí o sí)

* `protein_surface_mesh` (experimental o core, pero implementado)
* outputs: `verts`, `faces`, `normals` (+ `v_attr` básico)
* **DoD**

  * 1 encoder mesh + export torch + notebook demo

---

## 6) Embeddings (full, pero sin hacerlos requisito del core)

### 6.1 Política y soporte core

* embeddings **opcionales** (inyección externa)
* contract: si existe `node_emb (N,D)` → se guarda con keys + dims + meta
* **Stats determinísticos de embedding**

  * `embedding_stats → graph_attr` (mean/std/norms)
* **DoD**

  * pipeline funciona sin embeddings
  * si embeddings existen: export + dataset + pack lo soportan

### 6.2 PLM residue embeddings (extra)

* módulo `embeddings/` con interfaz estable (lazy import)
* ejemplo con 1 PLM (ESM2 o el que ya usas en Sylphy) pero como **extra**
* **DoD**

  * `pip install morpheo[plm]` habilita demo
  * sin extra, no rompe nada

### 6.3 Atom embeddings (realista)

* nativo: **no obligatorio**
* soportar inyección externa `(N_atoms,D)` para `atom_*`
* **DoD**

  * docs “cómo inyectar” + un ejemplo

---

## 7) Exporters finalizados

### 7.1 Torch exporter (todos los tipos)

* graph: `x`, `edge_index`, `edge_attr`, `graph_attr`, `pos` si existe
* matrix: `x` (+ `bin_edges`)
* cloud: `pos` + `x` opcional
* grid: `x` + meta por sample
* mesh: `verts`, `faces`, `v_attr`
* **DoD**

  * tests por cada tipo

### 7.2 PyG exporter (graphs)

* residue + atom graphs
* mapeo robusto `node_attr → x` (fallback explícito)
* `graph_attr` como tensor en `Data`
* **DoD**

  * tests: `residue_*_ns` y `atom_*_ns`

---

## 8) Dataset, collates y pack indexing (uso real)

### 8.1 `morpheo pack`

* index con columnas estándar:

  * `id`, `encoder`, `type`, `level`, `family`, `artifact_dir`
  * dims: `node_dim`, `edge_dim`, `graph_attr_dim`
  * counts: `n_nodes`, `n_edges`
  * matrix/grid/cloud specifics (shape/channels/bin_edges flag)
* **DoD**

  * filtrar por tipo/encoder sin abrir npz

### 8.2 `MorpheoFolderDataset`

* `prefer=graph|matrix|cloud|grid|mesh`
* carga lazy + validación
* **DoD**

  * ejemplos DataLoader por tipo

### 8.3 Collates

* graph strict
* graph pad-features (mezcla)
* matrix pad
* cloud pad
* grid pad
* mesh collate (si mesh)
* **DoD**

  * tests mínimos por collate

---

## 9) CLI y UX para usuarios reales

* `morpheo encode`

  * `--encoder`, `--chain`, `--model-index`, `--params`, `--no-cache`, `--validate`
* `morpheo batch`

  * manifest CSV/TSV, resume, report de errores
* `morpheo pack`
* `morpheo encoders`
* `morpheo info`
* `morpheo cache clear` + `cache stats`
* **DoD**

  * 3 comandos reproducibles en README que generan artifacts + pack + dataset

---

## 10) Logging, errores y “polish”

* logging consistente (core + encoders + CLI)
* errores accionables (ej. “missing CA”, “E=0”, “no atoms after filtering”)
* ruff (check + format) y estilo consistente
* **DoD**

  * `ruff check .` + `ruff format .` OK

---

## 11) Tests finales (load-bearing, no miles)

* IO: PDB + CIF
* Encoders: 1 por tipo (graph/matrix/cloud/grid) + atom graph + mesh (si existe)
* Exporters: torch + pyg
* Dataset/pack: pack → dataset → DataLoader
* Edge cases:

  * `E=0`, `N pequeño`, missing plddt, no CA, filtrado deja 0 átomos
* **DoD**

  * `pytest -q` en limpio, CPU-only; skips correctos

---

## 12) Docs y notebooks (lo que hace que sea integrable)

### 12.1 README (must)

* instalación + extras
* quickstart CLI (encode → pack → dataset)
* quickstart Python (torch + pyg)
* tabla de encoders (registry)
* **DoD**

  * alguien externo reproduce sin leer tu mente

### 12.2 Docs mínimas (4–6 páginas)

* artifacts schema
* encoders taxonomy
* exporters
* dataset/pack/collates
* embeddings policy + inyección
* mesh (si aplica)

### 12.3 Notebooks (end-to-end)

* `01_quickstart_encode_export`
* `02_graph_gdl_ready` (PyG)
* `03_voxel_grid_ready`
* `04_atom_graph_ready`
* `05_mesh_surface_ready` (si mesh)
* **DoD**

  * corren con `tests/data/`

---

# DoD final de “librería full finalizada”

1. `pip install -e .` y CLI usable
2. encoders residue + atom + matrix + cloud + grid (+ mesh si se exige)
3. artifacts con schema + validación + `morpheo info`
4. exporters torch + pyg (y mesh torch si aplica)
5. pack + dataset + collates funcionando por tipo
6. docs + notebooks reproducibles con data del repo
7. tests + CI + build verde
