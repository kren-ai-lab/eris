# Planificación realista (sin código aún)

Una hoja de ruta simple y honesta:

### **v0.0.1 (skeleton)**

* repo + packaging + CI + pre-commit + CLI mínima + interfaces (listo con lo de arriba)

### **v0.1.0 (MVP usable)**

* I/O real (PDB/mmCIF) + selección cadena/modelo
* Cache real por hash (input + encoder + params)
* 2 encoders baseline:

  * `distogram_ca`
  * `residue_contact_graph` o `residue_knn_graph`
* Artefactos: `npz + meta.json`
* Tests con 1–2 estructuras de ejemplo

### **v0.2.0 (diferencial)**

* soporte AF2/ESMFold metadata (pLDDT si existe)
* features por nodo/edge
* batch CLI
* integración suave con multimodal

# Morpheo Checklist Master

## A. Ingesta, parsing y normalización estructural

### A1. Inputs soportados

* [ ] PDB (.pdb) local
* [ ] mmCIF (.cif/.mmcif) local
* [ ] (Should) PDB ID → descarga (RCSB) con cache
* [ ] (Should) AlphaFold/ESMFold outputs (mmCIF) como input directo
* [ ] (Could) MMTF, PQR (si se necesita)

### A2. Selección y slicing

* [ ] Selección de modelo (model index)
* [ ] Selección de cadena(s)
* [ ] Selección por rango de residuos (start–end)
* [ ] Selección por lista de residuos (IDs)
* [ ] Soporte multi-chain (complejos) con “chain grouping”
* [ ] Estrategias para representar *interfaces* (chain A–B)

### A3. Sanitización

* [ ] Política de altloc (A/major occupancy)
* [ ] Manejo de residues incompletos (missing atoms)
* [ ] Manejo de missing residues (gaps)
* [ ] Política de heteroatoms: keep/remove ligands
* [ ] Política de waters (keep/remove)
* [ ] Política de hydrogens (keep/remove, add? opcional)
* [ ] Renumeración consistente (mapping original ↔ interno)

### A4. Coordenadas y referencia

* [ ] Coordenadas por átomo y por residuo (CA, centroid, backbone)
* [ ] Normalización opcional (centering, scaling)
* [ ] Frame local por residuo (orientations) (should)
* [ ] Unidades y consistencia (Å)

### A5. Quality metadata

* [ ] Propagar pLDDT (si existe en mmCIF AF2)
* [ ] Propagar B-factors, occupancy
* [ ] Propagar resolución experimental (si disponible)
* [ ] Flags de calidad por residuo y por estructura

---

## B. Artefactos, hashing, cache y reproducibilidad

### B1. Provenir y fingerprinting

* [ ] Hash de input (file hash o canonicalized content hash)
* [ ] Fingerprint de encoder (name + version + params)
* [ ] Registro de dependencias (versions: gemmi/biopython/torch, etc.)
* [ ] Provenance completo en meta (timestamp opcional)

### B2. Cache

* [ ] Cache on-disk por clave (input_hash + encoder_fp + selection)
* [ ] Cache invalidation por version bump
* [ ] Modo “no-cache”
* [ ] Cache stats (hits/misses)

### B3. Serialización de artefactos

* [ ] Convención única: `artifact.(npz|pt|pkl)+meta.json`
* [ ] (Should) Compatibilidad zarr/np.memmap para grandes volúmenes
* [ ] Validación de artefactos (schema)

---

## C. Representaciones estructurales (familias)

### C1. Mapas 2D y matrices (baseline fuerte)

* [ ] Contact map (binary) por umbral (CA–CA, heavy atom)
* [ ] Distance map / distogram (float)
* [ ] (Should) Multi-channel maps: distance + orientation + confidence
* [ ] (Could) Local environment maps (windowed)

### C2. Grafos (residuo y átomo)

**Graph construction (must)**

* [ ] Residue graph: nodes=residuos
* [ ] Edge por cutoff de distancia
* [ ] Edge por kNN en 3D
* [ ] Edge por contactos químicos (H-bond, salt bridges) (should)
* [ ] Multi-edge typing (contact, knn, sequential) (should)
* [ ] Directed vs undirected (config)

**Atomic graph (should)**

* [ ] Nodes=átomos, bonds según química + proximidad
* [ ] Opción “hybrid”: residues con subgraph atómico

### C3. Point clouds

* [ ] Point cloud de átomos (xyz + features)
* [ ] Point cloud de superficie (sampled points) (should)
* [ ] Point cloud de residuos (centroids / CA)
* [ ] (Should) Normal vectors para superficie
* [ ] (Could) FPFH / geometric descriptors

### C4. Grillas / Voxels

* [ ] Voxel grid 3D occupancy
* [ ] Multi-channel voxels (element type, charge, hydrophobicity)
* [ ] Config de resolución (Å/voxel) y bounding box
* [ ] (Could) Sparse voxel formats

### C5. Mesh / Surface

* [ ] Mesh generation (triangulated surface)
* [ ] Surface features (curvature, electrostatics proxy)
* [ ] Mesh sampling para GDL en superficies (should)

### C6. Representaciones “local frames”

* [ ] Orientations/residue frames (SE(3) equivariant support-ready)
* [ ] Edge direction vectors / relative position encoding

---

## D. Features de nodos (residuales y atómicos)

### D1. Residue node features (must set)

* [ ] AA identity (one-hot o index)
* [ ] Physchem descriptors (charge, hydrophobicity, volume)
* [ ] Secondary structure (DSSP o aproximado) (should)
* [ ] RSA/SASA (should)
* [ ] pLDDT / B-factor / occupancy (si disponible)
* [ ] Sequence position encodings (absolute/relative)
* [ ] Backbone torsions (phi/psi/omega) (should)

### D2. Atomic node features (must set para atomic)

* [ ] Element type (one-hot)
* [ ] Atom name / atom role (backbone/sidechain)
* [ ] Partial charge (si disponible) (could)
* [ ] Aromaticity / hybridization (could)
* [ ] (Should) Mask/flags por missing atoms

### D3. “Embeddings” como features (clave para ti)

* [ ] Residual embeddings desde pLMs (ESM2/ProtT5 via Sylphy) como node features
* [ ] Atomic embeddings (si tienes encoder atómico o reglas) (should)
* [ ] Fusion strategies: concat, sum, gated fusion, projection MLP (later)
* [ ] Normalización + control de dimensionalidad (PCA/linear proj) (should)

---

## E. Features de aristas (edge features)

### E1. Geométricas (must)

* [ ] Distancia (scalar)
* [ ] Δvector (x,y,z) o unit vector
* [ ] Distance encoding (RBF / Fourier features) (should)
* [ ] Sequential distance (|i−j|) si nodes=residuos

### E2. Químicas/biológicas (should)

* [ ] Contact type (H-bond, salt bridge, hydrophobic)
* [ ] Same chain vs cross-chain flag
* [ ] Interface flag (si complejo)
* [ ] Confidence aggregation (min/mean pLDDT edge)

### E3. Edge typing / relations (should)

* [ ] Multi-relational graph support (edge_type)
* [ ] Heterographs (DGL) (could)

---

## F. Graph embeddings y representaciones derivadas

### F1. Graph-level embeddings (must para “graph embedding”)

* [ ] Pooling: mean/sum/max sobre node embeddings
* [ ] Attention pooling (should)
* [ ] Set2Set / DeepSets pooling (should)
* [ ] Readout configurable (per graph)

### F2. Subgraph embeddings (should)

* [ ] Patch embeddings (local neighborhoods)
* [ ] Interface embeddings para complejos
* [ ] Domain-level embeddings (si hay segmentation)

### F3. Positional/structural encodings (should)

* [ ] Laplacian eigenvectors (graph positional encoding)
* [ ] Random walk features
* [ ] Weisfeiler–Lehman hashing features (could)

---

## G. Export “ready for PyTorch / GDL”

### G1. Formatos objetivo (must)

* [ ] **PyTorch**: `torch.save()` de dicts estandarizados
* [ ] **PyTorch Geometric (PyG)**: export a `torch_geometric.data.Data`
* [ ] (Should) **DGL**: export a `dgl.DGLGraph`
* [ ] (Should) `npz + meta.json` como formato universal intermedio

### G2. Dataset writing (must)

* [ ] Writer para “one artifact per structure” (folder layout)
* [ ] Manifest CSV/JSONL con paths + labels + splits
* [ ] Soporte de sharding (tar/zarr) (should)

### G3. DataLoader compatibility (must)

* [ ] `Dataset` wrapper (PyTorch) para cargar artifacts
* [ ] Collate functions (PyG batch, padding maps, voxel batching)
* [ ] Lazy loading + memory safety (mmap/zarr) (should)

---

## H. Batch processing y pipelines

* [ ] Batch CLI: procesa una lista (CSV/TSV) con inputs
* [ ] Multiprocessing / joblib / Ray (should)
* [ ] Resume/skip if exists (idempotencia)
* [ ] Reporte final: n_ok / n_failed + reasons
* [ ] Logs estructurados por item

---

## I. Configuración y esquemas

* [ ] Config “encoder spec” (pydantic / jsonschema)
* [ ] Validation fuerte + defaults
* [ ] Config versioning (schema_version)
* [ ] “Repro recipe” export: config + env versions + command line

---

## J. Testing y QA “nivel Sylphy”

### J1. Unit tests

* [ ] Registry
* [ ] Hashing determinista
* [ ] I/O con fixtures mínimos
* [ ] Export formats (PyG Data shape checks)

### J2. Integration tests

* [ ] Pipeline batch sobre 2–3 estructuras
* [ ] Cache hit reproducible

### J3. Lint/typing/CI

* [ ] Ruff + mypy + pytest en CI
* [ ] Pre-commit
* [ ] Coverage mínimo (target p.ej. 70% al inicio)

---

## K. Documentación y ejemplos (muy importante para estudiantes)

* [ ] “Concepts” doc: inputs/outputs/artefacts
* [ ] Tutorial 1: distogram/contact map
* [ ] Tutorial 2: residue graph → PyG Data
* [ ] Tutorial 3: point cloud / voxel (más adelante)
* [ ] Cookbook: cómo agregar un encoder nuevo (plugin)
* [ ] FAQ: errores típicos (missing chain, altloc, etc.)

---

# “Lista fuerte” de estrategias a incorporar (resumen por catálogo)

### Representations catalog

* **2D**: contact map, distogram, multi-channel maps
* **Graphs**: residue cutoff, residue kNN, atomic graph, interface graph, multi-relational
* **Point clouds**: atoms, residues, surface sampled points (+normals)
* **Grids/Voxels**: occupancy, multi-channel physicochem grids, sparse voxels
* **Mesh/Surface**: triangulated mesh, mesh GDL-ready
* **Frames**: local frames, orientation-aware edges

### Features catalog

* **Node (residue)**: AA + physchem + SS + SASA + torsions + quality + **Sylphy embeddings**
* **Node (atom)**: element + atom role + optional chemistry features + (atomic embeddings)
* **Edge**: distance + vector + RBF encoding + sequential distance + contact type + chain flags

### GDL outputs

* **PyG Data**
* **DGLGraph**
* **torch dict**
* Universal intermediate: `npz + meta.json` (+ manifest)

---

## Siguiente paso (para convertir esto en plan realista)

Yo lo bajaría a 3 niveles:

1. **MVP v0.1** (imprescindible para paper + usuarios):

* mmCIF/PDB I/O robusto
* distogram/contact map
* residue graph (cutoff o kNN)
* residue node features básicas + Sylphy embeddings
* edge distance + RBF
* export PyG Data + torch dict + npz

2. **v0.2**

* point clouds
* batch + dataloaders
* interface graphs
* quality/pLDDT support

3. **v1.0**

* voxels + mesh
* atomic graphs/hybrid
* positional encodings avanzados
* heterographs
