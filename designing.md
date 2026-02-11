## 1) Node descriptors “bio” sin Sylphy ni Roxy

### Qué meter (mínimo pero útil)

**A) aa_index (categorical)**

* `aa_index` (0–20) ya lo tienes

**B) one-hot**

* 21 dims (incluye UNK)

**C) physicochemical descriptors (hardcoded)**
Una tabla pequeña (20 AA) con 6–10 propiedades clásicas, por ejemplo:

* hydrophobicity (Kyte-Doolittle)
* volume (o molecular weight)
* polarity
* charge (−1/0/+1)
* aromatic (0/1)
* H-bond donor/acceptor proxy (0/1)
* aliphatic index proxy (0/1)

**D) confidence**

* `plddt` si existe
* `bfactor_ca` si existe (normalizado a 0–1 si viene tipo 0–100)

**Resultado**: `node_attr` tipo `(N, F)` siempre determinístico.

### Implementación recomendada (estructura)

* `morpheo/features/amino_acids.py`

  * tabla `AA_TO_INDEX`, `INDEX_TO_AA`
  * `AA_PHYS` → `np.ndarray shape (21, P)` con UNK = 0
* `morpheo/features/node.py`

  * `build_node_features(aa_index, plddt=None, bfactor=None, mode="onehot+physchem+plddt")`

---

## 2) Residue embeddings sin Sylphy/Roxy

Como no quieres PLMs, hay dos opciones “Morpheo-native”:

### Opción 1: “static embedding table” (learnable en training, exportable como indices)

* Morpheo guarda `aa_index` y (opcional) `physchem`
* El embedding real lo hace el modelo downstream (`nn.Embedding(21, d)`)

👉 Esto es lo más limpio: Morpheo no entrena nada, solo exporta inputs.

### Opción 2: “frozen embeddings” (determinístico)

No lo recomiendo ahora, porque te obliga a elegir un método (ProtVec k-mer, etc.) y eso se empieza a parecer a Sylphy-lite.

**Mi recomendación**: quédate con **Opción 1**. Morpheo entrega índices + descriptors.

---

## 3) Edge descriptors mejores (baratos y muy útiles)

Para tus grafos `*_ns`, agrega estas features en `edge_attr`:

* `dist` (ya)
* `rbf(dist)` con K bins (ya lo tienes en PyG export, pero mejor generarlo en el encoder si quieres)
* `seq_sep = |i-j|` (si puedes mapear a posición secuencial)
* `is_local = seq_sep <= 4` (o similar)
* `same_chain` (por ahora siempre True si seleccionas cadena)
* opcional: `contact_bin` si usas cutoff/bins

Esto hace que el grafo sea ML-ready sin depender de nada.

---

## 4) “Graph embedding” sin Sylphy/Roxy

Acá hay que ser precisos: **graph embedding** puede significar 2 cosas.

### A) Graph-level vector determinístico (sin entrenamiento)

Esto sí lo puede hacer Morpheo y es útil para clustering/EDA:

* `n_nodes`, `n_edges`
* degree mean/std/max
* average edge length (mean/std)
* clustering coefficient (si te animas con networkx)
* spectral summary: top-k eigenvalues de Laplacian (barato si N ~ 500? puede ser pesado; top-10 con `scipy.sparse.linalg` si lo tienes)

Yo lo implementaría como:

* `morpheo/features/graph_stats.py -> compute_graph_stats(edge_index, edge_dist, n_nodes) -> dict`
  y guardarlo en `meta.json` (o como `graph_stats.npz` si prefieres).

### B) Graph embedding “real” (aprendido)

Eso **no debería vivir en Morpheo** (vive en el modelo GNN).
Lo que sí puedes hacer es dar “hooks” para pooling:

* En export PyG ya estás listo para usar `global_mean_pool`, `global_add_pool`, etc.
* Para cloud/grid/matrix igual.

**Mi recomendación**:

* implementa **(A) stats determinísticos** ahora
* deja (B) para training scripts, no Morpheo

---

## 5) Dónde lo meto en tus encoders actuales

Tus targets naturales:

* `residue_contact_graph_ns.py`
* `residue_knn_graph_ns.py`
* `residue_point_cloud_ca.py` (para `x` opcional en clouds)

Cambios mínimos:

* al final del encoder, además de `pos/residue_ids/edge_index/edge_dist`, agrega:

  * `aa_index`
  * `plddt` y/o `bfactor_ca` si existen
  * `node_attr` (construido por builder)
  * `edge_attr` (dist + rbf + seq_sep)
