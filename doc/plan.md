# Development plan

Falsifier-first: each phase has a go/no-go gate. Do not proceed past a failed gate without addressing it.

## Phase 1 — Smoother diagnostic (the gate) — ✅ DONE
**Goal:** does point Gauss–Seidel keep smoothing the linearized p-Laplacian as p→1?
- `experiments/smoother_diagnostic.jl`, `experiments/deflation_sweep.jl` (base Julia only).
- **Result:** µ_s = 0.23 (p2) → 0.46 (p1.2) → 0.87 (p1.05, small ε); FLAT in coarse-space size K
  ⇒ genuine smoother problem, not coarsening. Point-GS works to p≈1.5, breaks at the sharp p→1 limit.
- **Gate: PASSED with a caveat** — direct FAS is viable for moderate p; the p→1 Cheeger regime needs
  a cut-aware smoother (Phase 2). Always symmetric GS.

## Phase 2 — Cut-aware smoother + single-vector FAS p-eigensolver
**Goal:** a smoother that holds µ_s < ~0.3 into p→1, and the FAS cycle for the 2nd p-eigenvector.
- **Smoother design (the genuine nonlinear-AMG contribution):** relax on AGGREGATES of strongly-
  coupled nodes (the near-flat intra-cluster edges), not pointwise. Options to try, in order:
  1. Aggregation/block relaxation on LAMG+ affinity aggregates (relax each aggregate's interior
     against its boundary) — reuses LAMG's aggregation directly.
  2. Line/path relaxation along strong-coupling chains.
  3. Kaczmarz-style edge relaxation weighted by conductance (FAMG already has a GS+Kaczmarz relaxer —
     inspect `famg_constrained.jl:relax_gs_kaczmarz!`).
- **FAS cycle:** pointwise/aggregate Gauss–Seidel–Newton on r = Δ_p x − λ|x|^{p-2}x; λ ← R_p(x) once
  per cycle; normalization ‖x‖_p=1 (Brandt §5.6); LAMG+ aggregation + Galerkin coarse operator.
- **Continuation:** p=2 → target via NLF-style law sequence; warm-start hierarchy (frozen-τ).
- **Validate** against a slow reference (dense inverse-power / SCF p-eigenvector) on small graphs.
- **GATE:** two-level factor < ~0.3 into p≈1.1 with the new smoother; FAS cycle count O(1),
  gap-independent. If the smoother can't get there, restrict the claim to the p∈[1.3,2] regime.

## Phase 3 — Subspace (d-dim) FAS eigensolver — the embedding
**Goal:** first d nontrivial p-eigenvectors as an embedding.
- Block FAS: d vectors, EIS/BAMG least-squares interpolation capturing the d-dim low subspace
  (caliber c, k≈2c test vectors; refit per cycle; affinity-aligned aggregation keeps PᵀAP sparse).
- Coarsest-level Ritz projection (dense p-eigenproblem) for subspace orthogonality; deflate constant.
- **GATE:** embedding matches a reference Laplacian-Eigenmaps (p=2) result; p→1 sharpens clusters
  (measurable via cut conductance / cluster separation); O(m) scaling holds.

## Phase 4 — Scale + benchmark
- kNN graphs: MNIST/FashionMNIST; single-cell (Tabula Muris / 10x), 10k → 1M+ nodes.
- Baselines: Laplacian Eigenmaps (ARPACK), UMAP, t-SNE (for the embedding-quality story).
- Metrics: O(m) wall-clock & memory; DETERMINISM (run-to-run variance vs UMAP/t-SNE random init);
  kNN-preservation / trustworthiness / cluster ARI-NMI vs known labels.
- **GATE:** near-linear scaling; decisively lower run-to-run variance than UMAP; comparable/better
  quality.

## Phase 5 — Paper
- Venue: JMLR / NeurIPS (ML) or SISC (numerics).
- Story: "Deterministic near-linear nonlinear spectral embedding via direct FAS multigrid" — the
  cut-aware smoother is the technical core; determinism/reproducibility is the applied hook.
- Honest positioning vs Kushnir–Galun–Brandt 2010 (p=2 multigrid spectral clustering): our novelty is
  the nonlinear p→1 regime + the cut-aware smoother + continuation-in-FMG + d-dim embedding.

## Parked
- Dense-repulsion Elastic-Embedding variant + fast summation (Brandt–Lubrecht LOP55, MuST LOP164,
  Livne–Wright RBF) — only if graph-only repulsion proves insufficient for embedding quality.
