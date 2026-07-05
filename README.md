# ElasticEmbedding

**Nonlinear spectral embedding of irregular graphs via the graph p-Laplacian eigenproblem, solved by
a direct Brandt FAS multigrid cycle with continuation in p.**

A low-dimensional embedding of a graph (kNN data graph, social/citation network, single-cell) is
computed as the low nonlinear eigenspace of the graph **p-Laplacian**:

    Δ_p x = λ |x|^{p-2} x ,   Δ_p x = B diag(w |Bᵀx|^{p-2}) Bᵀ x

- **p = 2** is Laplacian Eigenmaps (linear spectral embedding).
- **p → 1** sharpens toward the optimal **Cheeger cut** (Bühler–Hein): tighter cluster separation.

Everything is a sum over the graph edges — **O(m)** per cycle. No dense all-pairs repulsion, no fast
summation. The anti-collapse mechanism is a normalization constraint (an eigenproblem), which is why
this is the sparse, well-posed core of the "elastic embedding" idea.

## Why this design

Instead of stacked loops (outer eigensolver → continuation in p → inner linear solve), we solve the
**original eigenproblem directly** with one Brandt FAS cycle (1983 FAS-eigenproblem; 1984 Guide §13.1).
The convergence rate is then governed by a single number — the smoother factor µ_s — independent of
the spectral gap and the continuation length. Continuation in p is embedded in the FMG hierarchy
(coarse levels ≈ p2, finest → p1). See [doc/design.md](doc/design.md) and
[doc/plan.md](doc/plan.md).

Reuses:
- **LAMG+** (github.com/orenlivne/lamgplus) — affinity aggregation, smoothing, V-cycle.
- **NLF** (github.com/orenlivne/nlf) — source-form p-Laplacian B ρ_p(Bᵀx), chord-Newton, continuation.

## Status — Phase 1 complete (the smoother gate)

The load-bearing risk was whether point Gauss–Seidel keeps smoothing as p→1 (where intra-cluster edge
weights w|Bᵀx|^{p-2}→∞ make the linearized Laplacian anisotropic). **Measured:**

| p | µ_s (symmetric point-GS) |
|---|---|
| 2.0 | 0.23 |
| 1.5 | ~0.30 |
| 1.2 | 0.46 |
| 1.1 | 0.55 |
| 1.05 | 0.87 |

Point-GS works to p≈1.5, is slow at p≈1.2, and is defeated in the sharp p→1 limit — and enriching the
coarse space (deflation K=2→32) does **not** help, proving it's a genuine smoother problem. So the
project's core contribution is a **cut-aware / aggregation smoother** for the p→1 regime.

```
julia experiments/smoother_diagnostic.jl   # µ_s(p) table + verdict
julia experiments/deflation_sweep.jl       # µ_s vs coarse-space size (K) — isolates smoother vs coarsening
```

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Smoother diagnostic (go/no-go gate) | ✅ done |
| 2 | Cut-aware / aggregation smoother; single-vector FAS p-eigensolver | next |
| 3 | Subspace (d-dim) FAS eigensolver, EIS interpolation, continuation in p | |
| 4 | Scale + benchmark on kNN / single-cell graphs vs Laplacian Eigenmaps, UMAP | |
| 5 | Paper | |

## Layout
- `src/ElasticEmbedding.jl` — graph primitives, p-Laplacian operators, smoothers, diagnostics (Phase 1)
- `experiments/` — runnable diagnostics
- `doc/design.md` — full mathematical design
- `doc/plan.md` — phased development plan with gates
- `test/` — unit tests
