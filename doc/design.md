> ⚠️ **STALE — superseded by `doc/formulation.md`.** This document describes an early
> p-Laplacian-eigenproblem framing we pivoted away from. It does NOT reflect the current genuine
> Elastic-Embedding formulation (dense Gaussian repulsion), the indefinite operator J = L⁺−μL⁻, or the
> deflation finding. Read `doc/formulation.md` for the authoritative current picture. Kept for history.

# Design: Nonlinear Spectral Embedding by Direct FAS Multigrid

**Goal.** Compute a low-dimensional embedding of an irregular graph (kNN data graph, social,
citation, single-cell) as the low nonlinear eigenspace of the graph **p-Laplacian**, solved by a
single **Brandt FAS eigenproblem cycle** (not stacked outer/inner loops), with **continuation in p**
embedded in the FMG hierarchy. Everything is a sum over the graph edges — **O(m)** per cycle. No dense
all-pairs term, no fast summation.

This is the "elastic embedding" idea reduced to its sparse, well-posed core: the anti-collapse
mechanism is a **normalization constraint** (an eigenproblem), not a dense repulsion. Laplacian
Eigenmaps is the p=2 linear special case; p→1 sharpens toward the Cheeger cut.

---

## 1. The mathematical problem

Graph G=(V,E), |V|=n, |E|=m, signed incidence B ∈ R^{n×m}, edge weights w>0. For an embedding
coordinate x ∈ R^n, the **p-Dirichlet energy** and **Rayleigh quotient** are

    E_p(x) = Σ_{(i,j)∈E} w_ij |x_i − x_j|^p = ‖ W^{1/p} Bᵀ x ‖_p^p
    R_p(x) = E_p(x) / ‖x‖_p^p

Stationary points of R_p (with normalization) satisfy the **nonlinear eigenproblem**

    Δ_p x = λ |x|^{p-2} x ,      Δ_p x := B diag( w |Bᵀx|^{p-2} ) Bᵀ x = B ρ_p(Bᵀx)          (★)

with eigenvalue λ = R_p(x) at a critical point. Note Δ_p is exactly the NLF source-form operator
B ρ_p(Bᵀx); the eigenproblem adds the RHS λ|x|^{p-2}x and the normalization ‖x‖_p = 1.

- **p = 2:** (★) is the linear graph-Laplacian eigenproblem L x = λ x. The first nontrivial
  eigenvector is the Fiedler vector; the first d span the Laplacian-Eigenmaps embedding.
- **p → 1:** thresholding the second p-eigenvector converges to the optimal **Cheeger cut**
  (Bühler–Hein) — sharper cluster separation than p=2.

**Embedding.** We want the first d nontrivial p-eigenvectors X = [x^(1),…,x^(d)] ∈ R^{n×d}, each
orthogonal to the trivial constant mode and mutually (p-)orthonormal:

    minimize  Σ_{k=1}^d R_p(x^(k))   subject to   Xᵀ D X = I_d,  X ⊥ 1                          (SUB)

The constraint prevents collapse to 0 and prevents the d vectors from coalescing. (D = degree /
weight matrix; the exact normalization is a design choice, §7.)

---

## 2. Why direct FAS, not stacked loops (Brandt's principle)

The naive route is three nested loops: outer eigensolver (inverse power / SCF) → middle continuation
in p → inner linear solve. Brandt's "solve the original problem" principle (1984 Guide §13.1) says
collapse them:

> "Instead of solving an eigenproblem by the inverse power method, with multigrid as the fast
> inverter, you can multigrid directly the original eigenvalue problem (§8.3.1). Instead of using
> multigrid for solving each step in some outer iterative process … apply it directly to the
> originally given problem."

Reason: the outer-loop iteration counts multiply. Inverse-power count scales like 1/(spectral gap),
which → ∞ for near-disconnected (Cheeger) graphs; the p-continuation count scales like 1/(p−1). A
direct FAS cycle replaces the product of these counts with an **O(1) cycle count governed by a single
number — the smoother rate µ_s** — independent of the gap and of the continuation length
(Brandt–McCormick–Ruge 1983, LOP33).

---

## 3. FAS eigenproblem architecture

One FMG cycle on the nonlinear system (★) + normalization. Per level:

- **Smoother.** Pointwise Gauss–Seidel–Newton on the residual r = Δ_p x − λ|x|^{p-2}x with λ frozen:
  update each x_i by one Newton step of the scalar equation r_i(x_i)=0 using its graph neighbors.
  (Chord variant: freeze the edge weights w|Bᵀx|^{p-2} across the sweep — NLF's chord-Newton.)
- **Eigenvalue update.** λ ← R_p(x), recomputed once per cycle after coarse-grid visits (LOP33 §5.3).
- **Global constraint.** ‖x‖_p = 1 (and XᵀDX=I for the subspace) enforced by the §5.6 global-constraint
  mechanism, active mainly at the coarsest levels; on the finest levels the constraint is carried by
  the coarse correction, not re-imposed every sweep.
- **Coarsening.** LAMG+ affinity aggregation (algebraic, irregular-graph native). Affinity ≈ diffusion
  distance from a few relaxed test vectors → aggregates respect emerging clusters, do NOT cross cuts.
- **Interpolation.** EIS / BAMG least-squares to capture the low-mode SUBSPACE, not just the constant
  (§4). Caliber kept low; refit per cycle.
- **Coarse-grid correction.** FAS: coarse problem is the fine residual restricted, plus the coarse
  operator's own action (τ-correction), with a coarse eigenvalue and coarse normalization.

The linearized operator at the current iterate is B diag(w|Bᵀx|^{p-2}) Bᵀ — a **weighted graph
Laplacian**, LAMG+'s native object. So LAMG+ machinery (aggregation, smoothing, cycle) is reused
directly; the nonlinearity is handled by FAS, not by an outer Newton loop.

---

## 4. Interpolation design (the load-bearing coarse-space decision)

For the linear Laplacian *solve* interpolation need only reproduce the constant. For the
*eigenproblem* the coarse correction must represent the whole low-mode subspace {v_1=1, v_2=Fiedler,
…, v_d}. Corpus-grounded rules (Bootstrap AMG LOP163; EIS = Exact Interpolation Scheme, Kushnir–
Galun–Brandt LOP154):

- **Caliber ↔ subspace dim:** caliber-c least-squares interpolation reproduces a c-dimensional local
  subspace exactly. Unique LS solution iff #test-vectors k ≥ c; use **k ≈ 2c** for robustness
  (LOP163 §2.1). EIS shows **caliber-2 with ~8 test vectors** already fits many eigenvectors
  (LOP154); caliber-4 gives no substantial gain.
- **Do NOT** raise caliber to d or carry d DOFs per aggregate (smoothed-aggregation style) — both
  densify PᵀAP and break O(m) (LAMG LOP168 §3.1.3). Instead: **one low-caliber P, refit each cycle
  to all d test vectors at once**, with **affinity-aligned aggregation** so overlapping coarse
  neighborhoods keep PᵀAP sparse even for a d-dim fit.
- **Bootstrap resolves the chicken-and-egg** (need eigenvectors to coarsen, need coarsening to get
  eigenvectors): crude P → crude test vectors → multigrid eigensolver → better test vectors → better
  affinity → better P. The number of test vectors the current P fails to reproduce **measures** d.

---

## 5. Continuation in p, embedded in FMG

Not an outer loop. p advances as FMG proceeds to finer levels (1984 Guide §8.3.2):

    coarsest levels:  p ≈ 2   (linear, well-conditioned = Laplacian Eigenmaps, solvable exactly)
    finest level:     p → 1   (nonlinear, sharp cut features)

Start from the deterministic **spectral (p=2) embedding**, deform toward p→1. A symmetry-breaking
bifurcation may appear where the trivial branch loses stability; detect the singular linearized
operator (smallest eigenvalue → 0) and branch-switch along the null eigenvector, then pseudo-arclength
continue (NLF machinery). Warm-start the aggregation across p-steps (frozen-τ) so setup does not
dominate. Determinism of the embedding comes from starting at the spectral solution and tracking a
single branch.

---

## 6. The load-bearing risk — smoother degeneracy as p → 1 (FIRST experiment)

As p→1, edges where the eigenvector is FLAT (intra-cluster, |Bᵀx|_e → 0) get linearized weight
w|e|^{p-2} → ∞ (p−2 < 0), while cut edges (large |Bᵀx|) get small weight. The linearized Laplacian
becomes highly anisotropic (near-infinite intra-cluster coupling), and **point Gauss–Seidel degrades**
(Brandt's degenerate-coefficient regime, 1984 Guide §10).

**This is the gate for the whole approach.** Measure µ_s(p): freeze x at a good iterate, form
L(x,p) = B diag(w|Bᵀx|^{p-2}) Bᵀ, run GS on L(x,p) v = 0, measure the asymptotic reduction of the
error orthogonal to the bottom-K eigenspace (the part the coarse grid can't fix).

**MEASURED (2026-07-04, 2-block SBM n=600; experiments/smoother_diagnostic.jl + deflation_sweep.jl):**

| p | µ_s (symmetric point-GS) |
|---|---|
| 2.0 | 0.23 (excellent) |
| 1.5 | ~0.28–0.33 |
| 1.2 | 0.46 (marginal) |
| 1.1 | 0.55 (slow) |
| 1.05, ε=1e-4 | 0.87 (defeated) |

Two robust findings:
1. **Genuine smoother problem, NOT a coarse-space problem.** µ_s is FLAT from K=2 to K=32 — removing
   more low modes via the coarse grid does not help. Ideal K-mode deflation upper-bounds what real
   coarsening achieves, so the bottleneck is the smoother. Classic anisotropy: point relaxation cannot
   smooth across the strong (intra-cluster) coupling the p→1 weights create.
2. **Degraded, not dead; ε is a real knob.** Point-SGS works to p≈1.5, is usable-but-slow at p≈1.2,
   breaks only in the extreme p→1 Cheeger limit with small regularization.

**Consequence (drives Phase 2):** a **cut-aware / aggregation smoother** — relax on aggregates of
strongly-coupled nodes (or block/line relaxation aligned with the strong-coupling direction) instead
of pointwise — is required to reach the sharp p→1 Cheeger regime. This smoother is the project's
genuine nonlinear-AMG contribution; the diagnostic proves it necessary, not speculative.

CAVEAT: always use SYMMETRIC GS (forward+backward). An earlier "µ_s≈0.19" reading came from a bogus
`reverse!` proxy that is not a valid backward sweep — discarded.

---

## 7. Design choices to pin down during build
- Normalization: ‖x‖_p = 1 vs xᵀDx = 1 (generalized). p=2 → generalized gives Laplacian Eigenmaps.
- Subspace orthogonality for p≠2: p-orthogonality is not an inner product; use Ritz projection at the
  coarsest level (solve the small dense p-eigenproblem there) as the practical mechanism.
- Deflation of the constant mode 1 (always an exact null vector of Δ_p).

---

## 8. Related work & honest novelty
- **Kushnir–Galun–Brandt 2010 (LOP154)** already built multigrid spectral clustering **at p=2** (EIS +
  affinity aggregation + multigrid eigensolver). We must NOT claim "multigrid for spectral clustering."
- **Our novelty:** (i) the **nonlinear p→1 regime** (nobody has done direct FAS on Δ_p x = λ|x|^{p-2}x
  toward the Cheeger cut); (ii) a **modern near-linear engine** (LAMG+/approxChol) as the linearized
  inner operator; (iii) **continuation in p embedded in FMG** with branch-switching through the
  bifurcation; (iv) delivering a **d-dimensional embedding** (subspace), not just a 2-way cut.
- Bühler–Hein 2009: p→1 → optimal Cheeger cut (the "why p<2 is worth it" theorem).

## 9. Reused machinery
- **LAMG+** (LAMG.jl): affinity aggregation, low-degree elimination, smoothing, V-cycle — the linear
  engine for the frozen weighted Laplacian and the coarse hierarchy.
- **NLF**: source-form p-Laplacian operator B ρ_p(Bᵀx), chord-Newton, continuation-in-p, arclength.
- We add: the FAS eigenproblem cycle, the Rayleigh/λ update, the global normalization constraint, the
  EIS subspace interpolation, and the p→1 smoother (if the diagnostic demands one).

## 10. Parked — dense-repulsion variant (future)
If graph-only repulsion proves insufficient for embedding quality, revisit the DENSE Elastic-Embedding
repulsion Σ_{all n,m} w⁻ exp(−‖x_n−x_m‖²), evaluated in O(n) by fast summation:
Brandt–Lubrecht 1990 (LOP55), Brandt 1991 particle interactions (LOP59), Livne–Brandt MuST 2010
(LOP164), and Livne–Wright fast RBF evaluation. Gaussian in d=2,3 is the smooth, grid-only best case.
Not needed for the sparse eigenproblem build.
```
```
