# Elastic-Embedding Multigrid — Current Formulation & Solver (authoritative)

This document supersedes `doc/design.md` (which describes an earlier p-Laplacian-eigenproblem framing
that we pivoted away from). It states the problem we actually solve, the operator and its spectral
structure, the FAS two-level cycle **exactly as implemented** in `src/ElasticEmbedding.jl`, what
"deflation" means and why it is needed, and the honest empirical status.

Conventions: the embedding is `X ∈ R^{d×N}` (column `n` is point `x_n ∈ R^d`, `d = 1,2,3`), `N` points,
graph with `m` edges. `L⁺` is the sparse attractive graph Laplacian; `L⁻(X)` the dense repulsive one.

---

## 1. The problem — genuine Elastic Embedding

Given `N` data points, build a kNN graph and embed the nodes in `R^d` (d=2 or 3) by minimizing

    E(X) = Σ_{n<m} w⁺_nm ‖x_n − x_m‖²   +   λ Σ_{n<m} w⁻_nm exp(−‖x_n − x_m‖² / σ²)
           \________ attractive _______/       \_________ repulsive __________/

There are **two weight structures**, and only the first is a sparse graph:

- `w⁺` — **attractive, SPARSE**: kNN Gaussian affinities in the (high-dim) data space. Defines `L⁺ =
  D⁺ − W⁺`, a graph Laplacian. This is "the graph."
- `w⁻` — **repulsive, DENSE (all pairs)**: pushes every pair apart so the embedding cannot collapse.
  Taken uniform (`w⁻_nm = 1`) here. Evaluated as a Gaussian **in the embedding space** `R^d`.

The attractive term measures **output** distances `‖x_n−x_m‖` (Euclidean in `R^d`), weighted by
**input** affinities `w⁺` (from the data metric, baked in once). So the graph supplies only weights;
the metric in the energy is the Euclidean metric of the **low-dimensional embedding**. That is also why
the dense Gaussian sum lives in low dimension — the regime where fast summation (future work) applies.

**Why this is the sparse-plus-dense form, not a pure eigenproblem.** Restricting the repulsion to the
graph edges (an earlier simplification) turns EE into a constrained spectral problem; genuine EE keeps
the **dense** all-pairs repulsion, which is what makes it a real neighbor-embedding method (t-SNE/UMAP
family) with a genuine fold. See §3.

---

## 2. The operator (Euler–Lagrange)

Let `a = 1..d` index the coordinate; `x_n^a` is the a-th component of `x_n` (entry `(a,n)` of `X`). The
energy is a scalar; the unknowns are the `N·d` scalars `x_n^a`. Setting `∂E/∂x_n^a = 0`:

    2 Σ_m w⁺_nm (x_n^a − x_m^a)  −  (2λ/σ²) Σ_m w⁻_nm e^{−‖x_n−x_m‖²/σ²} (x_n^a − x_m^a) = 0

Write `μ = λ/σ²` and `w̃⁻_nm(X) = w⁻_nm e^{−‖x_n−x_m‖²/σ²}` (state-dependent). Collecting the a-th
coordinate row `x^a = (x_1^a,…,x_N^a)`, the equation for **every** a is the SAME `N×N` system

    ( L⁺ − μ L⁻(X) ) x^a = 0 ,      L⁻(X) = Laplacian of the weights w̃⁻(X).                     (★)

So the attractive part is **separable across coordinates** (one Laplacian applied to each of the d rows;
Hessian `= (L⁺ − μL⁻) ⊗ I_d`), and the **only** coupling between the d coordinates is that the repulsive
weights `w̃⁻` depend on the full d-dim distance. In matrix form the residual/operator is

    A(X) = 2 X L⁺ − 2μ X L⁻(X)          (d×N; `ee_A` in the code, up to the mass factors of §5).

`A(X) = 0` is a nonlinear system with the trivial solution `X = 0` plus, for `μ` past a bifurcation, the
nontrivial embedding. The metric/geometry is entirely in the low-dim output; the graph is just weights.

Code: `ee_A(Y, Lstiff, μ, mass; σ2)` returns `2·(Y·Lstiff) − 2μ·(Y·L⁻(Y))` with the repulsion weights
`mass_I·mass_J·exp(−‖·‖²/σ²)`. On the fine level `Lstiff = L⁺`, `mass = ones(N)`.

---

## 3. Spectral structure — indefinite, near-singular, low-rank (this drives everything)

The linearized (chord) operator is

    J(X) = L⁺ − μ L⁻(X).

Both `L⁺` and `L⁻` are PSD graph Laplacians killing the constant `1`. Their **difference** is
indefinite. At collapse `X=0`, `L⁻(0)` is the complete-graph Laplacian `= N·I` on `1⊥`, so on `1⊥`

    J(0) eigenvalues  =  ν_k − μN ,   k = 2..N,   ν_k = graph-Laplacian eigenvalues of L⁺.

Hence a **bifurcation cascade**: mode k goes negative at `μ*_k = ν_k / N`, starting with the Fiedler
value `ν_2` (the embedding is born along the Fiedler vector). To get a d-dim embedding, `μ` sits just
past the first few `μ*_k`, so **J has only ~d negative/near-null eigenvalues; the rest are positive.**
It is `PD + low-rank-indefinite`, NOT wave-like. The near-null modes **are the embedding directions**
(the columns of X). Measured on a swiss-roll bench: `J` had 1 negative eigenvalue and a near-null
cluster (`|λ| = 2.4e-16, 2e-3, 5e-3, 1e-2`). This low-rank indefiniteness is the crux of §7–§8.

**Continuation.** We march `μ` from small (below `μ*_2`, where `X=0` is the stable minimum) up to the
target, warm-starting — a homotopy through the fold. This is also the source of determinism: start at
the spectral (Laplacian-Eigenmaps) solution and track one branch. (Continuation not yet wired into the
solver; the two-level results below are measured at a fixed `μ = 2μ*_2`.)

---

## 4. Relaxation (smoother) — frozen-Gaussian Gauss–Seidel

The operator is **integro-differential**: `L⁺` is the differential (local, symbol `~|k|²`, principal at
high frequency) part; `μL⁻` is a smooth **integral** operator (Gaussian kernel ⇒ Gaussian-decaying
symbol ⇒ subprincipal). Brandt's rule for such operators (1984 Guide §8.6, §3.4 "principal
linearization") is: **relax the differential principal part pointwise; carry the smooth nonlocal term on
the coarse grids** — do NOT give it its own relaxation, and do NOT use distributive relaxation (that is
for systems / non-elliptic operators with no definite diagonal; a scalar Laplacian plus a smooth
integral perturbation does not qualify). This was confirmed against the corpus.

So the smoother **freezes the Gaussian force and does Gauss–Seidel on `L⁺`**: one sweep solves, per
coordinate row,

    L⁺ x^a = μ L⁻(X_frozen) x^a          (lagged: L⁻ frozen from the current X, RHS recomputed each sweep)

This uses `L⁺`'s **positive diagonal** and is stable. (Relaxing the *full* `A` diverges near `μ*`
because `A_ii = deg⁺_i − μ·deg⁻_i` can go negative — the indefiniteness lives in the subtracted smooth
term, so freezing it to keep `L⁺`'s definite diagonal is exactly right.)

Code: `ee_smooth!(X, Lp, μ, ν; σ2)`:
```julia
Lm = build_Lminus_dense(X; σ2)                  # freeze L⁻(X)
for _ in 1:ν, a in 1:d
    xa = X[a,:]; gauss_seidel!(xa, Lp; b = μ .* (Lm * xa), sweeps=1); X[a,:] = xa
end
```

---

## 5. Coarsening — mass-aware aggregation + a coarse Gaussian

- **Test vectors** are generated by relaxing the **FULL operator** (a few random vectors, each pushed
  through the frozen-Gaussian lagged-GS smoother). "Always via relaxation." These `TV` reflect J's low
  modes, not just L⁺'s — essential, because the slow modes are mass-shaped.
- **Aggregation**: LAMG's `aggregate(L⁺; X_ext = TV)` — energy-ratio-guarded affinity aggregation, fed
  the mass-aware test vectors. Gives `agg` (node→aggregate) and `nc` aggregates. Each aggregate has a
  **seed** (representative node) and a **mass** `m_I = |aggregate I|`.
- **Coarse stiffness**: Galerkin `L⁺_H = Pᵀ L⁺ P`.
- **Coarse repulsion (the coarsened Gaussian)**: the aggregate acts as a super-node of mass `m_I` at its
  coarse position; the coarse repulsion weight between aggregates is
  `m_I m_J · exp(−‖Ȳ_I − Ȳ_J‖² / σ²)`. σ is a fixed physical length, so it does **not** rescale per
  level. This is `ee_A(Y, L⁺_H, μ, mass)` with `mass = [m_I]`.

The compatible-relaxation ("mock") test confirmed this coarse variable set is excellent (μ_mock ≈ 0.13
on SBM, 0.066 on swiss roll) — the coarse SET and smoother are not the bottleneck.

---

## 6. Interpolation & restriction

- **Interpolation `P` (n×nc)**: caliber-1 piecewise-constant (each fine node = its aggregate value),
  `piecewise_constant_interpolation(agg)`. We also implemented and tested geometric/affine
  (`geometric_interpolation`) and BAMG least-squares caliber-c variants — **none robustly beat
  caliber-1 once the real bottleneck (§7) was identified.**
- **Restriction `R` (nc×n)**: seed injection, `R[I, seed_I] = 1`, with seeds forced caliber-1 in `P`, so
  **`R·P = I` exactly** (FAS-safe: a coarse function restricts to itself).
- **Residual restriction**: `Pᵀ` (i.e. multiply the fine residual by `P` on the right).

---

## 7. The FAS two-level cycle — exactly as implemented

`ee_two_level_P!(X, Lp, LpH, P, R, mass, μ; ν1=1, ν2=2, σ2)` — a **1-2 cycle** (1 pre-, 2 post-sweeps;
more post-sweeps give smoother iterates for recombination). With `X` d×N, `P` n×nc, `R` nc×n:

```
1. pre-smooth:      ee_smooth!(X, Lp, μ, ν1)                          # ν1 frozen-Gaussian GS sweeps
2. restrict soln:   Y0 = X * R'                                       # d×nc coarse embedding (seed values)
3. FAS coarse RHS:  fH = ee_A(Y0, LpH, μ, mass)  −  ee_A(X, Lp, μ, ones(N)) * P
                       └ A_H(RX) ┘    └────── residual restriction  R_res · A_h(X)  (R_res = Pᵀ) ──────┘
4. coarse solve:    Y  = ee_coarse_solve(Y0, LpH, μ, mass, fH)        # solve A_H(Y) = fH (exact-ish)
5. correct:         X += (Y − Y0) * P'                               # interpolate the coarse change
6. post-smooth:     ee_smooth!(X, Lp, μ, ν2)
```

The τ-correction `fH = A_H(RX) − R_res A_h(X)` is FAS: it makes the exact fine solution `X*` a **fixed
point** of the cycle for any `P, R` (verified: from a converged `X*`, one cycle leaves it, on irregular
graphs). The coarse solve (`ee_coarse_solve`) minimizes the τ-shifted coarse energy by Barzilai–Borwein
gradient descent (the coarse problem is small).

**Iterate recombination (DIIS), on by default** — required because caliber-1 alone leaves the low-mode
subspace under-corrected. After each cycle push `(X, A(X))` into a window (size ~3–5); form
`X_acc = Σ c_k X_k` minimizing `‖Σ c_k A(X_k)‖` subject to `Σ c_k = 1` (`ee_diis`); accept if it lowers
the residual:
```
for each cycle:
    ee_two_level_P!(X, …); R = ee_A(X, …)
    push (X, R) to window (drop oldest past `window`)
    Xa = ee_diis(window_X, window_R); if ‖A(Xa)‖ < ‖R‖: X = Xa
```
DIIS is a Krylov acceleration that **implicitly deflates the few slow modes** — which is exactly why it
partially masked the instability of §8, and why explicit deflation (§8) is the clean fix.

---

## 8. Deflation — what it is, why it's needed, how it's done

**Why.** `J = L⁺ − μL⁻` is indefinite and near-singular (§3): a few near-null / negative eigenvalues,
the embedding modes. The Galerkin coarse correction (step 4–5 above) inverts the coarse operator
`J_H = Pᵀ J P`, which inherits the near-singularity — so `J_H⁻¹` **amplifies** anything near J's null.
This is the Helmholtz-multigrid pathology. Diagnosed decisively (`experiments/linear_twogrid.jl`): a
linearized two-grid with an **exact** (dense) coarse solve gives factor **1.000** for BOTH caliber-1 and
the best geometric interpolation — so the failure is neither the coarse solve nor the interpolation; it
is the indefinite coarse-correction instability. (On irregular graphs like the SBM the single negative
mode = Fiedler = block indicator is captured by the aggregate coarse space, so the correction stays
stable and no explicit deflation is needed — factor ~0.2. On geometric graphs the near-null modes are
smooth manifold modes that the aggregate space does not represent, so the correction destabilizes.)

**What deflation is.** Let `V` hold the few near-null / negative eigenvectors of `J` (the "deflation
subspace"). Deflation = **orthogonally project the error off `span(V)` each cycle**, so the cycle only
ever acts on the well-conditioned complement `V⊥`:

    defl(e) = e − V (Vᵀ e)          (orthogonal projector I − VVᵀ, for orthonormal V)

Applied after the coarse correction and after post-smoothing. This removes the amplified modes, so the
smoother + coarse correction converge on the remainder.

**Exact code** (`tg_deflated` in `experiments/linear_twogrid.jl`, linearized two-grid):
```julia
V = eigen(Symmetric(J)).vectors[:, 1:K]         # bottom-K near-null modes of J   (see note)
defl(e) = (e .-= V * (V' * e); e)
smooth!(e) = (rhs = μ .* (Lm * e); gauss_seidel!(e, Lp; b=rhs, sweeps=1); e)   # frozen-Gaussian GS
for _ in 1:sweeps
    smooth!(e)                                  # pre-smooth (ν1=1)
    e .+= P * (pinv(JH) * (-(P' * (J * e))))    # Galerkin coarse correction (this is what is unstable)
    defl(e)                                      # ← DEFLATION: remove the amplified near-null modes
    smooth!(e); smooth!(e)                        # post-smooth (ν2=2)
    defl(e)
end
```
Measured: factor `1.000` (no deflation) → `K=2: 0.246, K=4: 0.201, K=8: 0.188` with **plain caliber-1**.
K=1 (constant only) is not enough — you must also deflate the embedding mode (the `|λ|≈2e-3` eigenvector)
and its near-null neighbors.

**Why it's nearly free in practice.** The modes to deflate — J's near-null eigenvectors — **are the
embedding coordinates `X` themselves** (and a few nearby smooth modes). So the production solver deflates
against the columns of `X` (which it is computing anyway) plus a handful of extra vectors obtained
cheaply from the smoother/Lanczos — **not** the full dense `eigen(J)` used in the demo (that is O(N³),
present only to prove the mechanism).

This vindicates the project's original spectral analysis: `J = PD + low-rank-indefinite`; deflate the
`~d` near-null modes = the embedding.

---

## 9. Empirical status (honest)

Bench: swiss-roll and blob kNN graphs, and a 4-block SBM, N≈600–1200, d=1 for factor measurements,
`μ = 2μ*_2`. "factor" = asymptotic residual reduction per cycle.

| quantity | value | source |
|---|---|---|
| FAS stationarity (X* fixed point), irregular | passes (‖ΔX‖/‖X*‖ ~1e-6) | two_level.jl |
| two-level factor, SBM (irregular), caliber-1 + recomb | ~0.2–0.5, size-independent (N=300→1200) | two_level.jl |
| two-level factor, swiss roll (geometric), caliber-1 | ~0.85–0.99 (raw ~0.99; recomb masks) | swiss_mock.jl, verify_geo.jl |
| mock-cycle ideal (smoother + perfect coarse projection) | 0.066 (swiss) | swiss_mock.jl |
| interpolation swaps (geometric/barycentric/BAMG/smoothed-agg) | do NOT robustly fix geometric (fragile / broken stationarity) | geo_*.jl, verify_geo.jl |
| **root cause: exact Galerkin two-grid** | **1.000** (both interpolations) | linear_twogrid.jl |
| **fix: deflated two-grid (caliber-1)** | **K=2→0.25, K=8→0.19** | linear_twogrid.jl |

**What we got wrong along the way (recorded for honesty):** we spent several iterations treating the
geometric-graph wall as an interpolation-order problem and building geometric/caliber-2 interpolations;
adversarial verification (the parallel workflow + `verify_geo.jl`) showed those "wins" were fragile,
non-reproducible, and recombination-masked, and the linearized exact-solve test relocated the cause to
the indefinite coarse-correction instability. The fix is deflation, not interpolation.

---

## 10. What is NOT yet done / next steps

1. **Deflated FAS two-level cycle** in the solver — deflate against X's columns (+ a few smoother/Lanczos
   vectors), not a full eigensolve. Target: robust ~0.1–0.2 on geometric graphs with clean stationarity.
   `experiments/linear_twogrid.jl` is the working proof-of-concept to port into `ee_two_level_P!`.
2. **Multilevel V-cycle** (recurse the two-level; the O(N) proof), carrying deflation on each level.
3. **FMG μ-continuation** driver (real initialization from the spectral seed + the determinism story).
4. **Fast summation** for the dense repulsion (replace the O(N²) `build_Lminus_dense`) — Brandt–Lubrecht
   / MuST / FGT; only after the cycle is right. Note this is a *commodity* (FIt-SNE precedent); the
   contribution is the deterministic deflated-multigrid optimizer + continuation, not the summation.
5. **Real single-cell / MNIST embeddings**, d=2/3, vs UMAP — the determinism/reproducibility pitch.

---

## 11. Code map

- `src/ElasticEmbedding.jl` — `ee_A`, `ee_smooth!`, `ee_coarse_solve`, `ee_two_level_P!`, `ee_diis`,
  `build_Lminus_dense`, `piecewise_constant_interpolation`(from LAMG), `geometric_interpolation`,
  graph builders (`sbm_graph`, `knn_affinity_graph`, `gaussian_blobs`), `laplacian_eigenmaps`,
  `cr_shrinkage` (mock cycle), aggregation stand-in, etc.
- `experiments/linear_twogrid.jl` — **the decisive diagnostic + deflation demo** (start here).
- `experiments/verify_geo.jl` — adversarial verification (raw vs recomb, stationarity).
- `experiments/swiss_mock.jl` — mock-vs-two-level localization.
- `experiments/slow_mode_energy.jl` — the slow mode is mass-shaped (local-energy Brandt diagnostic).
- `experiments/two_level.jl` — stationarity, size-independence, recombination, slow-mode analysis.
- `experiments/geo_*.jl` — the interpolation-design exploration (negative results, kept for the record).
- LAMG+ dependency: github.com/orenlivne/lamgplus (aggregation, caliber-2, mock cycle, recombination).
