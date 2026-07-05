"""
    ElasticEmbedding

Nonlinear spectral embedding of irregular graphs via the graph p-Laplacian eigenproblem,
solved by a direct Brandt FAS eigenproblem cycle with continuation in p. See `doc/design.md`.

This first cut implements the **Phase-1 smoother diagnostic** (the go/no-go gate) plus the shared
graph primitives, on base Julia only. The FAS eigensolver (Phase 2+) reuses LAMG+ / NLF and is
stubbed at the bottom of this file.
"""
module ElasticEmbedding

using SparseArrays, LinearAlgebra, Random, Statistics

export sbm_graph, incidence, weighted_laplacian, edge_differences,
       plaplacian_weights, frozen_plaplacian, bottom_eigenvectors, fiedler_vector,
       gauss_seidel!, sym_gauss_seidel!, smoothing_factor

# ----------------------------------------------------------------------------------------------
# Graph builders
# ----------------------------------------------------------------------------------------------

"""
    sbm_graph(sizes, p_in, p_out; rng, weight=:unit) -> (n, edges, w, block)

Stochastic block model. `sizes` gives the block sizes; an edge is added with probability `p_in`
within a block and `p_out` across blocks. Returns node count `n`, oriented `edges` (i<j), weights
`w`, and `block[i]` = block index of node i. `p_in >> p_out` gives clear communities (a near-Cheeger
cut whose Fiedler vector exposes the p→1 smoother-degeneracy we want to probe).
"""
function sbm_graph(sizes::Vector{Int}, p_in::Float64, p_out::Float64;
                   rng::AbstractRNG = Random.default_rng(), weight::Symbol = :unit)
    n = sum(sizes)
    block = Vector{Int}(undef, n)
    let c = 0
        for (b, s) in enumerate(sizes); for _ in 1:s; c += 1; block[c] = b; end; end
    end
    edges = Tuple{Int,Int}[]
    for i in 1:n-1, j in i+1:n
        p = block[i] == block[j] ? p_in : p_out
        if rand(rng) < p; push!(edges, (i, j)); end
    end
    w = weight === :unit ? ones(length(edges)) : (0.5 .+ rand(rng, length(edges)))
    return n, edges, w, block
end

"""
    incidence(n, edges, w) -> (B, w)

Signed node-edge incidence B ∈ R^{n×m}: for edge e=(i,j), B[i,e]=-1, B[j,e]=+1. Matches the
convention in NP/GraphSSL so B, w drop straight into NLF's `newton_flow!`.
"""
function incidence(n::Int, edges::Vector{Tuple{Int,Int}}, w::Vector{Float64})
    m = length(edges)
    I = Int[]; J = Int[]; V = Float64[]
    @inbounds for (e, (i, j)) in enumerate(edges)
        push!(I, i); push!(J, e); push!(V, -1.0)
        push!(I, j); push!(J, e); push!(V, +1.0)
    end
    sparse(I, J, V, n, m), copy(w)
end

# ----------------------------------------------------------------------------------------------
# p-Laplacian operators
# ----------------------------------------------------------------------------------------------

"""Edge potential differences g = Bᵀx (length m)."""
edge_differences(B::SparseMatrixCSC, x::AbstractVector) = B' * x

"""Weighted graph Laplacian L = B diag(c) Bᵀ (SPD, null space = constants)."""
weighted_laplacian(B::SparseMatrixCSC, c::AbstractVector) = B * Diagonal(c) * B'

"""
    plaplacian_weights(g, w, p; eps_rel=1e-3) -> c

Frozen linearized p-Laplacian edge conductances c_e = w_e |g_e|^{p-2}, with |g_e| floored at
`eps_rel * max|g|`. For p<2 the floor caps the conductance ratio (the ε-regularization NLF applies
via `cond_floor`); the floor magnitude controls how severe the anisotropy is — the very quantity the
smoother diagnostic must probe. The global (p-1) tangent factor is omitted (it rescales L uniformly
and does not change the smoothing factor).
"""
function plaplacian_weights(g::AbstractVector, w::AbstractVector, p::Float64; eps_rel::Float64 = 1e-3)
    gmax = maximum(abs, g)
    floor = eps_rel * (gmax > 0 ? gmax : 1.0)
    @. w * max(abs(g), floor)^(p - 2)
end

"""
    frozen_plaplacian(B, w, x, p; eps_rel=1e-3) -> L

The linearized p-Laplacian L(x,p) = B diag(w|Bᵀx|^{p-2}) Bᵀ at frozen iterate x. This is the operator
whose smoothing factor governs the FAS eigensolver's convergence.
"""
function frozen_plaplacian(B::SparseMatrixCSC, w::AbstractVector, x::AbstractVector, p::Float64;
                           eps_rel::Float64 = 1e-3)
    c = plaplacian_weights(edge_differences(B, x), w, p; eps_rel = eps_rel)
    weighted_laplacian(B, c)
end

# ----------------------------------------------------------------------------------------------
# Eigenvectors (dense; small graphs only — for reference/diagnostic)
# ----------------------------------------------------------------------------------------------

"""Bottom-K eigenvectors of L (ascending eigenvalue), dense — for small n only."""
function bottom_eigenvectors(L::AbstractMatrix, K::Int)
    F = eigen(Symmetric(Matrix(L)))
    F.vectors[:, 1:K]
end

"""Fiedler vector = 2nd-smallest eigenvector of L (the nontrivial mode)."""
fiedler_vector(L::AbstractMatrix) = bottom_eigenvectors(L, 2)[:, 2]

# ----------------------------------------------------------------------------------------------
# Gauss–Seidel smoother + smoothing-factor diagnostic
# ----------------------------------------------------------------------------------------------

"""
    gauss_seidel!(v, L, b; sweeps=1, reverse=false)

In-place Gauss–Seidel on Lv=b. L is symmetric, so its CSC column i equals row i. `reverse=true`
sweeps nodes n→1 (a true backward sweep). For the homogeneous diagnostic pass b=0 (default); then v
itself is the error and each sweep applies the smoother's error-propagation operator.
"""
function gauss_seidel!(v::AbstractVector, L::SparseMatrixCSC; b::AbstractVector = Float64[],
                       sweeps::Int = 1, reverse::Bool = false)
    n = size(L, 1)
    rows = rowvals(L); vals = nonzeros(L)
    use_b = !isempty(b)
    order = reverse ? (n:-1:1) : (1:n)
    @inbounds for _ in 1:sweeps
        for i in order
            s = 0.0; dii = 0.0
            for idx in nzrange(L, i)
                j = rows[idx]; lij = vals[idx]
                if j == i; dii = lij; else; s += lij * v[j]; end
            end
            v[i] = ((use_b ? b[i] : 0.0) - s) / dii
        end
    end
    v
end

"""Symmetric Gauss–Seidel: one forward sweep followed by one backward sweep (the standard,
symmetric multigrid smoother). Applies `sweeps` such forward+backward pairs."""
function sym_gauss_seidel!(v::AbstractVector, L::SparseMatrixCSC; b::AbstractVector = Float64[], sweeps::Int = 1)
    for _ in 1:sweeps
        gauss_seidel!(v, L; b = b, sweeps = 1, reverse = false)
        gauss_seidel!(v, L; b = b, sweeps = 1, reverse = true)
    end
    v
end

"""
    smoothing_factor(L; K=4, sweeps=25, warmup=5, ntrials=3, rng) -> (mu, rho)

Estimate the Gauss–Seidel smoothing factor for L.

- `rho`  : plain GS asymptotic residual-reduction factor per sweep (conflates near-null low modes,
           which are the coarse grid's job — reported for context).
- `mu`   : the true smoothing factor — reduction per sweep of the error component ORTHOGONAL to the
           bottom-K eigenspace (deflated each sweep). This isolates high-frequency error, which is
           what the smoother must kill. `mu < ~0.3` uniformly in p ⇒ direct FAS is viable.
"""
function smoothing_factor(L::SparseMatrixCSC; K::Int = 4, sweeps::Int = 25, warmup::Int = 5,
                          ntrials::Int = 3, smoother::Symbol = :gs,
                          rng::AbstractRNG = Random.default_rng())
    n = size(L, 1)
    V = bottom_eigenvectors(L, K)                       # deflation basis (orthonormal)
    deflate!(x) = (x .-= V * (V' * x); x)
    smooth!(z) = smoother === :sgs ? sym_gauss_seidel!(z, L; sweeps = 1) : gauss_seidel!(z, L; sweeps = 1)
    mus = Float64[]; rhos = Float64[]
    for _ in 1:ntrials
        # --- mu: deflated (smoothing) factor ---
        e = randn(rng, n); deflate!(e); e ./= norm(e)
        ratios = Float64[]
        for s in 1:sweeps
            n0 = norm(e)
            smooth!(e); deflate!(e)
            push!(ratios, norm(e) / n0)
        end
        push!(mus, exp(mean(log.(ratios[warmup+1:end]))))
        # --- rho: plain GS residual-reduction factor ---
        v = randn(rng, n); v .-= mean(v)
        rr = Float64[]
        for s in 1:sweeps
            r0 = norm(L * v)
            gauss_seidel!(v, L; sweeps = 1); v .-= mean(v)
            r1 = norm(L * v)
            push!(rr, r0 > 0 ? r1 / r0 : 1.0)
        end
        push!(rhos, exp(mean(log.(rr[warmup+1:end]))))
    end
    return mean(mus), mean(rhos)
end

# ----------------------------------------------------------------------------------------------
# Genuine Elastic Embedding (DENSE all-pairs repulsion) — direct O(N²), for the determinism gate.
# The production path replaces the O(N²) repulsion with O(N) fast summation (Brandt–Lubrecht / FGT)
# and gradient descent with a multigrid optimizer; the MATH (energy/gradient) is identical.
#
#   E(X;λ) = Σ_{n<m} w+_nm ||x_n−x_m||²  +  λ Σ_{n<m} exp(−||x_n−x_m||²)
#   ∇E     = 2 X L+  −  2λ X L−(X)      (L+ sparse attractive; L−(X) = Laplacian of exp-weights, DENSE)
# ----------------------------------------------------------------------------------------------

export knn_affinity_graph, gaussian_blobs, laplacian_eigenmaps,
       ee_energy, ee_gradient, ee_minimize, ee_continuation, ee_hessian, ee_newton, procrustes_rmsd, knn_purity

"""Gaussian blobs: `k` clusters of `nper` points in `dim`-dim space. Returns (data d×N, labels)."""
function gaussian_blobs(nper::Int, k::Int, dim::Int; sep::Float64 = 6.0, rng::AbstractRNG = Random.default_rng())
    N = nper * k
    X = zeros(dim, N); labels = zeros(Int, N)
    for c in 1:k
        center = sep .* randn(rng, dim)
        for i in 1:nper
            idx = (c - 1) * nper + i
            X[:, idx] .= center .+ randn(rng, dim); labels[idx] = c
        end
    end
    X, labels
end

"""Symmetric kNN graph with Gaussian affinities from `data` (dim×N). Dense distances (gate-scale N)."""
function knn_affinity_graph(data::AbstractMatrix, k::Int)
    N = size(data, 2)
    D2 = [sum(abs2, @view(data[:, i]) .- @view(data[:, j])) for i in 1:N, j in 1:N]
    nbrs = [partialsortperm(D2[i, :], 2:k+1) for i in 1:N]          # exclude self
    σ2 = median([D2[i, nbrs[i][end]] for i in 1:N])
    edges = Tuple{Int,Int}[]; wv = Float64[]; seen = Set{Tuple{Int,Int}}()
    for i in 1:N, j in nbrs[i]
        a, b = min(i, j), max(i, j)
        if !((a, b) in seen); push!(seen, (a, b)); push!(edges, (a, b)); push!(wv, exp(-D2[a, b] / σ2)); end
    end
    B, w = incidence(N, edges, wv)
    B, w, edges
end

"""Laplacian-Eigenmaps seed: d-dim embedding from L+'s bottom nontrivial eigenvectors (deterministic)."""
function laplacian_eigenmaps(Lp::AbstractMatrix, d::Int)
    V = bottom_eigenvectors(Lp, d + 1)               # col 1 ≈ constant
    X = Matrix(permutedims(V[:, 2:d+1]))             # d × N
    for i in 1:d                                     # deterministic sign fix
        j = argmax(abs.(@view X[i, :]))
        if X[i, j] < 0; @views X[i, :] .= .-X[i, :]; end
    end
    X
end

"""EE energy for embedding X (d×N), attractive Laplacian Lp, repulsion strength λ (uniform w−=1)."""
function ee_energy(X::AbstractMatrix, Lp::AbstractMatrix, λ::Float64)
    Eattr = tr(X * Lp * X')
    N = size(X, 2); dd = size(X, 1); Erep = 0.0
    @inbounds for n in 1:N-1, m in n+1:N
        d2 = 0.0; for k in 1:dd; δ = X[k, n] - X[k, m]; d2 += δ * δ; end
        Erep += exp(-d2)
    end
    Eattr + λ * Erep
end

"""EE gradient (d×N): 2 X L+ (attractive, sparse) − 2λ Σ exp-weighted differences (repulsive, dense)."""
function ee_gradient(X::AbstractMatrix, Lp::AbstractMatrix, λ::Float64)
    G = 2.0 .* (X * Lp)
    N = size(X, 2); dd = size(X, 1)
    @inbounds for n in 1:N-1, m in n+1:N
        d2 = 0.0; for k in 1:dd; δ = X[k, n] - X[k, m]; d2 += δ * δ; end
        wt = exp(-d2)
        for k in 1:dd
            f = -2.0 * λ * wt * (X[k, n] - X[k, m])
            G[k, n] += f; G[k, m] -= f
        end
    end
    G
end

"""Minimize EE at fixed λ by backtracking gradient descent from X0. Returns (X, energy)."""
function ee_minimize(X0::AbstractMatrix, Lp::AbstractMatrix, λ::Float64; iters::Int = 300, gtol::Float64 = 1e-5)
    X = copy(X0); E = ee_energy(X, Lp, λ); step = 1.0
    for _ in 1:iters
        G = ee_gradient(X, Lp, λ); gn = norm(G); gn < gtol && break
        s = step; Xn = X .- s .* G; En = ee_energy(Xn, Lp, λ); bt = 0
        while En > E - 1e-4 * s * gn^2 && bt < 40
            s *= 0.5; Xn = X .- s .* G; En = ee_energy(Xn, Lp, λ); bt += 1
        end
        X = Xn; E = En; step = min(1.0, 2s)
    end
    X, E
end

"""λ-continuation: warm-started sweep of ee_minimize over an increasing schedule `λs` from `Xseed`."""
function ee_continuation(Xseed::AbstractMatrix, Lp::AbstractMatrix, λs::AbstractVector; iters::Int = 200)
    X = copy(Xseed)
    for λ in λs; X, _ = ee_minimize(X, Lp, λ; iters = iters); end
    X
end

"""
    ee_hessian(X, Lp, λ; σ2) -> H  (dN×dN dense)

Full EE Hessian at embedding X (d×N): H = 2 L⁺⊗I_d − L_W, where L_W is the block Laplacian of the d×d
edge weights W_ij = 2λ·exp(−‖x_i−x_j‖²/σ²)·(I_d − (2/σ²) r_ij r_ijᵀ), r_ij = x_i−x_j. The `I` part is the
chord term (2(L⁺−λL⁻)⊗I); the `−(2/σ²)rrᵀ` part is the mass-derivative (what the reduced-V corrector needs
and what makes H indefinite away from the minimum). Index (a,i) → (i−1)d+a. Dense O(N²) build.
"""
function ee_hessian(X::AbstractMatrix, Lp::AbstractMatrix, λ::Float64; σ2::Float64 = 1.0)
    d, N = size(X); H = zeros(d * N, d * N); Lpm = Matrix(Lp)
    idx(a, i) = (i - 1) * d + a
    for i in 1:N, a in 1:d, j in 1:N
        Lpm[i, j] != 0 && (H[idx(a, i), idx(a, j)] += 2 * Lpm[i, j])
    end
    Id = Matrix(I, d, d)
    for i in 1:N-1, j in i+1:N
        r = X[:, i] .- X[:, j]; d2 = dot(r, r); s = exp(-d2 / σ2)
        W = 2λ * s .* (Id .- (2 / σ2) .* (r * r'))
        bi = (i - 1) * d; bj = (j - 1) * d
        @views H[bi+1:bi+d, bj+1:bj+d] .+= W; @views H[bj+1:bj+d, bi+1:bi+d] .+= W
        @views H[bi+1:bi+d, bi+1:bi+d] .-= W; @views H[bj+1:bj+d, bj+1:bj+d] .-= W
    end
    H
end

"""
    ee_newton(X0, Lp, λ; iters, gtol, reg) -> (X, gradnorm)

Damped full-Hessian Newton ground-truth solver for EE. Converges to a genuine stationary point (gradnorm
→ 1e-12) at ANY μ, including near a bifurcation where `ee_minimize` (gradient descent) suffers critical
slowing. Energy line search + tiny diagonal regularization (H is indefinite away from the min and has the
translation null mode). Dense O((dN)³) — a reference solver, not the scalable one.
"""
function ee_newton(X0::AbstractMatrix, Lp::AbstractMatrix, λ::Float64; iters::Int = 60, gtol::Float64 = 1e-11, reg::Float64 = 1e-8, σ2::Float64 = 1.0)
    d, N = size(X0); X = copy(X0); E = ee_energy(X, Lp, λ)
    for _ in 1:iters
        G = ee_gradient(X, Lp, λ); gn = norm(G); gn < gtol && break
        H = ee_hessian(X, Lp, λ; σ2 = σ2); H[diagind(H)] .+= reg
        Δ = reshape(-(H \ vec(G)), d, N)
        s = 1.0; bt = 0; Xn = X .+ s .* Δ; En = ee_energy(Xn, Lp, λ)
        while (!isfinite(En) || En > E - 1e-4 * s * dot(G, -Δ)) && bt < 40
            s *= 0.5; Xn = X .+ s .* Δ; En = ee_energy(Xn, Lp, λ); bt += 1
        end
        bt ≥ 40 && (Xn = X .- 0.1 .* G ./ gn; En = ee_energy(Xn, Lp, λ))
        X = Xn; E = En
    end
    X, norm(ee_gradient(X, Lp, λ))
end

"""Procrustes-aligned normalized RMSD between two embeddings (d×N): rotation+reflection+scale invariant."""
function procrustes_rmsd(X::AbstractMatrix, Y::AbstractMatrix)
    Xc = X .- Statistics.mean(X, dims = 2); Yc = Y .- Statistics.mean(Y, dims = 2)
    F = svd(Yc * Xc'); R = F.V * F.U'; Ya = R * Yc
    s = sum(Xc .* Ya) / (sum(Ya .* Ya) + 1e-300); Ya .*= s
    sqrt(Statistics.mean((Xc .- Ya) .^ 2)) / (sqrt(Statistics.mean(Xc .^ 2)) + 1e-12)
end

"""kNN label purity of an embedding X (d×N): fraction of each point's k embedding-neighbors sharing its label."""
function knn_purity(X::AbstractMatrix, labels::AbstractVector, k::Int)
    N = size(X, 2)
    D2 = [sum(abs2, @view(X[:, i]) .- @view(X[:, j])) for i in 1:N, j in 1:N]
    tot = 0.0
    for i in 1:N
        nb = partialsortperm(D2[i, :], 2:k+1)
        tot += count(j -> labels[j] == labels[i], nb) / k
    end
    tot / N
end

# ----------------------------------------------------------------------------------------------
# Aggregation + compatible relaxation (coarsening-quality test for the EE FAS cycle)
# ----------------------------------------------------------------------------------------------

export affinity_aggregation, aggregate_project!, kaczmarz!, gauss_seidel_dense!,
       cr_shrinkage, build_Lminus_dense, ee_operator

"""Dense repulsive Laplacian L⁻(X) with weights w̃⁻_nm = exp(−‖x_n−x_m‖²/σ²) (X is d×N)."""
function build_Lminus_dense(X::AbstractMatrix; σ2::Float64 = 1.0)
    N = size(X, 2); dd = size(X, 1)
    W = zeros(N, N)
    @inbounds for i in 1:N-1, j in i+1:N
        d2 = 0.0; for k in 1:dd; δ = X[k, i] - X[k, j]; d2 += δ * δ; end
        w = exp(-d2 / σ2); W[i, j] = w; W[j, i] = w
    end
    Diagonal(vec(sum(W, dims = 2))) - W
end

"""The full EE operator A = L⁺ − μ L⁻(X) (dense) at embedding X, coefficient μ = λ/σ²."""
ee_operator(Lp::AbstractMatrix, X::AbstractMatrix, μ::Float64; σ2::Float64 = 1.0) =
    Matrix(Lp) .- μ .* build_Lminus_dense(X; σ2 = σ2)

"""
    affinity_aggregation(Lp; ntv=4, maxsize=4, θ=0.3, rng) -> (agg, nc)

LAMG-style algebraic aggregation of the graph Laplacian Lp. Test vectors are a few GS-relaxed random
vectors; edge affinity c_uv = (Σ_k x_k(u)x_k(v))² / (Σ_k x_k(u)² Σ_k x_k(v)²); greedy grouping of a
seed with its highest-affinity unaggregated neighbors (affinity ≥ θ, up to `maxsize`). Stand-in for
LAMG+ aggregation. Returns per-node aggregate id `agg` and coarse count `nc`.
"""
function affinity_aggregation(Lp::SparseMatrixCSC; ntv::Int = 4, maxsize::Int = 4, θ::Float64 = 0.3,
                              rng::AbstractRNG = Random.default_rng())
    n = size(Lp, 1)
    TV = zeros(n, ntv)
    for k in 1:ntv
        v = randn(rng, n); v .-= mean(v)
        gauss_seidel!(v, Lp; sweeps = 3); v .-= mean(v)
        TV[:, k] = v ./ (norm(v) + 1e-300)
    end
    tvn = [dot(@view(TV[i, :]), @view(TV[i, :])) for i in 1:n]
    rows = rowvals(Lp); diagv = [Lp[i, i] for i in 1:n]
    agg = zeros(Int, n); nc = 0
    for i in sortperm(diagv; rev = true)               # high-degree seeds first
        agg[i] != 0 && continue
        nc += 1; agg[i] = nc; sz = 1
        cand = Tuple{Float64,Int}[]
        for idx in nzrange(Lp, i)
            j = rows[idx]; (j == i || agg[j] != 0) && continue
            aff = dot(@view(TV[i, :]), @view(TV[j, :]))^2 / (tvn[i] * tvn[j] + 1e-30)
            push!(cand, (aff, j))
        end
        sort!(cand; rev = true)
        for (aff, j) in cand
            (sz >= maxsize || aff < θ) && break
            agg[j] == 0 || continue
            agg[j] = nc; sz += 1
        end
    end
    agg, nc
end

"""Zero the coarse variables: subtract each aggregate's average from its members (⇒ aggregate mean 0).
This is the compatible-relaxation projection (coarse vars = aggregate averages)."""
function aggregate_project!(e::AbstractVector, agg::Vector{Int}, nc::Int)
    sums = zeros(nc); cnts = zeros(Int, nc)
    @inbounds for i in eachindex(e); sums[agg[i]] += e[i]; cnts[agg[i]] += 1; end
    @inbounds for i in eachindex(e); e[i] -= sums[agg[i]] / cnts[agg[i]]; end
    e
end

"""Kaczmarz sweep on A e = 0: project e onto each row's constraint hyperplane (diagonal-sign-agnostic)."""
function kaczmarz!(e::AbstractVector, A::AbstractMatrix; sweeps::Int = 1)
    n = size(A, 1)
    @inbounds for _ in 1:sweeps, i in 1:n
        ai = @view A[i, :]
        nrm2 = dot(ai, ai); nrm2 == 0 && continue
        e .-= (dot(ai, e) / nrm2) .* ai
    end
    e
end

"""Dense forward Gauss–Seidel on A e = 0 (used to expose the negative-diagonal breakdown near μ*)."""
function gauss_seidel_dense!(e::AbstractVector, A::AbstractMatrix; sweeps::Int = 1)
    n = size(A, 1)
    @inbounds for _ in 1:sweeps, i in 1:n
        s = 0.0; for j in 1:n; j == i && continue; s += A[i, j] * e[j]; end
        e[i] = -s / A[i, i]
    end
    e
end

"""
    cr_shrinkage(N, relax!, agg, nc; ν=1, sweeps=50, warmup=15, rng) -> factor

Compatible-relaxation shrinkage factor: from a random compatible error, repeatedly do ν relaxation
sweeps then zero the coarse variables, and return the asymptotic per-cycle reduction. Small ⇒ the
coarse variable set (aggregation) is good — fixing it leaves fast-relaxing error.
"""
function cr_shrinkage(N::Int, relax!, agg::Vector{Int}, nc::Int; ν::Int = 1, sweeps::Int = 50,
                      warmup::Int = 15, rng::AbstractRNG = Random.default_rng())
    e = randn(rng, N); aggregate_project!(e, agg, nc); e ./= norm(e)
    ratios = Float64[]
    for _ in 1:sweeps
        n0 = norm(e)
        for _ in 1:ν; relax!(e); end
        aggregate_project!(e, agg, nc)
        push!(ratios, norm(e) / max(n0, 1e-300))
    end
    exp(mean(log.(ratios[warmup+1:end] .+ 1e-300)))
end

# ----------------------------------------------------------------------------------------------
# Two-level FAS cycle for genuine Elastic Embedding  (Phase-2 core; direct O(N²) Gaussian for now)
# ----------------------------------------------------------------------------------------------
export ee_A, ee_centroids, ee_restrict_sum, ee_interp, ee_smooth!, ee_coarse_solve,
       ee_two_level!, ee_two_level_P!, ee_two_level_deflated!, ee_two_level_galerkin!,
       ee_chord_newton_step!, ee_defl_basis, ee_reduced_hessian, ee_reduced_newton_step!, ee_corrector!,
       deflate_correct!, ee_diis, ee_solve, geometric_interpolation

"""EE operator A(Y) = 2 Y Lstiff − 2μ Y L⁻(Y), repulsion weight mass_I·mass_J·exp(−‖·‖²/σ²)."""
function ee_A(Y, Lstiff, μ, mass; σ2 = 1.0)
    G = 2.0 .* (Y * Lstiff); nc = size(Y, 2); dd = size(Y, 1)
    @inbounds for I in 1:nc-1, J in I+1:nc
        d2 = 0.0; for k in 1:dd; δ = Y[k, I] - Y[k, J]; d2 += δ * δ; end
        wt = mass[I] * mass[J] * exp(-d2 / σ2)
        for k in 1:dd; f = -2μ * wt * (Y[k, I] - Y[k, J]); G[k, I] += f; G[k, J] -= f; end
    end
    G
end

ee_centroids(X, agg, nc) = (Y = zeros(size(X, 1), nc); c = zeros(Int, nc);
    (for i in axes(X, 2); @views Y[:, agg[i]] .+= X[:, i]; c[agg[i]] += 1; end);
    (for I in 1:nc; @views Y[:, I] ./= c[I]; end); Y)
ee_restrict_sum(G, agg, nc) = (F = zeros(size(G, 1), nc);
    (for i in axes(G, 2); @views F[:, agg[i]] .+= G[:, i]; end); F)
ee_interp(Y, agg) = (X = zeros(size(Y, 1), length(agg));
    (for i in eachindex(agg); @views X[:, i] = Y[:, agg[i]]; end); X)

"""Frozen-Gaussian lagged-GS smoother: ν sweeps of L⁺ x^a = μ L⁻(X_frozen) x^a per coordinate."""
function ee_smooth!(X, Lp, μ, ν; σ2 = 1.0)
    Lm = build_Lminus_dense(X; σ2 = σ2)
    for _ in 1:ν, a in 1:size(X, 1)
        xa = collect(@view X[a, :]); gauss_seidel!(xa, Lp; b = μ .* (Lm * xa), sweeps = 1); X[a, :] = xa
    end
    X
end

"""Exact coarse solve of A_H(Y)=fH by Barzilai–Borwein descent (NaN-guarded)."""
function ee_coarse_solve(Y0, LpH, μ, mass, fH; σ2 = 1.0, iters = 3000, tol = 1e-10)
    Y = copy(Y0); g = ee_A(Y, LpH, μ, mass; σ2 = σ2) .- fH; Yp = Y; gp = g; s = 1e-3
    for it in 1:iters
        ng = norm(g); (ng < tol || !isfinite(ng)) && break
        if it > 1
            dY = vec(Y .- Yp); dg = vec(g .- gp)
            s = clamp(abs(dot(dY, dg) / (dot(dg, dg) + 1e-300)), 1e-7, 1.0)
        end
        Yp = Y; gp = g; Y = Y .- s .* g; g = ee_A(Y, LpH, μ, mass; σ2 = σ2) .- fH
    end
    any(!isfinite, Y) ? Y0 : Y
end

"""One two-level FAS cycle. Default 1-2 (ν1=1 pre, ν2=2 post) — smoother post-iterates for recombination."""
function ee_two_level!(X, Lp, LpH, agg, nc, mass, μ; ν1 = 1, ν2 = 2, σ2 = 1.0)
    ee_smooth!(X, Lp, μ, ν1; σ2 = σ2)
    Y0 = ee_centroids(X, agg, nc)
    fH = ee_A(Y0, LpH, μ, mass; σ2 = σ2) .- ee_restrict_sum(ee_A(X, Lp, μ, ones(size(X, 2)); σ2 = σ2), agg, nc)
    X .+= ee_interp(ee_coarse_solve(Y0, LpH, μ, mass, fH; σ2 = σ2) .- Y0, agg)
    ee_smooth!(X, Lp, μ, ν2; σ2 = σ2)
    X
end

"""One two-level FAS cycle with GENERAL interpolation P (n×nc) and restriction R (nc×n), R·P=I.
Supports caliber-1 (piecewise-constant) and caliber-2 (LAMG selective) transfers uniformly.
Coarse solution Y0 = X Rᵀ (d×nc); FAS τ-corrected RHS uses residual restriction by Pᵀ (A_h(X)·P);
correction interpolated by (Y−Y0)·Pᵀ. Coarse stiffness LpH, coarse masses `mass`, seed/centroid Y."""
function ee_two_level_P!(X, Lp, LpH, P, R, mass, μ; ν1 = 1, ν2 = 2, σ2 = 1.0)
    ee_smooth!(X, Lp, μ, ν1; σ2 = σ2)
    Y0 = Matrix(X * R')
    fH = ee_A(Y0, LpH, μ, mass; σ2 = σ2) .- Matrix(ee_A(X, Lp, μ, ones(size(X, 2)); σ2 = σ2) * P)
    Y = ee_coarse_solve(Y0, LpH, μ, mass, fH; σ2 = σ2)
    X .+= Matrix((Y .- Y0) * P')
    ee_smooth!(X, Lp, μ, ν2; σ2 = σ2)
    X
end

"""
    geometric_interpolation(agg, X; c=0, ridge=1e-6) -> P, R

GEOMETRIC (embedding-aware) interpolation, for geometric/manifold graphs where the low modes vary
SMOOTHLY across aggregates and piecewise-constant caliber-1 staircases them. Coarse points are the
aggregate SEEDS at their embedding positions Y_I = X[:,seed_I]. Each non-seed fine node i interpolates
from its `c` nearest seeds (default 2(d+1)) with AFFINE least-squares weights — min‖Σ p_J Y_J − x_i‖²
s.t. Σ p_J = 1 — so LINEAR functions of the embedding are reproduced exactly (order-1 interpolation).
Seeds stay caliber-1 (P[seed_I,I]=1) and R = seed injection ⇒ R·P=I exactly (FAS-safe). Extrapolation
guard (|p|>3) falls back to caliber-1. Uses the current embedding X as the frozen geometry.
"""
function geometric_interpolation(agg::AbstractVector{Int}, X::AbstractMatrix; c::Int = 0, ridge::Float64 = 1e-6)
    d, n = size(X); nc = maximum(agg)
    seed = zeros(Int, nc); @inbounds for i in 1:n; a = agg[i]; seed[a] == 0 && (seed[a] = i); end
    isseed = falses(n); for a in 1:nc; isseed[seed[a]] = true; end
    Y = X[:, seed]                                   # d × nc seed positions
    cc = c > 0 ? c : 2 * (d + 1)
    Ip = Int[]; Jp = Int[]; Vp = Float64[]; dist = zeros(nc)
    @inbounds for i in 1:n
        if isseed[i]; push!(Ip, i); push!(Jp, agg[i]); push!(Vp, 1.0); continue; end
        xi = @view X[:, i]
        for J in 1:nc; s = 0.0; for k in 1:d; δ = xi[k] - Y[k, J]; s += δ * δ; end; dist[J] = s; end
        nn = partialsortperm(dist, 1:min(cc, nc)); m = length(nn); Yn = Y[:, nn]
        G = Yn' * Yn; G .+= (ridge * (tr(G) / m + 1e-300)) .* Matrix(I, m, m)
        p = try ([2G ones(m); ones(m)' 0.0] \ [2 .* (Yn' * xi); 1.0])[1:m] catch; fill(1.0 / m, m) end
        if any(!isfinite, p) || maximum(abs, p) > 3.0
            push!(Ip, i); push!(Jp, agg[i]); push!(Vp, 1.0)
        else
            for (idx, J) in enumerate(nn); abs(p[idx]) > 1e-8 && (push!(Ip, i); push!(Jp, J); push!(Vp, p[idx])); end
        end
    end
    sparse(Ip, Jp, Vp, n, nc), sparse(collect(1:nc), seed, ones(nc), nc, n)
end

"""
    deflate_correct!(X, Lp, μ, Q; σ2) -> X

Additive deflation of the near-null subspace of J = L⁺ − μL⁻(X). Q (N×k, orthonormal) spans the
near-null / negative modes (in practice: the constant, X's own coordinate rows — which are J's null
vectors since A(X*)=2X*J=0 — and a few relaxed test vectors). Re-solves the Q-space EXACTLY per
coordinate, c = −(QᵀJQ)⁺ Qᵀ J x^a, x^a += Q c, so the Q-projected residual is zeroed. This overwrites
any amplification the indefinite Galerkin coarse correction introduced on those modes, stabilizing the
cycle WITHOUT removing the embedding (whose correct Q-value is what this restores). pinv handles the
exact-null (constant) safely. O(N² k) from the dense J·Q (fast summation would make it O(N k)).
"""
function deflate_correct!(X, Lp, μ, Q; σ2 = 1.0)
    Lm = build_Lminus_dense(X; σ2 = σ2)
    QJQi = pinv(Matrix(Q' * (Lp * Q .- μ .* (Lm * Q))))          # (QᵀJQ)⁺  (k×k)
    JXt = Lp * X' .- μ .* (Lm * X')                              # N×d : J x^a per column
    X .-= permutedims(Q * (QJQi * (Q' * JXt)))                   # additive Q-space correction
    X
end

"""
    ee_two_level_galerkin!(X, Lp, P, μ, Q; ν1, ν2, σ2) -> X

Fixed FAS cycle for the geometric/indefinite case, matching the linearized demo that converges at ~0.2.
Two departures from `ee_two_level_deflated!`: (1) the coarse correction is a LINEAR Galerkin solve of the
frozen Jacobian J = L⁺ − μL⁻(X) — coarse operator J_H = PᵀJP, correction δ_H = −J_H⁺ Pᵀ r — instead of a
rediscretized mass-Gaussian nonlinear coarse solve; (2) PROJECTION deflation (δ ← δ − (δQ)Qᵀ), which is
stable for near-null Q, instead of the degenerate additive form. Smoother remains the nonlinear
frozen-Gaussian GS. r = A(X)/2 is the current nonlinear residual (so the correction is a Newton-like
step with a linearized coarse operator). O(N²·nc) from the dense J·P (fast summation → O(N·nc)).
"""
function ee_two_level_galerkin!(X, Lp, P, μ, Q; ν1 = 1, ν2 = 2, σ2 = 1.0)
    ee_smooth!(X, Lp, μ, ν1; σ2 = σ2)
    Lm = build_Lminus_dense(X; σ2 = σ2)
    JH = Matrix(P' * (Lp * P .- μ .* (Lm * P)))                  # nc×nc Galerkin coarse Jacobian
    r = ee_A(X, Lp, μ, ones(size(X, 2)); σ2 = σ2) ./ 2           # d×N nonlinear residual /2 (= J x^a per row)
    δH = -(pinv(JH) * Matrix(r * P)')                           # nc×d coarse correction: J_H δ_H = −Pᵀ r
    δ = permutedims(P * δH)                                     # d×N interpolate
    δ .-= (δ * Q) * Q'                                          # PROJECTION deflation of the correction
    X .+= δ
    ee_smooth!(X, Lp, μ, ν2; σ2 = σ2)
    X
end

"""
    ee_chord_newton_step!(X, Lp, P, Q, μ; n_inner, ν1, ν2, σ2) -> X

THE nonlinear cycle for geometric/indefinite EE — validated ~0.2/cycle (n_inner=1) or ~0.04 (n_inner=2),
size- and d-independent (d=1,2,3), on the swiss roll. Chord-Newton outer + a DEFLATED two-grid inner
solve of the correction equation J δ = −r, where J = L⁺ − μ Lm(X) is the chord Jacobian LAGGED at the
current X (needs no knowledge of X*), r = J x^a the current residual per coordinate. Inner two-grid =
the linearized deflation demo: smoother L⁺δ = μ Lm δ − r (lagged GS), Galerkin coarse J_H = PᵀJP, and
projection deflation of the near-null modes (δ ← δ − Q Qᵀδ). The ~d embedding (near-null) modes are
deflated here and advanced by μ-continuation (the "pin" in the two-level factor test = continuation's
job); the inner solver reduces only the V⊥ correction. Q = [1, X coords, few mass-aware TVs] (all free —
no eigensolve). NOTE this SUPERSEDES ee_two_level_galerkin!, whose full-X nonlinear smoother and
correction-only deflation stalled; the correct move is to smooth/deflate the CORRECTION eqn, not X.
"""
function ee_chord_newton_step!(X, Lp, P, Q, μ; n_inner = 1, ν1 = 1, ν2 = 2, σ2 = 1.0)
    Lm = build_Lminus_dense(X; σ2 = σ2)                        # chord Jacobian J = L⁺ − μLm, lagged at current X
    JHi = pinv(Matrix(P' * (Lp * P .- μ .* (Lm * P))))         # Galerkin coarse J_H = PᵀJP
    Jmul(v) = Lp * v .- μ .* (Lm * v)
    defl(v) = (v .-= Q * (Q' * v); v)
    for a in 1:size(X, 1)
        r = Jmul(X[a, :]); δ = zeros(size(X, 2))               # solve J δ = −r for the correction
        for _ in 1:n_inner
            for _ in 1:ν1; gauss_seidel!(δ, Lp; b = μ .* (Lm * δ) .- r, sweeps = 1); end
            δ .+= P * (JHi * (P' * (-r .- Jmul(δ)))); defl(δ)  # Galerkin coarse correction + error deflation
            for _ in 1:ν2; gauss_seidel!(δ, Lp; b = μ .* (Lm * δ) .- r, sweeps = 1); end
            defl(δ)
        end
        X[a, :] .+= δ
    end
    X
end

"""
    ee_defl_basis(Φ, X) -> Q  (N×k, orthonormal)

Deflation / near-null basis for the developed-regime inner solver: Q = orthonormalize([1, Φ, X-rows]),
where Φ (N×K) are L⁺'s bottom nontrivial eigenvectors (φ₂..φ_{K+1}). Since μL⁻ is a small perturbation,
J's near-null cluster ≈ L⁺'s bottom modes; Stage 0 showed [1,X,TV] under-covers it in the developed regime
(V⊥ factor → >1) while this basis holds ~0.15. Φ is sparse-Lanczos-cheap and the continuation needs it
anyway (μ*_k = ν_{k+1}/N, φ_k pitchfork seeds).
"""
ee_defl_basis(Φ::AbstractMatrix, X::AbstractMatrix) =
    Matrix(qr(hcat(ones(size(X, 2)), Φ, Matrix(X'))).Q)

"""
    ee_reduced_hessian(X, Lp, Q, μ; σ2) -> rH  (dk×dk, k=size(Q,2))

Hessian of the EE energy restricted to the deflation subspace, in reduced coordinates C = X·Q (d×k), with
the mass-derivative term the deflated inner solver drops. rH[(a,γ),(b,β)] (index (a,γ)→(γ−1)d+a):
  attraction  2·δ_ab·(QᵀL⁺Q)[γ,β]
  repulsion   −Σ_{i<j} W_ij[a,b]·ΔQ_γ·ΔQ_β,  W_ij = 2μ·exp(−‖rᵢⱼ‖²/σ²)(I−(2/σ²)rᵢⱼrᵢⱼᵀ),  ΔQ_γ=Q[i,γ]−Q[j,γ]
Assembled directly (no dN×dN Hessian): O(N²·(dk)²) now, O(N·(dk)²) with fast summation. This is the design's
Jred: the piece that lets the branch leave X=0 at finite amplitude (C=0 is a repellor above threshold).
"""
function ee_reduced_hessian(X::AbstractMatrix, Lp::AbstractMatrix, Q::AbstractMatrix, μ::Float64; σ2::Float64 = 1.0)
    d, N = size(X); k = size(Q, 2); rH = zeros(d * k, d * k)
    idx(a, γ) = (γ - 1) * d + a
    QLQ = Matrix(Q' * (Lp * Q))
    for γ in 1:k, β in 1:k, a in 1:d
        rH[idx(a, γ), idx(a, β)] += 2 * QLQ[γ, β]
    end
    Id = Matrix(I, d, d)
    for i in 1:N-1, j in i+1:N
        r = @views X[:, i] .- X[:, j]; d2 = dot(r, r); s = exp(-d2 / σ2)
        W = 2μ * s .* (Id .- (2 / σ2) .* (r * r'))
        for γ in 1:k
            dqγ = Q[i, γ] - Q[j, γ]; dqγ == 0 && continue
            for β in 1:k
                c = dqγ * (Q[i, β] - Q[j, β]); c == 0 && continue
                for a in 1:d, b in 1:d
                    rH[idx(a, γ), idx(b, β)] -= W[a, b] * c
                end
            end
        end
    end
    rH
end

"""
    ee_reduced_newton_step!(X, Lp, Q, μ; reg, σ2) -> X

One Newton step on the deflation-subspace amplitudes C = X·Q (the modes the deflated inner solver leaves
fixed). Solves rH·Δc = −(G·Q) with rH = `ee_reduced_hessian`, G = `ee_gradient`; writes ΔX = ΔC·Qᵀ.
Quadratically convergent on the reduced system once in the basin; advances the embedding amplitudes so the
continuation can drop the X*-pin.
"""
function ee_reduced_newton_step!(X::AbstractMatrix, Lp::AbstractMatrix, Q::AbstractMatrix, μ::Float64; reg::Float64 = 1e-9, σ2::Float64 = 1.0, linesearch::Bool = true)
    d, N = size(X); k = size(Q, 2)
    G = ee_gradient(X, Lp, μ)
    rH = ee_reduced_hessian(X, Lp, Q, μ; σ2 = σ2)
    # make the reduced Hessian positive so the step is a descent direction even away from the min
    if linesearch
        λmin = eigen(Symmetric(rH)).values[1]
        rH[diagind(rH)] .+= max(reg, reg - λmin)
    else
        rH[diagind(rH)] .+= reg
    end
    GQ = Matrix(G * Q)                                       # d×k reduced gradient
    rg = [GQ[a, γ] for γ in 1:k for a in 1:d]                # (γ−1)d+a order
    ΔX = reshape(-(rH \ rg), d, k) * Q'
    if linesearch
        E0 = ee_energy(X, Lp, μ); gd = dot(ee_gradient(X, Lp, μ), ΔX); s = 1.0; bt = 0
        while (En = ee_energy(X .+ s .* ΔX, Lp, μ); !isfinite(En) || En > E0 + 1e-4 * s * gd) && bt < 30
            s *= 0.5; bt += 1
        end
        X .+= s .* ΔX
    else
        X .+= ΔX
    end
    X
end

"""
    ee_corrector!(X, Lp, P, Φ, μ; n_outer, n_vperp, tol, σ2) -> (X, resid)

Fixed-μ corrector = the deflation split. Each outer sweep: (i) `n_vperp` deflated inner cycles crush the
V⊥ error at ~0.15/cycle (owns Q⊥), (ii) one reduced-V Newton step advances the embedding amplitudes (owns
Q), with Q = `ee_defl_basis(Φ, X)` refreshed each sweep. Converges linear-then-quadratic (block-triangular
iteration matrix, spectral radius max(≈0.15, ρ_Newton)). NO X*-pin — the amplitudes come from the equations.
"""
function ee_corrector!(X::AbstractMatrix, Lp::AbstractMatrix, P::AbstractMatrix, Φ::AbstractMatrix, μ::Float64;
                       n_outer::Int = 12, n_vperp::Int = 1, tol::Float64 = 1e-8, σ2::Float64 = 1.0)
    N = size(X, 2); resid = Inf
    for _ in 1:n_outer
        Q = ee_defl_basis(Φ, X)
        for _ in 1:n_vperp; ee_chord_newton_step!(X, Lp, P, Q, μ; n_inner = 1, σ2 = σ2); end
        ee_reduced_newton_step!(X, Lp, Q, μ; σ2 = σ2)
        resid = norm(ee_A(X, Lp, μ, ones(N); σ2 = σ2))
        resid < tol && break
    end
    X, resid
end

"""One DEFLATED two-level FAS cycle: pre-smooth, aggregation coarse correction, additive deflation of
the near-null modes Q (`deflate_correct!`), post-smooth. Q is the deflation basis (see deflate_correct!)."""
function ee_two_level_deflated!(X, Lp, LpH, P, R, mass, μ, Q; ν1 = 1, ν2 = 2, σ2 = 1.0)
    ee_smooth!(X, Lp, μ, ν1; σ2 = σ2)
    Y0 = Matrix(X * R')
    fH = ee_A(Y0, LpH, μ, mass; σ2 = σ2) .- Matrix(ee_A(X, Lp, μ, ones(size(X, 2)); σ2 = σ2) * P)
    X .+= Matrix((ee_coarse_solve(Y0, LpH, μ, mass, fH; σ2 = σ2) .- Y0) * P')
    deflate_correct!(X, Lp, μ, Q; σ2 = σ2)
    ee_smooth!(X, Lp, μ, ν2; σ2 = σ2)
    X
end

"""Energy-minimizing iterate recombination (DIIS): X ← Σ c_k X_k, min‖Σ c_k A(X_k)‖ s.t. Σc_k=1.
Robust to collinear residuals near convergence (normalize + ridge + fallback)."""
function ee_diis(Xs, Rs)
    m = length(Xs); G = [dot(vec(Rs[k]), vec(Rs[l])) for k in 1:m, l in 1:m]
    sc = tr(G) / m; sc ≤ 0 && return Xs[end]
    G = G ./ sc .+ 1e-8 .* Matrix(I, m, m)
    c = try ([G ones(m); ones(m)' 0.0] \ [zeros(m); 1.0])[1:m] catch; return Xs[end] end
    (any(!isfinite, c) || sum(abs, c) > 1e3) && return Xs[end]
    X = zero(Xs[1]); for k in 1:m; X .+= c[k] .* Xs[k]; end; X
end

# ----------------------------------------------------------------------------------------------
# Related / seed sharpening : p-Laplacian FAS eigensolver  (STUBS — reuse LAMG+ / NLF)
# ----------------------------------------------------------------------------------------------
# The direct Brandt FAS eigenproblem cycle (single cycle, not stacked loops):
#   - smoother: pointwise Gauss–Seidel–Newton on Δ_p x − λ|x|^{p-2}x = 0 (gauss_seidel! is the p=2 core)
#   - λ update by Rayleigh quotient once per cycle
#   - coarsening: LAMG+ affinity aggregation (via `using LAMG`)
#   - interpolation: EIS/BAMG least-squares to capture the low-mode subspace
#   - global normalization constraint (Brandt §5.6)
#   - continuation in p embedded in FMG (via NLF's law! + flow_continuation!)
#
# function fas_eigen(B, w, d, p_target; ...) end   # TODO Phase 2 (single vector) / Phase 3 (subspace)

end # module
