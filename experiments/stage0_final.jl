# STAGE 0 — FINAL gate with the L⁺-eigenvector deflation fix. Genuine multi-mode (balanced 2D sheet, d=2,
# rank-2 near-degenerate) + SIZE sweep, developed regime, Newton ground truth (stationary X*). Deflation
# basis Q = [1, L⁺ φ₂..φ_{K+1}, X-rows]. Gate passes iff V⊥ factor ~0.2 AND size-independent AND d-consistent.
# Run: julia --project=. experiments/stage0_final.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

function sheet(nu, nv; curv = 0.15, seed = 1)
    rng = MersenneTwister(seed); N = nu * nv; U = Float64[]; V = Float64[]
    for i in 1:nu, j in 1:nv
        push!(U, (i - 0.5) / nu + 0.15 / nu * randn(rng)); push!(V, (j - 0.5) / nv + 0.15 / nv * randn(rng))
    end
    D = zeros(3, N); D[1, :] = U; D[2, :] = V; D[3, :] = curv .* sin.(2π .* U) .* cos.(2π .* V); D
end

# Q from L⁺ bottom-K eigenvectors + X rows (the FIX). φ precomputed (dense eig here; Lanczos in production).
defl_basis(φ, X, K) = Matrix(qr(hcat(ones(size(X, 2)), φ[:, 2:K+1], Matrix(X'))).Q)

function agg_P(Lp, X, μ)
    N = size(Lp, 1); Lm = build_Lminus_dense(X; σ2 = 1.0); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v); for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    piecewise_constant_interpolation(aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)).aggregate)[1]
end

cyc_stat(Lp, X, P, Q, μ) = begin
    N = size(Lp, 1); Xc = copy(X); r0 = norm(ee_A(X, Lp, μ, ones(N)))
    ee_chord_newton_step!(Xc, Lp, P, Q, μ; n_inner = 1); Xc .-= ((Xc .- X) * Q) * Q'
    norm(ee_A(Xc, Lp, μ, ones(N))) / max(r0, 1e-300)
end
function vfac(Lp, X, P, Q, μ, d; sweeps = 30, n_inner = 1)
    N = size(Lp, 1); fs = Float64[]
    for sd in 1:3
        pert = 0.05 .* randn(MersenneTwister(sd), d, N); pert .-= (pert * Q) * Q'
        Y = X .+ pert; rr = Float64[]
        for _ in 1:sweeps
            e0 = norm(Y .- X); ee_chord_newton_step!(Y, Lp, P, Q, μ; n_inner = n_inner)
            Y .-= ((Y .- X) * Q) * Q'; E = Y .- X; nE = norm(E)
            (!isfinite(nE) || nE > 1e12) && (push!(rr, 10.0); break)
            push!(rr, nE / max(e0, 1e-300)); Y .= X .+ (0.05 / max(nE, 1e-300)) .* E
        end
        fin = filter(x -> isfinite(x) && x > 0, rr); push!(fs, isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end]))))
    end
    mean(fs)
end

d = 2; K = 6
@printf("STAGE 0 FINAL — 2D sheet d=%d, L⁺-eigenvector deflation, developed regime (c=3), Newton GT.\n", d)
@printf("nu×nv   N     ‖X*‖  gradnorm   sv1 sv2 sv3   V⊥(fix) n=1  n=2   cyc-stat\n")
println("-"^80)
for (nu, nv) in ((20, 15), (28, 21), (36, 28))
    D = sheet(nu, nv); N = size(D, 2)
    B, w, _ = knn_affinity_graph(D, 10); Lp = weighted_laplacian(B, w)
    φ = eigen(Symmetric(Matrix(Lp))).vectors; μ = 3.0 * eigen(Symmetric(Matrix(Lp))).values[2] / N
    Xgd, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 8000, gtol = 1e-11)
    X, gn = ee_newton(Xgd, Lp, μ; iters = 20)
    P = agg_P(Lp, X, μ); Q = defl_basis(φ, X, K); sv = svdvals(X .- mean(X, dims = 2))
    sv3 = length(sv) ≥ 3 ? sv[3] : 0.0
    @printf("%d×%d  %4d  %6.2f  %.1e   %.1f %.1f %.1f   %.3f      %.3f   %.2e\n",
        nu, nv, N, norm(X .- mean(X)), gn, sv[1], sv[2], sv3,
        vfac(Lp, X, P, Q, μ, d; n_inner = 1), vfac(Lp, X, P, Q, μ, d; n_inner = 2), cyc_stat(Lp, X, P, Q, μ))
end
@printf("\nGATE: V⊥ ~0.2 and roughly constant across N (size-independent), rank-2 (sv1≈sv2≫sv3).\n")
