# Decisive test: is DATA-kNN fundamentally hard (geometric/grid-like), or only the clean 2-D swiss
# roll? Real embedding targets (single-cell, MNIST) are CLUSTERED high-dim data — SBM-like where we
# win. Compare two-level factor on: clustered high-D blobs vs the swiss-roll manifold.
# Run: julia --project=. experiments/blobs_bench.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

function two_level_factor(Lp, N, μ)
    Xstar, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 2000, gtol = 1e-11)
    resid = norm(ee_A(Xstar, Lp, μ, ones(N)))
    Lm = build_Lminus_dense(Xstar; σ2 = 1.0); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate; nc = ag.n_coarse
    mass = Float64[count(==(I), agg) for I in 1:nc]
    P, R, _ = piecewise_constant_interpolation(agg); LpH = galerkin_coarse_operator(Lp, P)
    X = Xstar .+ 0.05 .* randn(MersenneTwister(3), 1, N); Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]; fs = Float64[]
    for _ in 1:20
        r0 = norm(ee_A(X, Lp, μ, ones(N))); ee_two_level_P!(X, Lp, LpH, P, R, mass, μ); Rr = ee_A(X, Lp, μ, ones(N))
        push!(Xs, copy(X)); push!(Rs, copy(Rr)); length(Xs) > 5 && (popfirst!(Xs); popfirst!(Rs))
        length(Xs) ≥ 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, μ, ones(N)); norm(Ra) < norm(Rr) && (X = Xa; Rr = Ra; Xs[end] = copy(X); Rs[end] = copy(Rr)))
        rn = norm(Rr); (!isfinite(rn) || rn > 1e8) && break; push!(fs, rn / max(r0, 1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, fs)
    (resid, isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end]))))
end

N = 1200
# Single CONNECTED Gaussian across intrinsic dimensions. Low dim = geometric/grid-like (hard);
# high dim = concentrated distances → random-like/irregular (our easy regime?).
println("Single connected Gaussian, kNN k=12, vs intrinsic dimension:")
for dim in (2, 3, 5, 10, 30, 100)
    data = randn(MersenneTwister(1), dim, N)
    B, w, _ = knn_affinity_graph(data, 12); Lp = weighted_laplacian(B, w)
    evs = eigen(Symmetric(Matrix(Lp))).values
    if evs[2] < 1e-6; @printf("  dim=%-3d  DISCONNECTED\n", dim); continue; end
    resid, fac = two_level_factor(Lp, size(Lp, 1), 2 * evs[2] / size(Lp, 1))
    @printf("  dim=%-3d  ν₂=%.2e  eqm-resid=%.1e  →  two-level factor %.3f\n", dim, evs[2], resid, fac)
end
