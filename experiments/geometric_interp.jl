# Geometric interpolation needs the geometry at the manifold's INTRINSIC dimension. The swiss roll is
# a 2-D sheet, so a d=1 embedding-geometry can't place points on it. Sweep the interpolation geometry:
# bottom-k Laplacian eigenvectors (= unrolled intrinsic coords) and the raw 3-D data. Target: the
# d=1 two-level factor → the μ_mock=0.066 the coarse set already supports.
# Run: julia --project=. experiments/geometric_interp.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
D = zeros(3, N); D[1, :] = t .* cos.(t); D[2, :] = h; D[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(D, 8); Lp = weighted_laplacian(B, w)
F = eigen(Symmetric(Matrix(Lp)))
μ = 2 * F.values[2] / N
Xstar = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 2000, gtol = 1e-11)[1]
Lm = build_Lminus_dense(Xstar; σ2 = 1.0); TV = zeros(N, 8)
for k in 1:8
    v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
    for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
    v .-= mean(v); TV[:, k] = v ./ norm(v)
end
ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate; nc = ag.n_coarse
mass = Float64[count(==(I), agg) for I in 1:nc]
laggedGS(e) = gauss_seidel!(e, Lp; b = μ .* (Lm * e), sweeps = 1)
μ_mock = cr_shrinkage(N, laggedGS, agg, nc; ν = 2, sweeps = 60, warmup = 20, rng = MersenneTwister(11))

function μ2lvl(P, R, LpH)
    X = Xstar .+ 0.05 .* randn(MersenneTwister(3), 1, N); Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]; fs = Float64[]
    for _ in 1:20
        r0 = norm(ee_A(X, Lp, μ, ones(N))); ee_two_level_P!(X, Lp, LpH, P, R, mass, μ); Rr = ee_A(X, Lp, μ, ones(N))
        push!(Xs, copy(X)); push!(Rs, copy(Rr)); length(Xs) > 5 && (popfirst!(Xs); popfirst!(Rs))
        length(Xs) ≥ 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, μ, ones(N)); norm(Ra) < norm(Rr) && (X = Xa; Rr = Ra; Xs[end] = copy(X); Rs[end] = copy(Rr)))
        rn = norm(Rr); (!isfinite(rn) || rn > 1e8) && break; push!(fs, rn / max(r0, 1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, fs); isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end])))
end
function geo(gm)
    P, R = geometric_interpolation(agg, gm)
    μ2lvl(P, R, galerkin_coarse_operator(Lp, P))
end

P1, R1, _ = piecewise_constant_interpolation(agg)
@printf("swiss roll, N=%d, μ_mock=%.3f (target)\n\n", N, μ_mock)
@printf("caliber-1 (piecewise constant):              %.3f\n", μ2lvl(P1, R1, galerkin_coarse_operator(Lp, P1)))
@printf("geometric, geom = d=1 embedding:             %.3f\n", geo(Xstar))
for gd in (2, 3, 5)
    @printf("geometric, geom = bottom-%d eigenvectors:      %.3f\n", gd, geo(Matrix(F.vectors[:, 2:1+gd]')))
end
@printf("geometric, geom = raw 3-D data coordinates:  %.3f\n", geo(D))
