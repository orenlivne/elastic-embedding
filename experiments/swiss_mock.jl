# Brandt-prescribed localization of the swiss-roll ~0.85 hardness: compare the MOCK-CYCLE factor
# μ_mock (compatible relaxation = smoother + perfect coarse-variable projection) against the actual
# two-level factor μ_2lvl. The GAP tells us whether the deficiency is the COARSE VARIABLE SET
# (μ_mock already bad) or the INTERPOLATION P/T (μ_mock good, μ_2lvl bad).
# Residual reduction is the wrong yardstick for the smooth mass-dominated mode (1984 Guide §8.6/§3.3).
# Run: julia --project=. experiments/swiss_mock.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

function build(kind)
    N = 1200
    if kind == :swiss
        t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
        D = zeros(3, N); D[1, :] = t .* cos.(t); D[2, :] = h; D[3, :] = t .* sin.(t)
        B, w, _ = knn_affinity_graph(D, 8)
    else # SBM reference (irregular, our easy regime)
        n, edges, w0, _ = sbm_graph(fill(300, 4), 10.0 / 300, 1.0 / 300; rng = MersenneTwister(1))
        B, w = incidence(n, edges, w0); return (weighted_laplacian(B, w), n)
    end
    (weighted_laplacian(B, w), N)
end

function localize(kind)
    Lp, N = build(kind)
    μ = 2 * eigen(Symmetric(Matrix(Lp))).values[2] / N
    Xstar = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 2000, gtol = 1e-11)[1]
    Lm = build_Lminus_dense(Xstar; σ2 = 1.0)
    TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate; nc = ag.n_coarse
    mass = Float64[count(==(I), agg) for I in 1:nc]
    # μ_mock: compatible relaxation with the frozen-Gaussian lagged-GS smoother + aggregate projection
    laggedGS(e) = gauss_seidel!(e, Lp; b = μ .* (Lm * e), sweeps = 1)
    μ_mock = cr_shrinkage(N, laggedGS, agg, nc; ν = 2, sweeps = 60, warmup = 20, rng = MersenneTwister(11))
    # μ_2lvl: actual two-level factor (caliber-1, recomb)
    P, R, _ = piecewise_constant_interpolation(agg); LpH = galerkin_coarse_operator(Lp, P)
    X = Xstar .+ 0.05 .* randn(MersenneTwister(3), 1, N); Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]; fs = Float64[]
    for _ in 1:20
        r0 = norm(ee_A(X, Lp, μ, ones(N))); ee_two_level_P!(X, Lp, LpH, P, R, mass, μ); Rr = ee_A(X, Lp, μ, ones(N))
        push!(Xs, copy(X)); push!(Rs, copy(Rr)); length(Xs) > 5 && (popfirst!(Xs); popfirst!(Rs))
        length(Xs) ≥ 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, μ, ones(N)); norm(Ra) < norm(Rr) && (X = Xa; Rr = Ra; Xs[end] = copy(X); Rs[end] = copy(Rr)))
        rn = norm(Rr); (!isfinite(rn) || rn > 1e8) && break; push!(fs, rn / max(r0, 1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, fs); μ_2lvl = isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end])))
    (nc, μ_mock, μ_2lvl)
end

println("kind      aggs   μ_mock (coarse SET)   μ_2lvl (with interp)   → deficiency")
println("-"^72)
for kind in (:sbm, :swiss)
    nc, mk, tl = localize(kind)
    diag = mk > 0.5 ? "COARSE SET (aggregation can't represent geometric modes)" :
           tl > 2mk ? "INTERPOLATION (P/T loses what the coarse set has)" : "both OK"
    @printf("%-8s  %-5d  %.3f                %.3f                  %s\n", kind, nc, mk, tl, diag)
end
