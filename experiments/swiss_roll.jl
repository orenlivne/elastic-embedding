# Real kNN manifold bench (swiss roll): (a) does caliber-2 fire where the graph has 1-D structure?
# (b) does mass-aware aggregation + caliber-2 push the two-level factor toward 0.1? (c) is the
# embedding meaningful (unrolls the manifold)?  Run: julia --project=. experiments/swiss_roll.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

function swiss_roll(N; rng, noise = 0.0)
    t = 1.5π .* (1 .+ 2 .* rand(rng, N))          # manifold coordinate (color)
    h = 21 .* rand(rng, N)
    D = zeros(3, N); D[1, :] = t .* cos.(t); D[2, :] = h; D[3, :] = t .* sin.(t)
    D .+= noise .* randn(rng, 3, N)
    D, t
end

N = 1200; data, tparam = swiss_roll(N; rng = MersenneTwister(1))
B, w, edges = knn_affinity_graph(data, 8)             # kNN graph on the 3-D points
Lp = weighted_laplacian(B, w)
ν2 = eigen(Symmetric(Matrix(Lp))).values[2]
@printf("Swiss roll: N=%d, kNN edges=%d, ν₂=%.4g %s\n", N, length(edges), ν2, ν2 < 1e-8 ? "(DISCONNECTED!)" : "(connected)")
μstar = ν2 / N; μ = 2μstar

equilibrium(d) = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 1500, gtol = 1e-11)[1]
Lm = build_Lminus_dense(equilibrium(1); σ2 = 1.0)
function mass_aware_tvs(K, ν)
    TV = zeros(N, K)
    for k in 1:K
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:ν; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    TV
end
function factor(P, R, LpH, mass, d; cyc = 20, win = 5, seed = 3)
    X = equilibrium(d) .+ 0.05 .* randn(MersenneTwister(seed), d, N)
    Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]; fs = Float64[]
    for _ in 1:cyc
        r0 = norm(ee_A(X, Lp, μ, ones(N))); ee_two_level_P!(X, Lp, LpH, P, R, mass, μ); Rr = ee_A(X, Lp, μ, ones(N))
        push!(Xs, copy(X)); push!(Rs, copy(Rr)); length(Xs) > win && (popfirst!(Xs); popfirst!(Rs))
        length(Xs) ≥ 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, μ, ones(N)); norm(Ra) < norm(Rr) && (X = Xa; Rr = Ra; Xs[end] = copy(X); Rs[end] = copy(Rr)))
        rn = norm(Rr); (!isfinite(rn) || rn > 1e8) && return Inf; push!(fs, rn / max(r0, 1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, fs); isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end])))
end
avg_factor(P, R, LpH, mass, d) = mean(factor(P, R, LpH, mass, d; seed = sd) for sd in 1:4)

for K in (8, 16)
    TV = mass_aware_tvs(K, 4)
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate; nc = ag.n_coarse
    mass = Float64[count(==(I), agg) for I in 1:nc]
    P1, R1, _ = piecewise_constant_interpolation(agg); LpH1 = galerkin_coarse_operator(Lp, P1)
    P2, R2, _, nup = caliber2_interpolation(agg, TV, Lp; τ = 0.5); LpH2 = galerkin_coarse_operator(Lp, P2)
    @printf("\nK=%d mass-aware TVs: aggregates=%d, caliber-2 UPGRADED %d nodes (%.0f%%)\n", K, nc, nup, 100nup / N)
    for d in (1, 2)
        @printf("  d=%d   caliber-1: %.3f    caliber-2: %.3f   (avg of 4 seeds)\n", d,
                avg_factor(P1, R1, LpH1, mass, d), avg_factor(P2, R2, LpH2, mass, d))
    end
end
