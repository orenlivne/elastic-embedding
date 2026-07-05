# Why is the swiss-roll two-level factor ~0.85? Disambiguate: (a) operating point μ too close to a
# dense bifurcation cluster (2-D manifolds have closely-spaced low modes) → larger μ should help; vs
# (b) geometric/grid-like graph = aggregation-AMG weak spot → bad at all μ. Also verify the
# equilibrium X* actually converged (else the factor is measured around garbage).
# Run: julia --project=. experiments/swiss_mu.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
data = zeros(3, N); data[1, :] = t .* cos.(t); data[2, :] = h; data[3, :] = t .* sin.(t)
B, w, edges = knn_affinity_graph(data, 8); Lp = weighted_laplacian(B, w)
evs = eigen(Symmetric(Matrix(Lp))).values
μstar = evs[2] / N
@printf("Swiss roll N=%d, edges=%d.  Low eigenvalues ν₂..ν₆ = %.3g %.3g %.3g %.3g %.3g\n",
        N, length(edges), evs[2], evs[3], evs[4], evs[5], evs[6])
@printf("(spacing ν₃/ν₂=%.2f — a DENSE low spectrum ⇒ many near-critical modes)\n\n", evs[3] / evs[2])

function run_at(μ)
    Xstar, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 2500, gtol = 1e-11)
    resid = norm(ee_A(Xstar, Lp, μ, ones(N)))
    Lm = build_Lminus_dense(Xstar; σ2 = 1.0)
    TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate; nc = ag.n_coarse
    mass = Float64[count(==(I), agg) for I in 1:nc]
    P, R, _ = piecewise_constant_interpolation(agg); LpH = galerkin_coarse_operator(Lp, P)
    X = Xstar .+ 0.05 .* randn(MersenneTwister(3), 1, N); fs = Float64[]
    Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]
    for _ in 1:20
        r0 = norm(ee_A(X, Lp, μ, ones(N))); ee_two_level_P!(X, Lp, LpH, P, R, mass, μ); Rr = ee_A(X, Lp, μ, ones(N))
        push!(Xs, copy(X)); push!(Rs, copy(Rr)); length(Xs) > 5 && (popfirst!(Xs); popfirst!(Rs))
        length(Xs) ≥ 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, μ, ones(N)); norm(Ra) < norm(Rr) && (X = Xa; Rr = Ra; Xs[end] = copy(X); Rs[end] = copy(Rr)))
        rn = norm(Rr); (!isfinite(rn) || rn > 1e8) && return (resid, Inf); push!(fs, rn / max(r0, 1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, fs)
    (resid, isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end]))))
end

println("μ/μ*    equilibrium ‖A(X*)‖    two-level factor (caliber-1, d=1)")
println("-"^58)
for r in (2.0, 5.0, 20.0, 100.0, 500.0)
    resid, fac = run_at(r * μstar)
    @printf("%-7.0f  %.2e            %.3f\n", r, resid, fac)
end
