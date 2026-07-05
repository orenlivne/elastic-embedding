# Push the two-level factor to ~0.1 the RIGHT way (Brandt): mass-aware test vectors + LAMG's exact
# selective caliber-2 (upgrade only nodes with exactly 2 strong neighbors — the 1-D anisotropy rule).
#   - test vectors: random vectors relaxed on the FULL operator L⁺−μL⁻ (lagged-GS) — "always via relaxation"
#   - aggregation:  LAMG aggregate with those mass-aware test vectors (X_ext)
#   - interpolation: caliber-1 (piecewise_constant) vs caliber-2 (LAMG caliber2_interpolation), same TVs
# Run:  julia --project=. experiments/caliber2.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

s = 150; n, edges, w, _ = sbm_graph(fill(s, 4), 10.0 / s, 1.0 / s; rng = MersenneTwister(1))
B, w = incidence(n, edges, w); Lp = weighted_laplacian(B, w)
μstar = eigen(Symmetric(Matrix(Lp))).values[2] / n; μ = 2μstar
@printf("N=%d, μ=%.4g\n", n, μ)

equilibrium(d) = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 1500, gtol = 1e-11)[1]
Xstar1 = equilibrium(1); Lm = build_Lminus_dense(Xstar1; σ2 = 1.0)

# mass-aware test vectors: relax random vectors on the FULL operator (lagged-GS on L⁺ with μL⁻ source)
function mass_aware_tvs(K, ν)
    TV = zeros(n, K)
    for k in 1:K
        v = randn(MersenneTwister(40 + k), n); v .-= mean(v)
        for _ in 1:ν; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    TV
end

function factor(P, R, LpH, mass, d; cyc = 20, win = 5)
    X = equilibrium(d) .+ 0.05 .* randn(MersenneTwister(3), d, n)
    Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]; fs = Float64[]
    for _ in 1:cyc
        r0 = norm(ee_A(X, Lp, μ, ones(n))); ee_two_level_P!(X, Lp, LpH, P, R, mass, μ); Rr = ee_A(X, Lp, μ, ones(n))
        push!(Xs, copy(X)); push!(Rs, copy(Rr)); length(Xs) > win && (popfirst!(Xs); popfirst!(Rs))
        length(Xs) ≥ 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, μ, ones(n)); norm(Ra) < norm(Rr) && (X = Xa; Rr = Ra; Xs[end] = copy(X); Rs[end] = copy(Rr)))
        rn = norm(Rr); (!isfinite(rn) || rn > 1e8) && return Inf; push!(fs, rn / max(r0, 1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, fs); isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end])))
end

for K in (4, 8, 16)
    TV = mass_aware_tvs(K, 4)
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate; nc = ag.n_coarse
    mass = Float64[count(==(I), agg) for I in 1:nc]
    P1, R1, _ = piecewise_constant_interpolation(agg); LpH1 = galerkin_coarse_operator(Lp, P1)
    P2, R2, _, nup = caliber2_interpolation(agg, TV, Lp; τ = 0.5); LpH2 = galerkin_coarse_operator(Lp, P2)
    @printf("\nK=%d TVs (mass-aware), aggregates=%d, caliber-2 upgraded %d nodes (%.0f%%):\n", K, nc, nup, 100nup / n)
    for d in (1, 2)
        @printf("  d=%d   caliber-1: %.3f    caliber-2: %.3f\n", d,
                factor(P1, R1, LpH1, mass, d), factor(P2, R2, LpH2, mass, d))
    end
end
