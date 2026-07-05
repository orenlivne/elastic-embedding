# Replace the stand-in affinity aggregation with the REAL LAMG+ aggregation (energy-ratio guard
# Q=2.5, multi-stage δ-thresholding, hub isolation) and re-measure the two-level EE factor.
# The interpolation bottleneck (smooth slow mode) should shrink if LAMG's energy-ratio-optimal
# aggregation captures the low modes better. Graph has avg deg ≈13 ⇒ no low-degree nodes ⇒ LAMG
# does PURE aggregation (elimination is a no-op) — the clean first test.
# Run:  julia --project=. experiments/lamg_aggregation.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

build_lp(s; rng = MersenneTwister(1)) = begin
    n, edges, w, _ = sbm_graph(fill(s, 4), 10.0 / s, 1.0 / s; rng = rng)
    B, w = incidence(n, edges, w); Lp = weighted_laplacian(B, w)
    (n = n, Lp = Lp, μstar = eigen(Symmetric(Matrix(Lp))).values[2] / n)
end
make_agg(Lp, agg, nc) = begin
    mass = Float64[count(==(I), agg) for I in 1:nc]
    P = sparse(1:length(agg), agg, ones(length(agg)), length(agg), nc)
    (agg = agg, nc = nc, mass = mass, LpH = Matrix(P' * Lp * P))
end
equilibrium(Lp, μ, d) = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 1500, gtol = 1e-11)[1]

function measure_factor(X0, Lp, A, μ, n; cycles = 18, recombine = true, window = 3)
    X = copy(X0); Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]; facs = Float64[]
    for _ in 1:cycles
        r0 = norm(ee_A(X, Lp, μ, ones(n)))
        ee_two_level!(X, Lp, A.LpH, A.agg, A.nc, A.mass, μ)
        R = ee_A(X, Lp, μ, ones(n))
        if recombine
            push!(Xs, copy(X)); push!(Rs, copy(R))
            length(Xs) > window && (popfirst!(Xs); popfirst!(Rs))
            if length(Xs) ≥ 2
                Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, μ, ones(n))
                norm(Ra) < norm(R) && (X = Xa; R = Ra; Xs[end] = copy(X); Rs[end] = copy(R))
            end
        end
        rn = norm(R); (!isfinite(rn) || rn > 1e8) && return Inf
        push!(facs, rn / max(r0, 1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, facs); isempty(fin) && return Inf
    exp(mean(log.(fin[max(1, end-4):end])))
end

pb = build_lp(150); n = pb.n
deg = diag(Matrix(pb.Lp))
@printf("GRAPH: N=%d, deg min=%.0f max=%.0f (no low-degree ⇒ LAMG elimination is a no-op)\n", n, minimum(deg), maximum(deg))

sagg, snc = affinity_aggregation(pb.Lp; ntv = 6, maxsize = 4, θ = 0.15, rng = MersenneTwister(5))
Astand = make_agg(pb.Lp, sagg, snc)
lag = aggregate(pb.Lp; rng = MersenneTwister(5))
Alamg = make_agg(pb.Lp, lag.aggregate, lag.n_coarse)
@printf("stand-in aggregation: %d aggregates (ratio %.2f)\n", snc, snc / n)
@printf("LAMG+   aggregation: %d aggregates (ratio %.2f)\n\n", lag.n_coarse, lag.n_coarse / n)

println("Two-level factor (1-2 cycle, recomb win3):")
println("d   r      stand-in    LAMG+")
println("-"^36)
for d in (1, 2), r in (1.5, 2.0)
    μ = r * pb.μstar
    Xp = equilibrium(pb.Lp, μ, d) .+ 0.05 .* randn(MersenneTwister(3), d, n)
    fs = measure_factor(Xp, pb.Lp, Astand, μ, n)
    fl = measure_factor(Xp, pb.Lp, Alamg, μ, n)
    @printf("%d   %.1f    %7.3f    %7.3f\n", d, r, fs, fl)
end

# lever 1: recombination window sweep (LAMG aggregation, d=2, r=2)
println("\nRecombination window sweep (LAMG+ agg, d=2, r=2):")
let μ = 2pb.μstar, Xp = equilibrium(pb.Lp, 2pb.μstar, 2) .+ 0.05 .* randn(MersenneTwister(3), 2, n)
    for w in (2, 3, 5, 8, 12)
        @printf("  window=%-2d  factor=%.3f\n", w, measure_factor(Xp, pb.Lp, Alamg, μ, n; window = w))
    end
end

# lever 2: aggregate on EE-relevant test vectors (bottom L⁺ eigenvectors) via X_ext
println("\nAggregation on EE-relevant test vectors (bottom-K L⁺ eigenvectors, window 8, d=2, r=2):")
let F = eigen(Symmetric(Matrix(pb.Lp))), μ = 2pb.μstar,
    Xp = equilibrium(pb.Lp, 2pb.μstar, 2) .+ 0.05 .* randn(MersenneTwister(3), 2, n)
    for K in (4, 8, 16)
        lag2 = aggregate(pb.Lp; X_ext = F.vectors[:, 1:K], rng = MersenneTwister(5))
        A2 = make_agg(pb.Lp, lag2.aggregate, lag2.n_coarse)
        @printf("  X_ext K=%-2d  aggs=%-3d  factor=%.3f\n", K, lag2.n_coarse, measure_factor(Xp, pb.Lp, A2, μ, n; window = 8))
    end
end
