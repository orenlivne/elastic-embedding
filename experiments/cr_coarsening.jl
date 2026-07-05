# Two measurements for the EE FAS design:
#
# PART A — smoother stability/shrinkage: plain per-sweep factor of each candidate relaxation on the
#   FULL operator A = L⁺ − μL⁻(X). Exposes the negative-diagonal breakdown of GS-on-A near μ* and
#   whether frozen-Gaussian GS(L⁺) and Kaczmarz(A) stay stable.
#
# PART B — compatible relaxation (mock cycle): with LAMG-style aggregation defining the coarse
#   variables (= aggregate averages), do relaxation sweeps then zero the coarse variables (subtract
#   each aggregate's average), repeat, and measure the asymptotic shrinkage. Fast ⇒ good coarsening
#   of the full operator (this is the coarsening-by-compatible-relaxation test).
#
# Swept across μ = λ/σ² up to the interesting value (past the first few bifurcations μ*_k = ν_k/N).
# d=1 embedding for a clean operator. Run:  julia experiments/cr_coarsening.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using Random, Printf, LinearAlgebra, SparseArrays, Statistics

rng = MersenneTwister(1)
n, edges, w, block = sbm_graph([100, 100, 100, 100], 0.08, 0.002; rng = rng)   # 4 clusters
B, w = incidence(n, edges, w)
Lp = weighted_laplacian(B, w); N = n
ev = eigen(Symmetric(Matrix(Lp))).values
@printf("N=%d, m=%d. Low graph eigenvalues ν₂..ν₅ = %.3g %.3g %.3g %.3g\n",
        N, length(edges), ev[2], ev[3], ev[4], ev[5])
μstar = ev[2] / N
@printf("First bifurcations μ*_k = ν_k/N:  %.3g %.3g %.3g (k=2,3,4)\n\n", ev[2]/N, ev[3]/N, ev[4]/N)

# LAMG-style aggregation of L⁺ (computed once; L⁺ is fixed). Stand-in for LAMG+ aggregation.
agg, nc = affinity_aggregation(Lp; ntv = 6, maxsize = 4, θ = 0.15, rng = MersenneTwister(5))
@printf("Aggregation: %d aggregates from %d nodes (ratio %.2f)\n\n", nc, N, nc / N)

# plain per-sweep factor (renormalized) — >1 means the relaxation DIVERGES
function plain_factor(relax!; sweeps = 40, warmup = 12, rng = MersenneTwister(7))
    e = randn(rng, N); e .-= mean(e); e ./= norm(e); r = Float64[]
    for _ in 1:sweeps
        n0 = norm(e); relax!(e); push!(r, norm(e) / max(n0, 1e-300)); e ./= max(norm(e), 1e-300)
    end
    exp(mean(log.(r[warmup+1:end] .+ 1e-300)))
end

seed = laplacian_eigenmaps(Matrix(Lp), 1)              # 1×N Fiedler seed
ratios = [0.5, 0.9, 1.1, 1.5, 2.0, 4.0, 8.0]

# frozen-Gaussian LAGGED GS: one GS sweep on L⁺ e = μ L⁻(X) e_old  (Gaussian as frozen source)
laggedGS(Lm, μ) = e -> (rhs = μ .* (Lm * e); gauss_seidel!(e, Lp; b = rhs, sweeps = 1))

println("PART A — plain relaxation per-sweep factor (>1 ⇒ diverges; low modes may amplify — CR deflates them)")
println("μ/μ*    laggedGS(frozen)   GS(A,full)   Kaczmarz(A)")
println("-"^56)
As = Dict{Float64,Matrix{Float64}}(); Lms = Dict{Float64,Matrix{Float64}}(); μs = Dict{Float64,Float64}()
for r in ratios
    μ = r * μstar; μs[r] = μ
    X, _ = ee_minimize(1e-2 .* seed, Lp, μ; iters = 250)     # equilibrium embedding at μ
    Lm = build_Lminus_dense(X); Lms[r] = Lm
    A = ee_operator(Lp, X, μ); As[r] = A
    fL = plain_factor(laggedGS(Lm, μ))
    fA = plain_factor(e -> gauss_seidel_dense!(e, A; sweeps = 1))
    fK = plain_factor(e -> kaczmarz!(e, A; sweeps = 1))
    @printf("%-6.1f  %12.3f    %10.3f   %10.3f\n", r, fL, fA, fK)
end

println("\nPART B — compatible-relaxation shrinkage with aggregation (small ⇒ good coarsening of the FULL operator)")
println("μ/μ*    CR: laggedGS(frozen)   CR: Kaczmarz(A)")
println("-"^50)
for r in ratios
    crL = cr_shrinkage(N, laggedGS(Lms[r], μs[r]), agg, nc; ν = 2, sweeps = 60, warmup = 20, rng = MersenneTwister(11))
    crK = cr_shrinkage(N, e -> kaczmarz!(e, As[r]; sweeps = 1), agg, nc; ν = 2, sweeps = 60, warmup = 20, rng = MersenneTwister(11))
    @printf("%-6.1f  %16.3f     %14.3f\n", r, crL, crK)
end

println("\nRead: PART A tells us which smoother stays stable through μ*; PART B tells us whether the")
println("LAMG aggregation + Gaussian coarsening yields a fast mock cycle (compatible relaxation).")
