# Phase-1b: disambiguate the p→1 smoothing degradation.
#
# The plain diagnostic deflates only K=4 low modes with FORWARD GS. Two things may make that
# pessimistic: (1) as p→1 the strong intra-cluster coupling ENLARGES the near-null space, so small-K
# deflation understates smoothing; (2) forward-only GS is not symmetric and smooths anisotropic
# operators poorly — real multigrid uses SYMMETRIC GS (forward+backward). This sweep varies the
# deflation dimension K (a proxy for how many low modes the coarse grid removes) and compares
# forward GS vs proper symmetric GS.
#
#   µ_s < ~0.3 at modest K with SGS → coarse space captures K low modes (EIS caliber↑) + symmetric
#       smoother ⇒ direct FAS viable. Designable, understood.
#   µ_s stays high for all K        → point-GS defeated; need block/line/aggregation relaxation.
#
# Run:  julia experiments/deflation_sweep.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using Random, Printf, LinearAlgebra

rng = MersenneTwister(1)
sizes = [300, 300]; p_in = 0.05; p_out = 0.002
n, edges, w, block = sbm_graph(sizes, p_in, p_out; rng = rng)
B, w = incidence(n, edges, w)
x  = fiedler_vector(Matrix(weighted_laplacian(B, w)))

for (p, er) in [(1.2, 1e-3), (1.1, 1e-3), (1.05, 1e-4)]
    L = frozen_plaplacian(B, w, x, p; eps_rel = er)
    @printf("\np=%.2f, ε=%.0e  (n=%d) —  µ_s vs deflation dim K\n", p, er, n)
    println("K (low modes coarse grid removes)   µ_s(fwd GS)   µ_s(sym GS)")
    println("-"^60)
    for K in [1, 2, 4, 8, 16, 32]
        mu_f, _ = smoothing_factor(L; K=K, sweeps=30, warmup=8, ntrials=3, smoother=:gs,  rng=MersenneTwister(7))
        mu_s, _ = smoothing_factor(L; K=K, sweeps=30, warmup=8, ntrials=3, smoother=:sgs, rng=MersenneTwister(7))
        @printf("K=%-4d                               %6.3f        %6.3f\n", K, mu_f, mu_s)
    end
end

println("\nInterpretation: if SYMMETRIC GS reaches µ_s<0.3 once the coarse grid removes the constant")
println("+ Fiedler (small K), the p→1 anisotropy is handled by (symmetric smoother + subspace")
println("coarsening) — the design already calls for both. If not, a dedicated smoother is the work.")
