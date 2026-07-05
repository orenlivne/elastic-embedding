# Phase-1 gate: the smoother diagnostic (doc/design.md §6).
#
# As p→1 and the p-eigenvector approaches a Cheeger cut, edges where the eigenvector is flat have
# |Bᵀx|→0, so the frozen conductance w|Bᵀx|^{p-2} explodes (p-2<0). The linearized p-Laplacian
# becomes highly anisotropic. QUESTION: does pointwise Gauss–Seidel still smooth (µ_s < ~0.3), so a
# direct FAS eigensolver is viable — or does µ_s→1 near p=1, meaning we must design a cut-aware
# smoother (which then becomes a contribution)?
#
# Run:  julia experiments/smoother_diagnostic.jl
# (base Julia only — no LAMG/NLF fetch needed)

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using Random, Printf, LinearAlgebra

rng = MersenneTwister(1)

# Two-block SBM with a clear community structure (a near-Cheeger cut).
sizes  = [300, 300]
p_in   = 0.05
p_out  = 0.002
n, edges, w, block = sbm_graph(sizes, p_in, p_out; rng = rng)
B, w = incidence(n, edges, w)
@printf("SBM: n=%d, m=%d edges, blocks=%s, p_in=%.3f p_out=%.4f\n",
        n, length(edges), sizes, p_in, p_out)

# Freeze x at the p=2 Fiedler vector (the reference low mode); probe the smoother along p→1.
L2 = weighted_laplacian(B, w)
x  = fiedler_vector(Matrix(L2))

ps        = [2.0, 1.8, 1.5, 1.2, 1.1, 1.05]
eps_rels  = [1e-2, 1e-3, 1e-4]     # regularization floor sweep (controls anisotropy severity)

println("\nGauss–Seidel smoothing factor µ_s (deflated, K=4)  and  ρ (plain GS) vs p")
println("  µ_s < 0.30  → direct FAS viable      µ_s → 1 → need a cut-aware smoother")
@printf("\n%-8s", "p")
for er in eps_rels; @printf("  µ_s(ε=%.0e)  ρ(ε=%.0e)", er, er); end
println("\n" * "-"^8 * "-"^(length(eps_rels)*26))

worst_mu = 0.0
for p in ps
    @printf("%-8.2f", p)
    for er in eps_rels
        L = frozen_plaplacian(B, w, x, p; eps_rel = er)
        mu, rho = smoothing_factor(L; K = 4, sweeps = 30, warmup = 8, ntrials = 3, rng = MersenneTwister(7))
        global worst_mu = max(worst_mu, mu)
        @printf("   %6.3f      %6.3f ", mu, rho)
    end
    println()
end

println("\n" * "="^60)
if worst_mu < 0.30
    @printf("VERDICT: worst µ_s = %.3f < 0.30  →  DIRECT FAS IS VIABLE. Proceed to Phase 2.\n", worst_mu)
elseif worst_mu < 0.6
    @printf("VERDICT: worst µ_s = %.3f (marginal). FAS likely viable; watch the p→1 tail.\n", worst_mu)
else
    @printf("VERDICT: worst µ_s = %.3f ≥ 0.6  →  point-GS DEGRADES. Design a cut-aware smoother\n", worst_mu)
    @printf("         (block/aggregation relaxation across the degenerate direction) — a contribution.\n")
end
