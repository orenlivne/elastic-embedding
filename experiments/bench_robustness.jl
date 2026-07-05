# The fold-robustness edge, tested on existing machinery. Claim: our continuation solver reaches the global
# optimum where plain/local gradient descent STAGNATES (critical slowing near the bifurcation/fold). Compare,
# on the SAME Gaussian-EE energy, at several developedness levels c:
#   E_newton  = full-Hessian Newton (global-ish reference, converges everywhere)
#   E_gd      = gradient descent (ee_minimize, LapEig init, many iters)  ← the "standard local optimizer"
#   E_cont    = our μ-continuation
# If E_cont ≈ E_newton ≪ E_gd (GD stuck at higher energy) → continuation is genuinely more robust.
# Run: julia --project=. experiments/bench_robustness.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

swiss(N) = (t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N);
            D = zeros(3, N); D[1, :] = t .* cos.(t); D[2, :] = h; D[3, :] = t .* sin.(t); D)
N = 800; D = swiss(N); B, w, _ = knn_affinity_graph(D, 10); Lp = weighted_laplacian(B, w)
ν2 = ee_bottom_eigvecs(Lp, 2)[1][1]

@printf("Gaussian EE, swiss d=1 N=%d. Final ENERGY (lower=better optimum) + gradnorm:\n", N)
@printf("c     E_newton      E_gd(grad)          E_cont(grad)        continuation vs GD\n"); println("-"^80)
for c in (1.5, 2.0, 3.0, 5.0)
    μ = c * ν2 / N
    Xg, gg = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 20000, gtol = 1e-12)
    Xn, gn = ee_newton(Xg, Lp, μ; iters = 25); En = ee_energy(Xn, Lp, μ)
    Egd = ee_energy(Xg, Lp, μ)
    Xc, _ = ee_continuation_solve(Lp, μ, 1); gc = norm(ee_gradient(Xc, Lp, μ)); Ec = ee_energy(Xc, Lp, μ)
    verdict = Ec < Egd - 1e-6 * abs(En) ? @sprintf("cont LOWER by %.2e", Egd - Ec) : "tie"
    @printf("%-4.1f  %+.4e  %+.4e(%.0e)  %+.4e(%.0e)  %s\n", c, En, Egd, gg, Ec, gc, verdict)
end
@printf("\nIf E_cont ≈ E_newton and GD stagnates at higher energy (large gradnorm) ⇒ continuation is more robust.\n")
@printf("If GD reaches the same energy ⇒ no robustness edge for a well-behaved optimizer.\n")
