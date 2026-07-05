# FMG for EE: the expensive part (continuation through the bifurcation cascade) runs on a SMALL COARSE grid;
# the fine grid only needs cheap refinement from the prolonged coarse embedding, with fine deflation
# eigenvectors INTERPOLATED from the coarse ones (no fine eigensolve). Demonstrate: (a) FMG matches the
# Newton ground truth, (b) fine-grid refinement is a handful of sweeps (vs the full cascade a cold fine
# continuation needs) — the FMG acceleration. Run: julia --project=. experiments/fmg_test.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

swiss(N) = (t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N);
            D = zeros(3, N); D[1, :] = t .* cos.(t); D[2, :] = h; D[3, :] = t .* sin.(t); D)
fixsign!(X) = for a in 1:size(X, 1); (X[a, argmax(abs.(X[a, :]))] < 0) && (X[a, :] .*= -1); end

function fmg2(Lp, μ_target, d; K = 6)
    N = size(Lp, 1)
    P1 = ee_aggregate_P(Lp)                              # fine → coarse interpolation (N × Nc)
    LpH = galerkin_coarse_operator(Lp, P1); Nc = size(LpH, 1)
    # ---- coarse continuation (the expensive cascade, on the small grid) ----
    Xc, _ = ee_continuation_solve(LpH, μ_target, d; K = K)
    # ---- prolong + amplitude rescale + fine refine ----
    X = Matrix(Xc * P1')                                 # d×N prolonged
    X .*= sqrt(N / Nc)                                   # coarse amp ~√Nc → fine ~√N
    fixsign!(X)
    # fine deflation eigenvectors INTERPOLATED from coarse (no fine eigensolve)
    evc = eigen(Symmetric(Matrix(LpH))).vectors[:, 2:K+1]
    Φf = Matrix(qr(P1 * evc).Q)                          # N×K interpolated φ's, orthonormalized
    X, Nc, P1, Φf
end

for (name, D, d, c) in (("swiss d=1", swiss(1200), 1, 5.0),)
    N = size(D, 2); B, w, _ = knn_affinity_graph(D, 10); Lp = weighted_laplacian(B, w)
    μt = c * eigen(Symmetric(Matrix(Lp))).values[2] / N
    Xg, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μt; iters = 8000, gtol = 1e-11)
    Xstar, gn = ee_newton(Xg, Lp, μt; iters = 25)
    X, Nc, P1, Φf = fmg2(Lp, μt, d)
    @printf("%s N=%d → coarse Nc=%d (%.1f× smaller). Newton gradnorm=%.1e\n", name, N, Nc, N / Nc, gn)
    @printf("  after prolong (no refine):  resid=%.2e  rmsd=%.3e\n", norm(ee_A(X, Lp, μt, ones(N))), procrustes_rmsd(Xstar, X))
    for s in 1:12
        Q = ee_defl_basis(Φf, X)
        ee_chord_newton_step!(X, Lp, P1, Q, μt; n_inner = 1); ee_reduced_newton_step!(X, Lp, Q, μt)
        @printf("  refine sweep %2d:  resid=%.2e  rmsd_to_Newton=%.3e\n", s, norm(ee_A(X, Lp, μt, ones(N))), procrustes_rmsd(Xstar, X))
    end
end
@printf("\nDoes the fine refine drive rmsd→0 (same minimum) or stall at 5%% (nearby minimum / amplitude offset)?\n")
