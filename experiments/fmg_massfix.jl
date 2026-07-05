# FMG mass-consistency fix, step 1: developedness matching. The prolonged coarse embedding was under-
# developed (rmsd~1) because the same physical μ gives c_coarse = μ·Nc/ν₂^coarse ≠ c_fine = μ·N/ν₂^fine.
# Run the coarse continuation to μ_coarse = c_fine·ν₂^coarse/Nc (same developedness), then prolong + rescale.
# Compare prolonged rmsd + fine sweeps: baseline (μ_target) vs μ-scaled. If μ-scaling gives a good start,
# the mass-distortion is small; if not, mass-weighted repulsion is needed next.
# Run: julia --project=. experiments/fmg_massfix.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

swiss(N) = (t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N);
            D = zeros(3, N); D[1, :] = t .* cos.(t); D[2, :] = h; D[3, :] = t .* sin.(t); D)
fixsign!(X) = for a in 1:size(X, 1); (X[a, argmax(abs.(view(X, a, :)))] < 0) && (X[a, :] .*= -1); end

function fmg_variant(Lp, μ_target, d, Xstar, ν2f; scale_mu)
    N = size(Lp, 1)
    P1 = ee_aggregate_P(Lp); LpH = galerkin_coarse_operator(Lp, P1); Nc = size(LpH, 1)
    ν2c = eigen(Symmetric(Matrix(LpH))).values[2]
    c_fine = μ_target * N / ν2f
    μ_coarse = scale_mu ? c_fine * ν2c / Nc : μ_target            # developedness match vs naive
    Xc, _ = ee_continuation_solve(LpH, μ_coarse, d)
    X = Matrix(Xc * P1') .* sqrt(N / Nc); fixsign!(X)
    Φf = Matrix(qr(P1 * eigen(Symmetric(Matrix(LpH))).vectors[:, 2:7]).Q)
    rmsd0 = procrustes_rmsd(Xstar, X)
    sweeps = 0
    for s in 1:25
        Q = ee_defl_basis(Φf, X)
        ee_chord_newton_step!(X, Lp, P1, Q, μ_target; n_inner = 1); ee_reduced_newton_step!(X, Lp, Q, μ_target)
        sweeps = s; norm(ee_A(X, Lp, μ_target, ones(N))) < 1e-7 && procrustes_rmsd(Xstar, X) < 1e-4 && break
    end
    Nc, rmsd0, procrustes_rmsd(Xstar, X), sweeps, μ_coarse * Nc / ν2c
end

N = 1200; D = swiss(N); B, w, _ = knn_affinity_graph(D, 10); Lp = weighted_laplacian(B, w)
ν2f = eigen(Symmetric(Matrix(Lp))).values[2]; μt = 5.0 * ν2f / N
Xg, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μt; iters = 8000, gtol = 1e-11)
Xstar, _ = ee_newton(Xg, Lp, μt; iters = 25)
@printf("swiss d=1 N=%d, c_fine=5. FMG prolongation quality + fine sweeps:\n", N)
for (tag, sm) in (("baseline (coarse→μ_target)", false), ("μ-scaled (coarse→same c)", true))
    Nc, r0, rf, sw, cc = fmg_variant(Lp, μt, 1, Xstar, ν2f; scale_mu = sm)
    @printf("  %-28s Nc=%d  coarse c=%.1f  prolong rmsd=%.3f  final rmsd=%.2e  fine sweeps=%d\n", tag, Nc, cc, r0, rf, sw)
end
@printf("\nIf μ-scaled gives small prolong rmsd + few sweeps ⇒ developedness was the issue (cheap fix).\n")
