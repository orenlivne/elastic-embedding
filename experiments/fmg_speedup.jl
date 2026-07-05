# FMG acceleration, quantified. The FMG win is NOT fewer fine sweeps per μ — it's avoiding the fine
# CASCADE: a cold fine continuation pays the multi-μ bifurcation tracking on the fine grid, while FMG does
# that on the small coarse grid and the fine grid only refines at μ_target. Count FINE corrector sweeps:
#   cold  = nsteps × (sweeps/step) on the FINE grid
#   FMG   = fine refinement sweeps only (cascade is on the coarse grid)
# Both must match the Newton ground truth. Run: julia --project=. experiments/fmg_speedup.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

swiss(N) = (t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N);
            D = zeros(3, N); D[1, :] = t .* cos.(t); D[2, :] = h; D[3, :] = t .* sin.(t); D)
fixsign!(X) = for a in 1:size(X, 1); (X[a, argmax(abs.(view(X, a, :)))] < 0) && (X[a, :] .*= -1); end

# cold FINE continuation, counting total fine corrector sweeps
function cold_fine(Lp, μ_target, d; nsteps = 16, K = 6)
    N = size(Lp, 1); ev = eigen(Symmetric(Matrix(Lp))); ν = ev.values; Φ = ev.vectors[:, 2:K+d+1]
    P = ee_aggregate_P(Lp); μ0 = 1.15 * ν[d+1] / N; total = 0
    X = zeros(d, N); for a in 1:d; X[a, :] = sqrt(max(μ0 - ν[a+1] / N, 0.0)) .* Φ[:, a]; end
    for μ in exp.(range(log(μ0), log(μ_target), length = nsteps))[2:end]
        for s in 1:20
            Q = ee_defl_basis(Φ[:, 1:K], X)
            ee_chord_newton_step!(X, Lp, P, Q, μ; n_inner = 1); ee_reduced_newton_step!(X, Lp, Q, μ)
            total += 1; norm(ee_A(X, Lp, μ, ones(N))) < 1e-7 && break
        end
    end
    X, total
end
# FMG, counting fine refinement sweeps
function fmg_count(Lp, μ_target, d; K = 6)
    N = size(Lp, 1); P1 = ee_aggregate_P(Lp); LpH = galerkin_coarse_operator(Lp, P1); Nc = size(LpH, 1)
    Xc, _ = ee_continuation_solve(LpH, μ_target * N / Nc, d; K = K)
    X = Matrix(Xc * P1') .* sqrt(N / Nc); fixsign!(X)
    Φf = Matrix(qr(P1 * eigen(Symmetric(Matrix(LpH))).vectors[:, 2:K+1]).Q); total = 0; below = 0
    for s in 1:30
        Q = ee_defl_basis(Φf, X)
        ee_chord_newton_step!(X, Lp, P1, Q, μ_target; n_inner = 1); ee_reduced_newton_step!(X, Lp, Q, μ_target)
        total += 1; below = norm(ee_A(X, Lp, μ_target, ones(N))) < 1e-7 ? below + 1 : 0; below ≥ 2 && break
    end
    X, total, Nc
end

@printf("FMG vs cold fine continuation — FINE corrector-sweep count (both match Newton GT):\n")
@printf("N     Nc    cold fine sweeps   FMG fine sweeps   speedup   cold rmsd   FMG rmsd\n")
println("-"^80)
for N in (600, 1200)
    D = swiss(N); B, w, _ = knn_affinity_graph(D, 10); Lp = weighted_laplacian(B, w)
    μt = 5.0 * eigen(Symmetric(Matrix(Lp))).values[2] / N
    Xg, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μt; iters = 8000, gtol = 1e-11)
    Xstar, _ = ee_newton(Xg, Lp, μt; iters = 25)
    Xc, coldn = cold_fine(Lp, μt, 1); Xf, fmgn, Nc = fmg_count(Lp, μt, 1)
    @printf("%-5d %-5d %10d %17d %10.1f×   %.2e   %.2e\n", N, Nc, coldn, fmgn, coldn / fmgn,
        procrustes_rmsd(Xstar, Xc), procrustes_rmsd(Xstar, Xf))
end
@printf("\nFMG WIN = the bifurcation cascade runs on the small coarse grid; the fine grid only refines.\n")
