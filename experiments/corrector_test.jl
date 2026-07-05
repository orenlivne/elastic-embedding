# Validate the fixed-μ CORRECTOR (deflated inner V⊥ + reduced-V Newton), NO X*-pin. Two checks:
#  (1) ee_reduced_hessian matches the full-Hessian projection ΠᵀHΠ (catch sign/index errors), small N.
#  (2) ee_corrector! converges from a perturbed start to the Newton ground truth: residual → 0 AND
#      procrustes_rmsd(X, X_newton) → 0, with the embedding amplitudes set by the equations (no pin).
# Run: julia --project=. experiments/corrector_test.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

function sheet(nu, nv; curv = 0.15, seed = 1)
    rng = MersenneTwister(seed); N = nu * nv; U = Float64[]; V = Float64[]
    for i in 1:nu, j in 1:nv
        push!(U, (i - 0.5) / nu + 0.15 / nu * randn(rng)); push!(V, (j - 0.5) / nv + 0.15 / nv * randn(rng))
    end
    D = zeros(3, N); D[1, :] = U; D[2, :] = V; D[3, :] = curv .* sin.(2π .* U) .* cos.(2π .* V); D
end
function agg_P(Lp, X, μ)
    N = size(Lp, 1); Lm = build_Lminus_dense(X; σ2 = 1.0); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v); for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    piecewise_constant_interpolation(aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)).aggregate)[1]
end

# ---- (1) reduced-Hessian correctness vs full ΠᵀHΠ, small N ----
D = sheet(12, 10); N = size(D, 2); d = 2
B, w, _ = knn_affinity_graph(D, 8); Lp = weighted_laplacian(B, w)
ev = eigen(Symmetric(Matrix(Lp))); φ = ev.vectors; μ = 3.0 * ev.values[2] / N
Xgd, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 6000, gtol = 1e-11)
X, _ = ee_newton(Xgd, Lp, μ; iters = 20)
Q = ee_defl_basis(φ[:, 2:5], X); k = size(Q, 2)
rH = ee_reduced_hessian(X, Lp, Q, μ)
H = ee_hessian(X, Lp, μ)
Π = zeros(d * N, d * k)
for γ in 1:k, a in 1:d, i in 1:N; Π[(i - 1) * d + a, (γ - 1) * d + a] = Q[i, γ]; end
rH_full = Π' * H * Π
@printf("(1) reduced-Hessian vs ΠᵀHΠ (N=%d,d=%d,k=%d): rel err = %.2e  (should be ~1e-13)\n",
    N, d, k, norm(rH .- rH_full) / norm(rH_full))

# ---- (2) corrector convergence to Newton GT, NO pin ----
println("\n(2) corrector convergence (no X*-pin), residual per outer sweep:")
for (nu, nv, dd) in ((24, 18, 2),)
    D = sheet(nu, nv); N = size(D, 2)
    B, w, _ = knn_affinity_graph(D, 10); Lp = weighted_laplacian(B, w)
    ev = eigen(Symmetric(Matrix(Lp))); φ = ev.vectors; μ = 3.0 * ev.values[2] / N
    Xgd, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), dd), Lp, μ; iters = 8000, gtol = 1e-11)
    Xstar, gnn = ee_newton(Xgd, Lp, μ; iters = 25)
    P = agg_P(Lp, Xstar, μ)
    for (frac, noise, tag) in ((0.97, 0.02, "close predictor"), (0.9, 0.05, "moderate"))
        X = frac .* Xstar .+ noise .* norm(Xstar) / sqrt(N) .* randn(MersenneTwister(7), dd, N)
        @printf("   --- start = %s (%.0f%%·X* + noise): resid=%.2e rmsd=%.3f ---\n", tag, 100frac, norm(ee_A(X, Lp, μ, ones(N))), procrustes_rmsd(Xstar, X))
        for s in 1:10
            Q = ee_defl_basis(φ[:, 2:5], X)
            ee_chord_newton_step!(X, Lp, P, Q, μ; n_inner = 1); ee_reduced_newton_step!(X, Lp, Q, μ)
            @printf("   sweep %2d: resid=%.3e  rmsd_to_Newton=%.3e\n", s, norm(ee_A(X, Lp, μ, ones(N))), procrustes_rmsd(Xstar, X))
        end
    end
end
@printf("\nPASS iff (1) rel err ~1e-13 AND (2) resid & rmsd → 0 (corrector recovers amplitudes with no pin).\n")
