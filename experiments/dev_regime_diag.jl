# WHY does the deflated two-grid diverge in the developed regime? Brandt-style component diagnosis on a
# GENUINELY CONVERGED developed X* (swiss d=1, c=8, gradnorm~1e-5) vs the near-threshold X* (c=1.5).
# Decompose the cycle:
#   (1) SMOOTHER stability: iterate the lagged error map e ← GS_L⁺(μ L⁻ e) (no r) — spectral radius ρ_s.
#       If ρ_s>1 the lagged smoother itself diverges (μL⁻ no longer a small perturbation of L⁺).
#   (2) μL⁻ vs L⁺ scale: top eig of μL⁻(X*) relative to L⁺ (is the perturbation still small?).
#   (3) DEFLATION coverage: ‖(I−QQᵀ)v_k‖ for the bottom eigenvectors v_k of J (does Q still span near-null?).
#   (4) AGGREGATION quality: mock two-grid factor (perfect coarse) vs actual — is P still good for J?
# Run: julia --project=. experiments/dev_regime_diag.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1, :] = t .* cos.(t); Dd[2, :] = h; Dd[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
ν = eigen(Symmetric(Matrix(Lp))).values; μ1 = ν[2] / N; νmax = ν[end]

# spectral radius of the lagged-GS smoother error map e ← GS_L⁺(μ L⁻ e), via power iteration
function smoother_rho(Lm, μ; iters = 60)
    e = randn(MersenneTwister(3), N); e .-= mean(e); e ./= norm(e); ρ = 0.0
    for _ in 1:iters
        rhs = μ .* (Lm * e); z = zeros(N); gauss_seidel!(z, Lp; b = rhs, sweeps = 1)
        ρ = norm(z) / norm(e); e = z ./ (norm(z) + 1e-300)
    end
    ρ
end

for c in (1.5, 8.0)
    μ = c * μ1
    X, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 15000, gtol = 1e-12)
    gn = norm(ee_gradient(X, Lp, μ)); amp = norm(X .- mean(X, dims = 2))
    Lm = build_Lminus_dense(X; σ2 = 1.0)
    J = Matrix(Lp) .- μ .* Lm
    Je = eigen(Symmetric(J)); Jv = Je.vectors
    μLm_top = μ * eigen(Symmetric(Matrix(Lm))).values[end]
    ρs = smoother_rho(Lm, μ)
    # deflation coverage: Q = [1, X, TVs]
    TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    Q = Matrix(qr(hcat(ones(N), Matrix(X'), TV[:, 1:8])).Q)
    cover = [norm(Jv[:, k] .- Q * (Q' * Jv[:, k])) for k in 1:6]      # ‖(I−QQᵀ)v_k‖, small = covered
    @printf("c=%.1f  amp=%.2f gradnorm=%.1e |  L⁺ range=[%.1e,%.1e]  μ·λmax(L⁻)=%.3e  |  ρ_smoother=%.3f\n",
        c, amp, gn, ν[2], νmax, μLm_top, ρs)
    @printf("        J bottom-6 eigs: %s\n", join([@sprintf("%.1e", Je.values[k]) for k in 1:6], " "))
    @printf("        deflation miss ‖(I−QQᵀ)v_k‖ for bottom-6: %s\n", join([@sprintf("%.2f", cover[k]) for k in 1:6], " "))
end
@printf("\nρ_smoother>1 ⇒ lagged smoother diverges (root cause = smoother, not coarse/deflation).\n")
