# Diagnose the multi-mode regime BEFORE re-running the Stage-0 gate. The first gate attempt picked
# μ = 2ν_{d+1}/N and got ‖X*‖=28.7 with only 1 active mode at d=2 — a suspicious regime. Understand it:
# for each μ and d, report whether ee_minimize's X* is actually STATIONARY (grad norm), its amplitude,
# its EFFECTIVE RANK (singular values — is a "d=2" embedding genuinely 2-D or degenerate rank-1?), and the
# near-null / negative structure of the chord Jacobian J = L⁺ − μL⁻(X*). Goal: locate a μ where a genuine
# rank-d embedding with ~d near-null J-modes exists (the regime the deflated solver is designed for).
# Run: julia --project=. experiments/regime_diag.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1, :] = t .* cos.(t); Dd[2, :] = h; Dd[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
ν = eigen(Symmetric(Matrix(Lp))).values
μ1 = ν[2] / N
@printf("swiss roll N=%d. ν2..ν5 = %.3e %.3e %.3e %.3e ; μ*_k=ν_{k+1}/N ratios to μ*_1: 1, %.1f, %.1f, %.1f\n\n",
    N, ν[2], ν[3], ν[4], ν[5], ν[3]/ν[2], ν[4]/ν[2], ν[5]/ν[2])
@printf("d   μ/μ*_1   gradnorm@X*   ‖X*‖    top-4 singvals of X*            #|λJ|<1e-3   #λJ<0\n")
println("-"^96)
for d in (1, 2)
    for c in (1.5, 2.0, 3.0, 5.0, 8.0, 12.0)
        μ = c * μ1
        X0 = 1e-2 .* laplacian_eigenmaps(Matrix(Lp), d)
        X, _ = ee_minimize(X0, Lp, μ; iters = 8000, gtol = 1e-12)
        gn = norm(ee_gradient(X, Lp, μ))
        Xc = X .- mean(X, dims = 2)
        sv = svdvals(Xc); sv4 = length(sv) ≥ 4 ? sv[1:4] : vcat(sv, zeros(4 - length(sv)))
        Je = eigen(Symmetric(Matrix(Lp) .- μ .* build_Lminus_dense(X; σ2 = 1.0))).values
        nnull = count(x -> abs(x) < 1e-3, Je); nneg = count(<(0), Je)
        @printf("%d   %5.1f    %.2e     %6.3f   [%.3f %.3f %.3f %.3f]      %2d          %2d\n",
            d, c, gn, norm(Xc), sv4[1], sv4[2], sv4[3], sv4[4], nnull, nneg)
    end
    println()
end
