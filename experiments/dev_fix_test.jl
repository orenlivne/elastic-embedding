# Localize the developed-regime divergence. c=8 (amp~28, converged X*), swiss d=1. Two candidate fixes:
#   (A) COVERAGE: deflate J's EXACT bottom-K modes (Q=[1, eig(J) bottom-K]) for K=2,4,8,16 — if the factor
#       drops below 1, the cause is that [1,X*,TV] under-covers J's near-null cluster in this regime.
#   (B) COARSE OPERATOR: the Galerkin PᵀL⁻P smears the dense Gaussian; compare lagged vs frozen Lm.
# If (A) fixes it → enrich the deflation basis. If only huge K works → the coarse correction is wrong and
# needs re-evaluated (centroid) repulsion. Run: julia --project=. experiments/dev_fix_test.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1, :] = t .* cos.(t); Dd[2, :] = h; Dd[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
ν = eigen(Symmetric(Matrix(Lp))).values; μ1 = ν[2] / N; d = 1

function vfac(X, P, Q, μ; n_inner = 1, sweeps = 30)
    fs = Float64[]
    for sd in 1:3
        pert = 0.05 .* randn(MersenneTwister(sd), d, N); pert .-= (pert * Q) * Q'
        Y = X .+ pert; rr = Float64[]
        for _ in 1:sweeps
            e0 = norm(Y .- X); ee_chord_newton_step!(Y, Lp, P, Q, μ; n_inner = n_inner)
            Y .-= ((Y .- X) * Q) * Q'; E = Y .- X; push!(rr, norm(E) / max(e0, 1e-300))
            (!isfinite(norm(E)) || norm(E) > 1e12) && break
            Y .= X .+ (0.05 / max(norm(E), 1e-300)) .* E
        end
        fin = filter(x -> isfinite(x) && x > 0, rr)
        push!(fs, isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end]))))
    end
    mean(fs)
end

for c in (8.0,)
    μ = c * μ1
    X, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 15000, gtol = 1e-12)
    Lm = build_Lminus_dense(X; σ2 = 1.0); J = Matrix(Lp) .- μ .* Lm
    Ve = eigen(Symmetric(J)).vectors
    TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); P, _, _ = piecewise_constant_interpolation(ag.aggregate)
    o = ones(N)
    @printf("Developed regime c=%.0f (amp=%.1f). V⊥ factor vs deflation basis:\n", c, norm(X .- mean(X)))
    @printf("  Q=[1,X*,TV(8)] (current)          %.3f\n", vfac(X, P, Matrix(qr(hcat(o, Matrix(X'), TV)).Q), μ))
    for K in (2, 4, 8, 16, 32)
        @printf("  Q=[1, eig(J) bottom-%2d]           %.3f\n", K, vfac(X, P, Matrix(qr(hcat(o, Ve[:, 1:K])).Q), μ))
    end
end
@printf("\nIf bottom-K (small K) → <1, coverage is the fix. If only huge K works, coarse operator is wrong.\n")
