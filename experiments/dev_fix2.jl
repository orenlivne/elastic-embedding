# The developed-regime divergence is a COVERAGE issue: [1,X*,TV] misses J's near-null cluster, deflating
# eig(J) bottom-4 fixes it (0.16). Since μL⁻ is a small perturbation, J's near-null ≈ L⁺'s bottom
# eigenvectors — sparse-Lanczos-cheap and ALREADY needed by the continuation (μ*_k=ν_k/N, φ_k predictors).
# Verify Q = [1, L⁺ bottom-K eigvecs, X*] recovers ~0.16 across the amplitude range. If so, the fix is
# "use L⁺ eigenvectors (not L⁺-relaxed TVs) as the deflation/aggregation test basis."
# Run: julia --project=. experiments/dev_fix2.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1, :] = t .* cos.(t); Dd[2, :] = h; Dd[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
ev = eigen(Symmetric(Matrix(Lp))); ν = ev.values; φ = ev.vectors; μ1 = ν[2] / N; d = 1

function vfac(X, P, Q, μ; sweeps = 30)
    fs = Float64[]
    for sd in 1:3
        pert = 0.05 .* randn(MersenneTwister(sd), d, N); pert .-= (pert * Q) * Q'
        Y = X .+ pert; rr = Float64[]
        for _ in 1:sweeps
            e0 = norm(Y .- X); ee_chord_newton_step!(Y, Lp, P, Q, μ; n_inner = 1)
            Y .-= ((Y .- X) * Q) * Q'; E = Y .- X; nE = norm(E)
            (!isfinite(nE) || nE > 1e12) && (push!(rr, 10.0); break)
            push!(rr, nE / max(e0, 1e-300)); Y .= X .+ (0.05 / max(nE, 1e-300)) .* E
        end
        fin = filter(x -> isfinite(x) && x > 0, rr); push!(fs, isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end]))))
    end
    mean(fs)
end

function agg_P(X, μ)
    Lm = build_Lminus_dense(X; σ2 = 1.0); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    piecewise_constant_interpolation(aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)).aggregate)[1]
end

o = ones(N)
@printf("V⊥ factor: current [1,X*,TV(8)] vs FIX [1, L⁺ φ₂..φ_{K+1}, X*] (L⁺ eigenvectors), across amplitude:\n")
@printf("c    amp     [1,X*,TV8]   [1,φ(4),X*]  [1,φ(8),X*]\n")
println("-"^58)
for c in (2.0, 5.0, 8.0, 12.0)
    μ = c * μ1
    X, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 15000, gtol = 1e-12)
    P = agg_P(X, μ); amp = norm(X .- mean(X))
    Lm = build_Lminus_dense(X; σ2 = 1.0); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    Qcur = Matrix(qr(hcat(o, Matrix(X'), TV)).Q)
    Q4 = Matrix(qr(hcat(o, φ[:, 2:5], Matrix(X'))).Q)
    Q8 = Matrix(qr(hcat(o, φ[:, 2:9], Matrix(X'))).Q)
    @printf("%-4.0f %6.1f   %.3f        %.3f        %.3f\n", c, amp, vfac(X, P, Qcur, μ), vfac(X, P, Q4, μ), vfac(X, P, Q8, μ))
end
@printf("\nIf [1,φ,X*] stays ~0.15-0.2 across amplitude, the fix is: deflate against L⁺ eigenvectors.\n")
