# A full-Hessian Newton ground-truth solver — converges even near the bifurcation where gradient descent
# crawls (critical slowing), giving a genuinely stationary X* to validate against at ANY μ. Also exercises
# the mass-derivative Hessian the reduced-V corrector will need. Then re-check the L⁺-eigenvector deflation
# fix across the FULL μ range (including the c=2 critical-slowing regime) with a valid ground truth.
# H = 2 L⁺⊗I_d − L_W (block Laplacian of W_ij = 2λ·exp(−‖r‖²/σ²)(I − (2/σ²) r rᵀ)),  r = x_i−x_j.
# Run: julia --project=. experiments/newton_gt.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

function ee_hessian(X, Lp, λ; σ2 = 1.0)
    d, N = size(X); H = zeros(d * N, d * N); Lpm = Matrix(Lp)
    idx(a, i) = (i - 1) * d + a
    for i in 1:N, a in 1:d, j in 1:N            # attraction 2 L⁺ ⊗ I_d
        Lpm[i, j] != 0 && (H[idx(a, i), idx(a, j)] += 2 * Lpm[i, j])
    end
    Id = Matrix(I, d, d)
    for i in 1:N-1, j in i+1:N                  # repulsion −L_W
        r = X[:, i] .- X[:, j]; d2 = dot(r, r); s = exp(-d2 / σ2)
        W = 2λ * s .* (Id .- (2 / σ2) .* (r * r'))
        bi = (i - 1) * d; bj = (j - 1) * d
        @views H[bi+1:bi+d, bj+1:bj+d] .+= W; @views H[bj+1:bj+d, bi+1:bi+d] .+= W
        @views H[bi+1:bi+d, bi+1:bi+d] .-= W; @views H[bj+1:bj+d, bj+1:bj+d] .-= W
    end
    H
end

# damped Newton with energy line search + tiny regularization (H indefinite away from min; gauge-null)
function ee_newton(X0, Lp, λ; iters = 60, gtol = 1e-11, reg = 1e-8)
    d, N = size(X0); X = copy(X0); E = ee_energy(X, Lp, λ)
    for it in 1:iters
        G = ee_gradient(X, Lp, λ); gn = norm(G); gn < gtol && break
        H = ee_hessian(X, Lp, λ); H[diagind(H)] .+= reg
        Δ = reshape(-(H \ vec(G)), d, N)
        s = 1.0; bt = 0; Xn = X .+ s .* Δ; En = ee_energy(Xn, Lp, λ)
        while (!isfinite(En) || En > E - 1e-4 * s * dot(G, -Δ)) && bt < 40
            s *= 0.5; Xn = X .+ s .* Δ; En = ee_energy(Xn, Lp, λ); bt += 1
        end
        (bt ≥ 40) && (Xn = X .- 0.1 .* G ./ gn; En = ee_energy(Xn, Lp, λ))   # gradient fallback
        X = Xn; E = En
    end
    X, norm(ee_gradient(X, Lp, λ))
end

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
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v); for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    piecewise_constant_interpolation(aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)).aggregate)[1]
end

@printf("Newton ground truth + L⁺-eigenvector deflation, swiss d=1, ALL regimes:\n")
@printf("c    gradnorm(GD)  gradnorm(Newton)  ‖X*‖   V⊥ [1,X*,TV8]   V⊥ [1,φ(4),X*]\n")
println("-"^78)
o = ones(N)
for c in (2.0, 3.0, 5.0, 8.0)
    μ = c * μ1
    Xgd, ggd = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 15000, gtol = 1e-12)
    X, gn = ee_newton(Xgd, Lp, μ)                # polish GD result to a true stationary point
    P = agg_P(X, μ)
    Lm = build_Lminus_dense(X; σ2 = 1.0); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v); for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    Qcur = Matrix(qr(hcat(o, Matrix(X'), TV)).Q); Qphi = Matrix(qr(hcat(o, φ[:, 2:5], Matrix(X'))).Q)
    @printf("%-4.0f %.2e     %.2e      %6.2f   %.3f          %.3f\n", c, ggd, gn, norm(X .- mean(X)), vfac(X, P, Qcur, μ), vfac(X, P, Qphi, μ))
end
@printf("\nWith a TRUE stationary X* at every μ: does [1,φ(4),X*] give ~0.16 across ALL regimes?\n")
