# THE CORRECT NONLINEAR SCHEME (validated). Chord-Newton outer + deflated two-grid inner solver of the
# correction equation J δ = −r, J = L⁺ − μ Lm(X) lagged at the current X (NO knowledge of X* needed for
# the operator). Inner cycle = the demo (deflate_X.jl tg → 0.19): smoother L⁺δ = μLm δ − r, Galerkin
# coarse PᵀJP, ERROR deflation δ ← δ − QQᵀδ. Embedding (near-null) modes are pinned = the CONTINUATION's
# job; the deflated inner solves the V⊥ correction at ~0.2/cycle. Earlier "1.0 stall" was a measurement
# artifact: the error hit the X* accuracy floor (‖J₀x*‖≈1e-4) so geomean-of-last-5 read the floor, not
# the rate. Renormalizing the error each step (like the demo) exposes the true ~0.2.
# Run: julia --project=. experiments/chord_newton.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1, :] = t .* cos.(t); Dd[2, :] = h; Dd[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
μ = 2 * eigen(Symmetric(Matrix(Lp))).values[2] / N

function setup(d)
    Xstar = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 2000, gtol = 1e-11)[1]
    Lm0 = build_Lminus_dense(Xstar; σ2 = 1.0); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm0 * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate
    P, _, _ = piecewise_constant_interpolation(agg)
    Q = Matrix(qr(hcat(ones(N), Matrix(Xstar'), TV[:, 1:8])).Q)
    (Xstar = Xstar, Lm0 = Lm0, P = P, Q = Q)
end

# One chord-Newton step: freeze J at X (lagged) or at X* (frozen); per coord run n_inner deflated
# two-grid cycles solving J δ = −r (r = J x^a); X += δ.
function chord_newton_step!(X, s, μ; n_inner = 1, ν1 = 1, ν2 = 2, frozen = false)
    Lm = frozen ? s.Lm0 : build_Lminus_dense(X; σ2 = 1.0)
    P, Q = s.P, s.Q
    JHi = pinv(Matrix(P' * (Lp * P .- μ .* (Lm * P))))
    Jmul(v) = Lp * v .- μ .* (Lm * v); defl(v) = (v .-= Q * (Q' * v); v)
    for a in 1:size(X, 1)
        r = Jmul(X[a, :]); δ = zeros(N)
        for _ in 1:n_inner
            for _ in 1:ν1; gauss_seidel!(δ, Lp; b = μ .* (Lm * δ) .- r, sweeps = 1); end
            δ .+= P * (JHi * (P' * (-r .- Jmul(δ)))); defl(δ)
            for _ in 1:ν2; gauss_seidel!(δ, Lp; b = μ .* (Lm * δ) .- r, sweeps = 1); end
            defl(δ)
        end
        X[a, :] .+= δ
    end
    X
end

function outer_factor(s, d, μ; frozen, n_inner = 1, sweeps = 25, seeds = 1:3)
    Q = s.Q; fs = Float64[]
    for sd in seeds
        pert = 0.05 .* randn(MersenneTwister(sd), d, N); pert .-= (pert * Q) * Q'
        X = s.Xstar .+ pert; rr = Float64[]
        for _ in 1:sweeps
            e0 = norm(X .- s.Xstar)
            chord_newton_step!(X, s, μ; n_inner = n_inner, frozen = frozen)
            X .-= ((X .- s.Xstar) * Q) * Q'                    # pin embedding modes (continuation's job)
            e1 = norm(X .- s.Xstar); (!isfinite(e1) || e1 > 1e10) && break
            push!(rr, e1 / max(e0, 1e-300))
            E = X .- s.Xstar; X .= s.Xstar .+ (0.05 / max(norm(E), 1e-300)) .* E   # renorm (avoid X* floor)
        end
        fin = filter(x -> isfinite(x) && x > 0, rr); push!(fs, isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end]))))
    end
    mean(fs)
end

@printf("Chord-Newton + deflated two-grid inner. Per-outer-step V⊥ ERROR factor (embedding pinned):\n")
@printf("d   frozen(n=1)  frozen(n=2)  LAGGED(n=1)  LAGGED(n=2)   [lagged = real-solve, no X* in operator]\n")
println("-"^88)
for d in (1, 2, 3)
    s = setup(d)
    @printf("%d     %.3f        %.3f        %.3f        %.3f\n", d,
        outer_factor(s, d, μ; frozen = true, n_inner = 1), outer_factor(s, d, μ; frozen = true, n_inner = 2),
        outer_factor(s, d, μ; frozen = false, n_inner = 1), outer_factor(s, d, μ; frozen = false, n_inner = 2))
end
