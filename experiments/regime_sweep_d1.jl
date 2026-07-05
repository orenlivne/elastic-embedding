# Does the deflated solver survive the DEVELOPED regime? Clean d=1 (no rank confound): sweep μ so the
# swiss-roll embedding amplitude grows from ≪σ (near-threshold, L⁻≈dense L⁻(0)) to ~√N·σ (developed,
# L⁻ local). Measure X* stationarity (gradnorm), amplitude, AND the V⊥ linear-convergence factor +
# cycle stationarity at each. If the factor stays ~0.2 across amplitudes, the solver is regime-robust and
# the d≥2 issue is purely the rank confound. Run: julia --project=. experiments/regime_sweep_d1.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1, :] = t .* cos.(t); Dd[2, :] = h; Dd[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
ν = eigen(Symmetric(Matrix(Lp))).values; μ1 = ν[2] / N
d = 1

function make_Q(X, μ)
    Lm = build_Lminus_dense(X; σ2 = 1.0); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5))
    P, _, _ = piecewise_constant_interpolation(ag.aggregate)
    P, Matrix(qr(hcat(ones(N), Matrix(X'), TV[:, 1:8])).Q)
end
cyc_stat(X, P, Q, μ) = begin
    Xc = copy(X); r0 = norm(ee_A(X, Lp, μ, ones(N)))
    ee_chord_newton_step!(Xc, Lp, P, Q, μ; n_inner = 1); Xc .-= ((Xc .- X) * Q) * Q'
    norm(ee_A(Xc, Lp, μ, ones(N))) / max(r0, 1e-300)
end
function vfac(X, P, Q, μ; n_inner = 1, sweeps = 25)
    fs = Float64[]
    for sd in 1:3
        pert = 0.05 .* randn(MersenneTwister(sd), d, N); pert .-= (pert * Q) * Q'
        Y = X .+ pert; rr = Float64[]
        for _ in 1:sweeps
            e0 = norm(Y .- X); ee_chord_newton_step!(Y, Lp, P, Q, μ; n_inner = n_inner)
            Y .-= ((Y .- X) * Q) * Q'; E = Y .- X; push!(rr, norm(E) / max(e0, 1e-300))
            Y .= X .+ (0.05 / max(norm(E), 1e-300)) .* E
        end
        push!(fs, exp(mean(log.(rr[end-4:end]))))
    end
    mean(fs)
end

@printf("Swiss-roll d=1, factor vs developing amplitude. σ=1, so amp≪1 near-threshold, amp~√N=%.0f developed.\n", sqrt(N))
@printf("μ/μ*_1  gradnorm@X*   ‖X*‖    amp/σ    cyc-stat   V⊥ n=1   n=2\n")
println("-"^68)
for c in (1.5, 2.0, 3.0, 5.0, 8.0, 12.0)
    μ = c * μ1
    X, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 12000, gtol = 1e-12)
    gn = norm(ee_gradient(X, Lp, μ)); amp = norm(X .- mean(X, dims = 2))
    P, Q = make_Q(X, μ)
    @printf("%5.1f   %.2e   %7.3f  %6.2f   %.2e   %.3f   %.3f\n",
        c, gn, amp, amp, cyc_stat(X, P, Q, μ), vfac(X, P, Q, μ; n_inner = 1), vfac(X, P, Q, μ; n_inner = 2))
end
@printf("\nRegime-robust iff V⊥ factor stays ~0.2 as amplitude grows ≪σ → ~√N·σ.\n")
