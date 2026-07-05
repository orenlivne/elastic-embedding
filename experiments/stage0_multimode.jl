# STAGE 0 — HARD GATE (design plan §4, risk #1). Every validated inner-solver number so far was in the
# SINGLE-active-mode regime (μ=2ν₂/N → J has ONE negative eigenvalue). Before building continuation we must
# confirm the deflated V⊥ factor stays ~0.2 and d-independent when d GENUINELY DISTINCT modes are active
# (μ past μ*_d = ν_{d+1}/N). The X*-pin stays in (the amplitudes are the continuation's job). If the factor
# or its d-independence degrades here, the "inner owns V⊥, continuation owns the embedding modes, they never
# fight" premise is unproven and the architecture is re-examined. Run: julia --project=. experiments/stage0_multimode.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1, :] = t .* cos.(t); Dd[2, :] = h; Dd[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
ν = eigen(Symmetric(Matrix(Lp))).values          # ν[1]=0, ν[2]=Fiedler, ...

function setup(d; margin = 2.0)
    μ = margin * ν[d + 1] / N                     # past μ*_d = ν_{d+1}/N ⇒ ≥ d active modes
    Xstar = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 6000, gtol = 1e-11)[1]
    Lm = build_Lminus_dense(Xstar; σ2 = 1.0)
    Jeig = eigen(Symmetric(Matrix(Lp) .- μ .* Lm)).values
    nneg = count(<(−1e-9), Jeig)                  # active (unstable) modes
    nnull = count(x -> abs(x) ≤ 1e-6 * ν[end], Jeig)
    TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate
    P, _, _ = piecewise_constant_interpolation(agg)
    Q = Matrix(qr(hcat(ones(N), Matrix(Xstar'), TV[:, 1:8])).Q)
    (Xstar = Xstar, P = P, Q = Q, μ = μ, nneg = nneg, nnull = nnull, ampl = norm(Xstar .- mean(Xstar, dims = 2)))
end

function vperp_factor(s, d; n_inner = 1, sweeps = 25, seeds = 1:3)
    Q = s.Q; fs = Float64[]
    for sd in seeds
        pert = 0.05 .* randn(MersenneTwister(sd), d, N); pert .-= (pert * Q) * Q'
        X = s.Xstar .+ pert; rr = Float64[]
        for _ in 1:sweeps
            e0 = norm(X .- s.Xstar)
            ee_chord_newton_step!(X, Lp, s.P, Q, s.μ; n_inner = n_inner)
            X .-= ((X .- s.Xstar) * Q) * Q'                                  # X*-pin (continuation's job)
            E = X .- s.Xstar; push!(rr, norm(E) / max(e0, 1e-300))
            X .= s.Xstar .+ (0.05 / max(norm(E), 1e-300)) .* E               # renorm (avoid X* floor)
        end
        push!(fs, exp(mean(log.(rr[end-4:end]))))
    end
    mean(fs)
end

@printf("STAGE 0 — multi-mode deflated V⊥ factor (X*-pinned), swiss roll N=%d\n", N)
@printf("d   μ/(ν2/N)   #active(neg)  #near-null   ‖X*‖    V⊥ factor n=1   n=2\n")
println("-"^78)
for d in (1, 2, 3)
    s = setup(d)
    @printf("%d    %6.2f        %2d          %2d       %6.3f      %.3f        %.3f\n",
        d, s.μ / (ν[2] / N), s.nneg, s.nnull, s.ampl,
        vperp_factor(s, d; n_inner = 1), vperp_factor(s, d; n_inner = 2))
end
@printf("\nGATE: pass iff V⊥ factor stays ~0.2 (n=1) / ~0.04 (n=2) AND is d-independent with ≥d active modes.\n")
