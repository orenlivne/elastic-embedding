# STAGE 0 (clean multi-mode). The swiss roll gives a rank-1 EE embedding (regime_diag.jl) so it cannot
# test the d≥2 solver. Gaussian blobs with k clusters give k−1 GENUINELY INDEPENDENT near-null modes
# (cluster-indicator vectors), so k=d+1 blobs → a real rank-d embedding. For each config: verify the
# ground-truth X* is STATIONARY (gradnorm), that it is genuinely rank-d (singular values), that J has ~d
# near-null modes; then measure the deflated inner solver's V⊥ LINEAR CONVERGENCE factor AND CYCLE
# STATIONARITY (design plan §4 gate + the "test every stage" directive). Run: julia --project=. experiments/stage0_blobs.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

function build(k, nper, dim; sep = 10.0, knn = 10, seed = 1)
    data, labels = gaussian_blobs(nper, k, dim; sep = sep, rng = MersenneTwister(seed))
    B, w, _ = knn_affinity_graph(data, knn); Lp = weighted_laplacian(B, w)
    Lp, labels
end

# solve to a genuine stationary X* (gradient descent to tight gtol), report the regime
function ground_truth(Lp, μ, d; iters = 20000, gtol = 1e-11)
    N = size(Lp, 1)
    X0 = 1e-2 .* laplacian_eigenmaps(Matrix(Lp), d)
    X, _ = ee_minimize(X0, Lp, μ; iters = iters, gtol = gtol)
    gn = norm(ee_gradient(X, Lp, μ))
    Xc = X .- mean(X, dims = 2); sv = svdvals(Xc)
    Je = eigen(Symmetric(Matrix(Lp) .- μ .* build_Lminus_dense(X; σ2 = 1.0))).values
    (X = X, gn = gn, sv = sv, nnull = count(x -> abs(x) < 1e-3, Je), nneg = count(<(0), Je), amp = norm(Xc))
end

function make_Q(Lp, X, μ)
    N = size(Lp, 1); Lm = build_Lminus_dense(X; σ2 = 1.0); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5))
    P, _, _ = piecewise_constant_interpolation(ag.aggregate)
    Q = Matrix(qr(hcat(ones(N), Matrix(X'), TV[:, 1:8])).Q)
    P, Q
end

# CYCLE STATIONARITY: apply the cycle to X* (pinned), residual must not grow
function cycle_stationarity(Lp, X, P, Q, μ; d)
    N = size(Lp, 1); Xc = copy(X)
    r0 = norm(ee_A(X, Lp, μ, ones(N)))
    ee_chord_newton_step!(Xc, Lp, P, Q, μ; n_inner = 1)
    Xc .-= ((Xc .- X) * Q) * Q'
    norm(ee_A(Xc, Lp, μ, ones(N))) / max(r0, 1e-300)
end

# LINEAR CONVERGENCE: V⊥ error factor (embedding pinned to X*, renormalized)
function vperp_factor(Lp, X, P, Q, μ; d, n_inner = 1, sweeps = 25, seeds = 1:3)
    N = size(Lp, 1); fs = Float64[]
    for sd in seeds
        pert = 0.05 .* randn(MersenneTwister(sd), d, N); pert .-= (pert * Q) * Q'
        Y = X .+ pert; rr = Float64[]
        for _ in 1:sweeps
            e0 = norm(Y .- X)
            ee_chord_newton_step!(Y, Lp, P, Q, μ; n_inner = n_inner)
            Y .-= ((Y .- X) * Q) * Q'
            E = Y .- X; push!(rr, norm(E) / max(e0, 1e-300))
            Y .= X .+ (0.05 / max(norm(E), 1e-300)) .* E
        end
        push!(fs, exp(mean(log.(rr[end-4:end]))))
    end
    mean(fs)
end

@printf("STAGE 0 — Gaussian blobs (k clusters → k−1 modes). Clean multi-mode test.\n")
@printf("k(d)  N    μ/μ*_1  gradnorm@X*  ‖X*‖   svals(1..d+1)          #null #neg | cyc-stat  V⊥ n=1  n=2\n")
println("-"^108)
for (k, nper, d) in ((3, 200, 2), (4, 200, 3))
    Lp, _ = build(k, nper, 10); N = size(Lp, 1)
    ν = eigen(Symmetric(Matrix(Lp))).values; μ1 = ν[2] / N
    for c in (2.0, 4.0, 8.0)
        μ = c * μ1
        g = ground_truth(Lp, μ, d)
        P, Q = make_Q(Lp, g.X, μ)
        cs = cycle_stationarity(Lp, g.X, P, Q, μ; d = d)
        f1 = vperp_factor(Lp, g.X, P, Q, μ; d = d, n_inner = 1)
        f2 = vperp_factor(Lp, g.X, P, Q, μ; d = d, n_inner = 2)
        svs = join([@sprintf("%.2f", g.sv[i]) for i in 1:min(d + 1, length(g.sv))], " ")
        @printf("%d(%d) %4d  %5.1f   %.2e   %6.2f  [%s]   %2d   %2d  |  %.2e   %.3f   %.3f\n",
            k, d, N, c, g.gn, g.amp, svs, g.nnull, g.nneg, cs, f1, f2)
    end
end
@printf("\nGATE: pass iff (X* stationary: gradnorm small) AND (genuinely rank-d: d comparable svals) AND\n")
@printf("      (cyc-stat ≈ 1, not ≫1) AND (V⊥ factor ~0.2 / ~0.04, d-independent).\n")
