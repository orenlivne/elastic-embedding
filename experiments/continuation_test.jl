# μ-CONTINUATION end-to-end, NO oracle. Start just above the last needed bifurcation μ*_d = ν_{d+1}/N with
# L⁺-eigenvector √-seeds (φ_{a+1} for coordinate a — NOT X*), march μ up to the target, correct at each
# step with ee_corrector!. Validate the final X against the Newton ground truth (procrustes_rmsd → small)
# on swiss d=1 AND the 2D sheet d=2. This is the milestone: a genuine EE embedding computed from scratch.
# Run: julia --project=. experiments/continuation_test.jl

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
swiss(N) = (t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N);
            D = zeros(3, N); D[1, :] = t .* cos.(t); D[2, :] = h; D[3, :] = t .* sin.(t); D)

function build_P(Lp)                             # aggregation on L⁺ once (structural)
    N = size(Lp, 1); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v); gauss_seidel!(v, Lp; sweeps = 4); v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    piecewise_constant_interpolation(aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)).aggregate)[1]
end

# μ-continuation from scratch to μ_target
function continuation(Lp, Φ, ν, μ_target, d; nsteps = 16, K = 6, verbose = false)
    N = size(Lp, 1); P = build_P(Lp)
    μ0 = 1.15 * ν[d+1] / N                        # just above μ*_d (all d modes active)
    X = zeros(d, N)
    for a in 1:d; X[a, :] = sqrt(max(μ0 - ν[a+1] / N, 0.0)) .* Φ[:, a]; end   # √-seed, φ_{a+1}
    Φd = Φ[:, 1:K]
    for μ in exp.(range(log(μ0), log(μ_target), length = nsteps))[2:end]
        _, res = ee_corrector!(X, Lp, P, Φd, μ; n_outer = 20, tol = 1e-7)
        verbose && @printf("     μ/μ*_1=%.2f  resid=%.2e\n", μ * N / ν[2], res)
    end
    X
end

for (name, D, d, c) in (("swiss d=1", swiss(600), 1, 5.0), ("sheet  d=2", sheet(24, 18), 2, 3.0))
    N = size(D, 2); B, w, _ = knn_affinity_graph(D, 10); Lp = weighted_laplacian(B, w)
    ev = eigen(Symmetric(Matrix(Lp))); ν = ev.values; Φ = ev.vectors[:, 2:9]   # φ_2..φ_9
    μ_target = c * ν[2] / N
    # ground truth
    Xg, _ = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ_target; iters = 8000, gtol = 1e-11)
    Xstar, gn = ee_newton(Xg, Lp, μ_target; iters = 25)
    # continuation from scratch
    X = continuation(Lp, Φ, ν, μ_target, d)
    rmsd = procrustes_rmsd(Xstar, X); res = norm(ee_A(X, Lp, μ_target, ones(N)))
    @printf("%s N=%d, μ_target=%.1f·μ*_1: Newton gradnorm=%.1e | continuation resid=%.2e  rmsd_to_Newton=%.3e\n",
        name, N, c, gn, res, rmsd)
end
@printf("\nMILESTONE: rmsd_to_Newton small ⇒ continuation computes the genuine embedding with NO oracle (X*-pin removed).\n")
