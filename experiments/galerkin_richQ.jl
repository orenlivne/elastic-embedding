# Decisive: does the NONLINEAR Galerkin+projection-deflation cycle reach the demo's ~0.2 if the
# deflation basis Q spans the full near-null cluster of J (bottom-8), measured on V⊥ (embedding modes
# pinned = continuation's job)? If yes, the fixed scheme works and we just need ~8 cheap near-null
# modes in Q. Run: julia --project=. experiments/galerkin_richQ.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1, :] = t .* cos.(t); Dd[2, :] = h; Dd[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
μ = 2 * eigen(Symmetric(Matrix(Lp))).values[2] / N
d = 1
Xstar = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 2000, gtol = 1e-11)[1]
Lm = build_Lminus_dense(Xstar; σ2 = 1.0); TV = zeros(N, 8)
for k in 1:8
    v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
    for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
    v .-= mean(v); TV[:, k] = v ./ norm(v)
end
ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate
P, _, _ = piecewise_constant_interpolation(agg)
J = Matrix(Lp) .- μ .* Lm; Ve = eigen(Symmetric(J)).vectors

# V⊥-pinned two-grid factor for the nonlinear Galerkin cycle. metric=:resid uses ‖A(X)‖ (full Hessian
# H=2J+mass-deriv), metric=:err uses ‖X−X*‖ (the actual error the demo measures).
function factor(Qraw; K = size(Qraw, 2), sweeps = 25, seeds = 1:3, metric = :resid)
    Q = Matrix(qr(Qraw).Q)[:, 1:K]; fs = Float64[]
    meas(X) = metric === :err ? norm(X .- Xstar) : norm(ee_A(X, Lp, μ, ones(N)))
    for sd in seeds
        pert = 0.05 .* randn(MersenneTwister(sd), d, N); pert .-= (pert * Q) * Q'
        X = Xstar .+ pert; rr = Float64[]
        for _ in 1:sweeps
            r0 = meas(X); ee_two_level_galerkin!(X, Lp, P, μ, Q)
            X .-= ((X .- Xstar) * Q) * Q'                       # pin embedding modes
            rn = meas(X); (!isfinite(rn) || rn > 1e10) && break
            push!(rr, rn / max(r0, 1e-300)); rn < 1e-13 && break
        end
        fin = filter(x -> isfinite(x) && x > 0, rr); push!(fs, isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end]))))
    end
    mean(fs)
end

o = ones(N); xs = vec(Xstar)
@printf("Nonlinear Galerkin+proj-defl cycle, V⊥ factor (embedding pinned):\n")
@printf("  Q = [1, X*, TV(4)]            (%2d)   %.3f\n", 6, factor(hcat(o, xs, TV[:, 1:4])))
@printf("  Q = [1, eig(J) bottom-8]      (%2d)   %.3f\n", 9, factor(hcat(o, Ve[:, 1:8])))
@printf("  Q = [1, eig(J) bottom-16]     (%2d)   %.3f\n", 17, factor(hcat(o, Ve[:, 1:16])))
@printf("  Q = [1, X*, TV(8)]           (%2d)   %.3f\n", 10, factor(hcat(o, xs, TV[:, 1:8])))
@printf("\nIf rich Q -> ~0.2, the FAS-fix scheme works; the task is getting ~8-16 near-null modes into Q cheaply.\n")
