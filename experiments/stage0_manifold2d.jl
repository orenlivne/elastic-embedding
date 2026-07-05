# STAGE 0 (HARD multi-mode). Blobs (stage0_blobs.jl) are easy — cluster modes are piecewise-constant and
# aggregation-representable. The HARD case is a genuine 2-D MANIFOLD whose two embedding modes are smooth
# (poorly aggregate-representable → where deflation must earn its keep). The swiss roll collapsed to rank-1
# because height(21) ≫ roll extent; a BALANCED sheet activates both intrinsic dims. Build a gently-curved
# 2-D sheet with comparable extents, confirm the EE embedding is genuinely rank-2 and STATIONARY, then test
# the deflated solver (cycle stationarity + V⊥ linear-convergence factor). Run: julia --project=. experiments/stage0_manifold2d.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

# balanced 2-D sheet embedded in 3-D: (u,v) grid, comparable extents, mild curvature in the 3rd axis
function sheet(nu, nv; curv = 0.15, seed = 1)
    rng = MersenneTwister(seed); N = nu * nv
    U = Float64[]; V = Float64[]
    for i in 1:nu, j in 1:nv
        push!(U, (i - 0.5) / nu + 0.15 / nu * randn(rng))
        push!(V, (j - 0.5) / nv + 0.15 / nv * randn(rng))
    end
    D = zeros(3, N); D[1, :] = U; D[2, :] = V; D[3, :] = curv .* sin.(2π .* U) .* cos.(2π .* V)
    D
end

function ground_truth(Lp, μ, d; iters = 20000, gtol = 1e-11)
    N = size(Lp, 1)
    X0 = 1e-2 .* laplacian_eigenmaps(Matrix(Lp), d)
    X, _ = ee_minimize(X0, Lp, μ; iters = iters, gtol = gtol)
    gn = norm(ee_gradient(X, Lp, μ)); Xc = X .- mean(X, dims = 2)
    Je = eigen(Symmetric(Matrix(Lp) .- μ .* build_Lminus_dense(X; σ2 = 1.0))).values
    (X = X, gn = gn, sv = svdvals(Xc), nnull = count(x -> abs(x) < 1e-3, Je), nneg = count(<(0), Je), amp = norm(Xc))
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
    P, Matrix(qr(hcat(ones(N), Matrix(X'), TV[:, 1:8])).Q)
end

cycle_stationarity(Lp, X, P, Q, μ) = begin
    N = size(Lp, 1); Xc = copy(X); r0 = norm(ee_A(X, Lp, μ, ones(N)))
    ee_chord_newton_step!(Xc, Lp, P, Q, μ; n_inner = 1); Xc .-= ((Xc .- X) * Q) * Q'
    norm(ee_A(Xc, Lp, μ, ones(N))) / max(r0, 1e-300)
end

function vperp_factor(Lp, X, P, Q, μ; d, n_inner = 1, sweeps = 25, seeds = 1:3)
    N = size(Lp, 1); fs = Float64[]
    for sd in seeds
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

d = 2
D = sheet(30, 24); N = size(D, 2)
B, w, _ = knn_affinity_graph(D, 10); Lp = weighted_laplacian(B, w)
ν = eigen(Symmetric(Matrix(Lp))).values; μ1 = ν[2] / N
@printf("2-D sheet N=%d. ν2,ν3,ν4 = %.3e %.3e %.3e (ratios %.2f %.2f). μ*_1=ν2/N.\n", N, ν[2], ν[3], ν[4], ν[3]/ν[2], ν[4]/ν[2])
@printf("μ/μ*_1  gradnorm@X*  ‖X*‖   sv1  sv2  sv3   #null #neg | cyc-stat  V⊥ n=1  n=2\n")
println("-"^90)
for c in (1.5, 2.0, 3.0, 5.0)
    μ = c * μ1; g = ground_truth(Lp, μ, d); P, Q = make_Q(Lp, g.X, μ)
    @printf("%5.1f   %.2e   %6.2f  %.2f %.2f %.2f   %2d   %2d  |  %.2e   %.3f   %.3f\n",
        c, g.gn, g.amp, g.sv[1], g.sv[2], length(g.sv) ≥ 3 ? g.sv[3] : 0.0, g.nnull, g.nneg,
        cycle_stationarity(Lp, g.X, P, Q, μ), vperp_factor(Lp, g.X, P, Q, μ; d = d, n_inner = 1),
        vperp_factor(Lp, g.X, P, Q, μ; d = d, n_inner = 2))
end
@printf("\nGATE: rank-2 (sv1≈sv2 ≫ sv3), X* stationary, cyc-stat≈1, V⊥ ~0.2/~0.04.\n")
