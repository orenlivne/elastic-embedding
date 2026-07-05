# Does deflation still converge fast when we deflate against the EMBEDDING COORDINATES X (free) instead
# of eig(J) (O(N^3))? At equilibrium A(X*)=2 X* J = 0, so X*'s rows are null vectors of J. Test bases:
#   [1]              constant only
#   [1, X*]          constant + embedding coords (the "nearly free" claim; d=1 here so 1 coord)
#   [1, X*, TV...]   + mass-aware test vectors (already computed for aggregation, also ~near-null)
# vs eig(J) K=2 / K=8 references. Linearized two-grid, caliber-1 P, swiss roll.
# Run: julia --project=. experiments/deflate_X.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1, :] = t .* cos.(t); Dd[2, :] = h; Dd[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
μ = 2 * eigen(Symmetric(Matrix(Lp))).values[2] / N
Xstar = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 2000, gtol = 1e-11)[1]
Lm = build_Lminus_dense(Xstar; σ2 = 1.0); TV = zeros(N, 8)
for k in 1:8
    v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
    for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
    v .-= mean(v); TV[:, k] = v ./ norm(v)
end
ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate
J = Matrix(Lp) .- μ .* Lm
P1, _, _ = piecewise_constant_interpolation(agg); JH = Matrix(P1' * (J * P1)); JHi = pinv(JH)

# linearized two-grid, deflating against an arbitrary basis Vraw (orthonormalized internally)
function tg(Vraw; ν1 = 1, ν2 = 2, sweeps = 60, warmup = 25)
    Q = Matrix(qr(Vraw).Q)[:, 1:size(Vraw, 2)]                # orthonormal deflation basis
    defl(e) = (e .-= Q * (Q' * e); e)
    smooth!(e) = (rhs = μ .* (Lm * e); gauss_seidel!(e, Lp; b = rhs, sweeps = 1); e)
    e = randn(MersenneTwister(7), N); defl(e); e ./= norm(e); rs = Float64[]
    for _ in 1:sweeps
        for _ in 1:ν1; smooth!(e); end
        e .+= P1 * (JHi * (-(P1' * (J * e)))); defl(e)
        for _ in 1:ν2; smooth!(e); end
        defl(e); f = norm(e); push!(rs, f); e ./= max(f, 1e-300)
    end
    exp(mean(log.(rs[warmup+1:end] .+ 1e-300)))
end

xs = vec(Xstar); o = ones(N); Ve = eigen(Symmetric(J)).vectors
@printf("Deflation basis                         dim   two-grid factor\n")
println("-"^60)
@printf("eig(J) bottom-2 (reference)              2     %.3f\n", tg(Ve[:, 1:2]))
@printf("eig(J) bottom-8 (reference)              8     %.3f\n", tg(Ve[:, 1:8]))
@printf("[1] constant only                        1     %.3f\n", tg(reshape(o, N, 1)))
@printf("[1, X*]  (constant + embedding coords)   2     %.3f\n", tg([o xs]))
@printf("[1, X*, TV(2)]                           4     %.3f\n", tg([o xs TV[:, 1:2]]))
@printf("[1, X*, TV(4)]                           6     %.3f\n", tg([o xs TV[:, 1:4]]))
@printf("[1, X*, TV(8)]                           10    %.3f\n", tg([o xs TV]))
