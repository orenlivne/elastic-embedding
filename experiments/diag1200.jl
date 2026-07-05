# Isolate the N=1200 NaN: is it the equilibrium solve, or the two-level cycle?
include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using Random, Printf, LinearAlgebra, SparseArrays

n, edges, w, _ = sbm_graph(fill(300, 4), 6.0 / 300, 1.0 / (3 * 300); rng = MersenneTwister(1))
B, w = incidence(n, edges, w); Lp = weighted_laplacian(B, w)
μstar = eigen(Symmetric(Matrix(Lp))).values[2] / n
μ = 2 * μstar
@printf("N=%d, μ*=%.4g, μ=%.4g\n", n, μstar, μ)

ee_A(Y, L, μ, mass) = (G = 2.0 .* (Y * L); nc = size(Y, 2); dd = size(Y, 1);
    for I in 1:nc-1, J in I+1:nc
        d2 = sum(k -> (Y[k, I] - Y[k, J])^2, 1:dd)
        wt = mass[I] * mass[J] * exp(-d2)
        for k in 1:dd; f = -2μ * wt * (Y[k, I] - Y[k, J]); G[k, I] += f; G[k, J] -= f; end
    end; G)

X0 = 1e-2 .* laplacian_eigenmaps(Matrix(Lp), 2)
@printf("seed: finite=%s ‖X0‖=%.3g\n", all(isfinite, X0), norm(X0))
for it in (200, 500, 1000, 2000, 4000)
    X, E = ee_minimize(X0, Lp, μ; iters = it, gtol = 1e-10)
    r = norm(ee_A(X, Lp, μ, ones(n)))
    @printf("ee_minimize iters=%-5d finite=%-5s ‖X‖=%.3g  ‖A(X)‖=%.3g  E=%.4g\n",
            it, all(isfinite, X), norm(X), r, E)
end
