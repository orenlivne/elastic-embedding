using Test, Random, LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding

@testset "graph + incidence" begin
    rng = MersenneTwister(0)
    n, edges, w, block = sbm_graph([50, 50], 0.1, 0.01; rng = rng)
    @test n == 100
    @test length(edges) == length(w)
    @test all(i < j for (i, j) in edges)
    B, w2 = incidence(n, edges, w)
    @test size(B) == (n, length(edges))
    # incidence columns sum to zero (one +1, one -1)
    @test all(abs.(vec(sum(B, dims = 1))) .< 1e-12)
end

@testset "Laplacian + p-weights" begin
    rng = MersenneTwister(1)
    n, edges, w, _ = sbm_graph([40, 40], 0.15, 0.01; rng = rng)
    B, w = incidence(n, edges, w)
    L = weighted_laplacian(B, w)
    @test issymmetric(L)
    @test norm(L * ones(n)) < 1e-10                 # constant is null
    g = edge_differences(B, ones(n))
    @test norm(g) < 1e-12                            # Bᵀ1 = 0
    x = fiedler_vector(Matrix(L))
    c = plaplacian_weights(edge_differences(B, x), w, 1.5)
    @test all(c .> 0)
end

@testset "smoother sanity (p=2)" begin
    rng = MersenneTwister(2)
    n, edges, w, _ = sbm_graph([100, 100], 0.08, 0.005; rng = rng)
    B, w = incidence(n, edges, w)
    L = weighted_laplacian(B, w)
    # symmetric GS on the deflated error must smooth well at p=2
    mu, _ = smoothing_factor(L; K = 2, sweeps = 25, warmup = 6, ntrials = 2, smoother = :sgs,
                             rng = MersenneTwister(3))
    @test mu < 0.4
    # one SGS sweep reduces a random high-frequency residual
    v = randn(rng, n); v .-= sum(v) / n
    r0 = norm(L * v)
    sym_gauss_seidel!(v, L; sweeps = 3); v .-= sum(v) / n
    @test norm(L * v) < r0
end
