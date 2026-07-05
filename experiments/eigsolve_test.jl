# Replace dense eigen() with a SCALABLE bottom-K eigensolver: subspace inverse iteration preconditioned by
# LAMG (each L⁺ solve is O(N)), Rayleigh-Ritz each step. Validate eigenvalues + eigenvector subspace vs
# dense eigen on the swiss-roll L⁺. Run: julia --project=. experiments/eigsolve_test.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

# bottom-K nontrivial eigenpairs of a graph Laplacian Lp (⊥ constant), via LAMG-preconditioned subspace
# inverse iteration + Rayleigh-Ritz. Returns (ν, Φ) with ν ascending, Φ N×K orthonormal (⊥ 1).
function ee_bottom_eigvecs(Lp, K; iters = 40, rng = MersenneTwister(1))
    N = size(Lp, 1); ml = setup(Lp)
    Y = randn(rng, N, K); Y .-= mean(Y, dims = 1); Y = Matrix(qr(Y).Q)
    local ν, Φ
    for _ in 1:iters
        Z = zeros(N, K)
        for k in 1:K
            b = Y[:, k] .- mean(Y[:, k])
            z, _ = solve(ml, b); z .-= mean(z); Z[:, k] = z      # L⁺⁻¹ b, ⊥ constant, O(N)
        end
        Q = Matrix(qr(Z).Q)                                      # orthonormalize the inverse-iterated block
        A = Symmetric(Q' * (Lp * Q)); F = eigen(A)               # Rayleigh-Ritz (K×K, tiny)
        ν = F.values; Φ = Q * F.vectors; Y = Φ                   # rotate to Ritz vectors
    end
    ν, Φ
end

for N in (600, 1200)
    t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
    D = zeros(3, N); D[1, :] = t .* cos.(t); D[2, :] = h; D[3, :] = t .* sin.(t)
    B, w, _ = knn_affinity_graph(D, 10); Lp = weighted_laplacian(B, w)
    K = 8
    ev = eigen(Symmetric(Matrix(Lp)))                            # dense reference
    νref = ev.values[2:K+1]; Φref = ev.vectors[:, 2:K+1]
    ν, Φ = ee_bottom_eigvecs(Lp, K)
    # subspace angle between computed Φ and reference bottom-K
    S = svdvals(Φ' * Φref); ang = maximum(acos.(clamp.(S, -1, 1)))
    @printf("N=%d: eigenvalue rel-err (bottom-%d): %s\n", N, K,
        join([@sprintf("%.1e", abs(ν[k] - νref[k]) / νref[k]) for k in 1:K], " "))
    @printf("        max subspace angle vs dense = %.2e  (small ⇒ same subspace)\n", ang)
end
@printf("\nIf eigenvalues match to ~1e-3 and subspace angle small ⇒ LAMG eigensolver replaces dense eigen.\n")
