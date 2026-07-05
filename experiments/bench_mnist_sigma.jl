# Find EE's empirical quality CEILING on MNIST by sweeping σ (repulsion range) and μ (strength). The
# hypothesis: Gaussian repulsion at σ=1 is too local (no inter-cluster force) → LapEig-like; larger σ gives
# longer-range repulsion (toward t-SNE's heavy tail) and MIGHT separate clusters better. If purity tops out
# well below t-SNE's 0.84 for all σ, EE (Gaussian) is quality-limited. Uses the saved MNIST subsample.
# Run: julia --project=. experiments/bench_mnist_sigma.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics, DelimitedFiles

BENCH = "/private/tmp/claude-501/-Users-oren-taldavi-code-pathfinder-ui/102a2d7d-1575-41ae-a246-582d565b3f39/scratchpad/bench"
data_all = Matrix(readdlm(joinpath(BENCH, "mnist_data.csv"), ',')')      # 784×1000
labs_all = Int.(vec(readdlm(joinpath(BENCH, "mnist_labels.csv"))))
rng = MersenneTwister(3); sel = randperm(rng, size(data_all, 2))[1:500]   # 500 for speed
data = data_all[:, sel]; labels = labs_all[sel]; N = size(data, 2)
B, w, _ = knn_affinity_graph(data, 15); Lp = weighted_laplacian(B, w)
ν2 = ee_bottom_eigvecs(Lp, 2)[1][1]
@printf("MNIST N=%d. Reference: t-SNE≈0.84, LapEig≈0.61 purity.\n", N)
@printf("EE quality vs σ² and developedness c:\n")
@printf("σ²    c=4 purity   c=8 purity   c=16 purity   (resid in parens)\n"); println("-"^62)
for σ2 in (1.0, 4.0, 16.0, 64.0)
    row = String[]
    for c in (4.0, 8.0, 16.0)
        X, res = ee_continuation_solve(Lp, c * ν2 / N, 2; nsteps = 20, n_outer = 30, σ2 = σ2)
        push!(row, @sprintf("%.3f(%.0e)", knn_purity(X, labels, 10), res))
    end
    @printf("%-5.0f %s\n", σ2, join(row, "   "))
end
@printf("\nIf best EE purity ≪ 0.84 across all σ²,c ⇒ EE (Gaussian repulsion) is quality-limited vs t-SNE.\n")
