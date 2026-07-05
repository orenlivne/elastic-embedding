# Does a HEAVY-TAILED repulsion lift EE quality toward t-SNE? Gaussian EE caps at ~0.65 (≈ LapEig). Swap the
# repulsion kernel to Student-t-like: E = Σ w⁺‖x_n−x_m‖² − λ Σ log(1+‖x_n−x_m‖²/σ²). Its repulsive force
# ∝ (x_n−x_m)/(1+d²/σ²) ~ 1/d for large d (HEAVY tail, long-range — the mechanism t-SNE/UMAP use). Optimize
# by momentum GD + early exaggeration (t-SNE-style), sweep λ, measure MNIST kNN-purity. This tests the
# QUALITY premise of the heavy-tailed pivot cheaply (no solver changes). Run: julia --project=. experiments/bench_heavytail.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics, DelimitedFiles

BENCH = "/private/tmp/claude-501/-Users-oren-taldavi-code-pathfinder-ui/102a2d7d-1575-41ae-a246-582d565b3f39/scratchpad/bench"
data_all = Matrix(readdlm(joinpath(BENCH, "mnist_data.csv"), ',')'); labs_all = Int.(vec(readdlm(joinpath(BENCH, "mnist_labels.csv"))))
data = data_all; labels = labs_all; N = size(data, 2)
B, w, _ = knn_affinity_graph(data, 15); Lp = weighted_laplacian(B, w)

# heavy-tailed EE energy + gradient (attraction = graph Laplacian; repulsion = −log(1+d²/σ²), all pairs)
function ht_grad(X, Lp, λ; σ2 = 1.0)
    d, N = size(X); G = 2.0 .* (X * Lp); E = tr(X * Lp * X')
    @inbounds for n in 1:N-1, m in n+1:N
        d2 = 0.0; for k in 1:d; δ = X[k, n] - X[k, m]; d2 += δ * δ; end
        E -= λ * log(1 + d2 / σ2); c = -2λ / σ2 / (1 + d2 / σ2)   # dE/dd2 · 2 ; force coefficient
        for k in 1:d; f = c * (X[k, n] - X[k, m]); G[k, n] += f; G[k, m] -= f; end
    end
    G, E
end
# momentum GD + early exaggeration (attraction ×4 for first 25% of iters), t-SNE-style
function ht_solve(Lp, λ, d; iters = 1500, lr = 50.0, σ2 = 1.0, rng = MersenneTwister(0))
    N = size(Lp, 1); X = 1e-4 .* randn(rng, d, N); V = zeros(d, N)
    for it in 1:iters
        exa = it < iters ÷ 4 ? 4.0 : 1.0
        G, _ = ht_grad(X, exa .* Lp, λ; σ2 = σ2)
        mom = it < 250 ? 0.5 : 0.8
        V .= mom .* V .- lr .* G; X .+= V; X .-= mean(X, dims = 2)
    end
    X
end

@printf("MNIST N=%d. Gaussian EE≈0.65, t-SNE=0.84, LapEig=0.61.\n", N)
@printf("HEAVY-TAILED EE (Student-t repulsion, momentum GD + early exaggeration):\n")
@printf("λ         kNN-purity(k=10)   ‖X‖\n"); println("-"^40)
purs = Float64[]
for λ in (0.5, 1.0, 2.0, 4.0, 8.0)
    X = ht_solve(Lp, λ, 2)
    p = knn_purity(X, labels, 10); push!(purs, p)
    @printf("%-6.1f    %.3f              %.1f\n", λ, p, norm(X .- mean(X, dims = 2)))
end
@printf("\nbest heavy-tailed purity = %.3f. If ≈0.80 ⇒ heavy tail lifts quality (pivot viable); if ≈0.65 ⇒ not.\n", maximum(purs))
