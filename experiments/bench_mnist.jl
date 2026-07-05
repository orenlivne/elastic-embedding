# Real-data benchmark: EE (our continuation solver) embedding quality on MNIST vs Laplacian Eigenmaps
# (linear p=2 baseline). Metric = kNN purity (fraction of each point's embedding-neighbors sharing its
# digit label) + trustworthiness. UMAP head-to-head added separately. Data = MNIST IDX (scratchpad/bench).
# Run: julia --project=. experiments/bench_mnist.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

const BENCH = "/private/tmp/claude-501/-Users-oren-taldavi-code-pathfinder-ui/102a2d7d-1575-41ae-a246-582d565b3f39/scratchpad/bench"

function read_idx_images(path)
    io = open(path); read(io, UInt32); n = ntoh(read(io, UInt32)); r = ntoh(read(io, UInt32)); c = ntoh(read(io, UInt32))
    data = read(io, Int(n) * Int(r) * Int(c)); close(io)
    Float64.(reshape(data, Int(r * c), Int(n))) ./ 255.0
end
function read_idx_labels(path)
    io = open(path); read(io, UInt32); n = ntoh(read(io, UInt32)); data = read(io, Int(n)); close(io); Int.(data)
end

# balanced subsample: `per` points per digit class
function subsample(imgs, labs, per; rng = MersenneTwister(0))
    idx = Int[]
    for c in 0:9; ci = findall(==(c), labs); append!(idx, ci[randperm(rng, length(ci))[1:per]]); end
    idx = idx[randperm(rng, length(idx))]
    imgs[:, idx], labs[idx]
end

# trustworthiness (Venna & Kaski): penalize embedding-neighbors that are far in high-dim
function trustworthiness(Xhi, Xlo, k)
    N = size(Xhi, 2)
    dhi = [sum(abs2, @view(Xhi[:, i]) .- @view(Xhi[:, j])) for i in 1:N, j in 1:N]
    dlo = [sum(abs2, @view(Xlo[:, i]) .- @view(Xlo[:, j])) for i in 1:N, j in 1:N]
    rankhi = [sortperm(dhi[i, :]) for i in 1:N]                    # ascending; [1]=self
    s = 0.0
    for i in 1:N
        hiK = Set(rankhi[i][2:k+1]); loK = partialsortperm(dlo[i, :], 2:k+1)
        rpos = invperm(rankhi[i])
        for j in loK; j in hiK || (s += (rpos[j] - 1) - k); end
    end
    1 - 2s / (N * k * (2N - 3k - 1))
end

imgs = read_idx_images(joinpath(BENCH, "t10k-images-idx3-ubyte"))
labs = read_idx_labels(joinpath(BENCH, "t10k-labels-idx1-ubyte"))
data, labels = subsample(imgs, labs, 100)                          # 1000 digits, balanced
N = size(data, 2); @printf("MNIST subsample N=%d (100/digit), dim=%d\n", N, size(data, 1))
B, w, _ = knn_affinity_graph(data, 15); Lp = weighted_laplacian(B, w)
ν2 = ee_bottom_eigvecs(Lp, 2)[1][1]

# Laplacian Eigenmaps baseline (2D)
Xle = laplacian_eigenmaps(Matrix(Lp), 2)
@printf("\nmethod                 kNN-purity(k=10)   trustworthiness(k=10)\n"); println("-"^62)
@printf("Laplacian Eigenmaps      %.3f              %.3f\n", knn_purity(Xle, labels, 10), trustworthiness(data, Xle, 10))

# EE via continuation, sweep developedness c
for c in (2.0, 4.0, 8.0)
    X, res = ee_continuation_solve(Lp, c * ν2 / N, 2; nsteps = 14)
    @printf("EE continuation c=%.0f      %.3f              %.3f     (resid=%.1e)\n",
        c, knn_purity(X, labels, 10), trustworthiness(data, X, 10), res)
end
@printf("\nHigher purity/trust = better cluster preservation. EE (nonlinear repulsion) should beat linear LapEig.\n")
