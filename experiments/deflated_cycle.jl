# Verify the DEFLATED nonlinear two-level cycle (ee_two_level_deflated!) reproduces the linearized ~0.2
# on geometric graphs, at d=1 AND d=2 (the real regime). Deflation basis Q = [1, X* rows, few TVs] --
# all free. Measure raw (no-recomb) + recomb factors + stationarity, averaged over 3 seeds, vs the
# non-deflated cycle. Run: julia --project=. experiments/deflated_cycle.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5π .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1, :] = t .* cos.(t); Dd[2, :] = h; Dd[3, :] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
μ = 2 * eigen(Symmetric(Matrix(Lp))).values[2] / N

function setup(d)
    Xstar = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 2000, gtol = 1e-11)[1]
    Lm = build_Lminus_dense(Xstar; σ2 = 1.0); TV = zeros(N, 8)
    for k in 1:8
        v = randn(MersenneTwister(40 + k), N); v .-= mean(v)
        for _ in 1:4; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate; nc = ag.n_coarse
    mass = Float64[count(==(I), agg) for I in 1:nc]
    P, R, _ = piecewise_constant_interpolation(agg); LpH = galerkin_coarse_operator(Lp, P)
    Qraw = hcat(ones(N), Matrix(Xstar'), TV[:, 1:4])                     # [1, X* rows, 4 TVs] — all free
    Q = Matrix(qr(Qraw).Q)[:, 1:size(Qraw, 2)]
    (Xstar = Xstar, P = P, R = R, LpH = LpH, mass = mass, Q = Q)
end

function factor(cycle!, s, d; recomb, seeds = 1:3)
    fs = Float64[]
    for sd in seeds
        X = s.Xstar .+ 0.05 .* randn(MersenneTwister(sd), d, N); Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]; rr = Float64[]
        for _ in 1:25
            r0 = norm(ee_A(X, Lp, μ, ones(N))); cycle!(X); Rr = ee_A(X, Lp, μ, ones(N))
            if recomb
                push!(Xs, copy(X)); push!(Rs, copy(Rr)); length(Xs) > 5 && (popfirst!(Xs); popfirst!(Rs))
                length(Xs) ≥ 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, μ, ones(N)); norm(Ra) < norm(Rr) && (X = Xa; Rr = Ra; Xs[end] = copy(X); Rs[end] = copy(Rr)))
            end
            rn = norm(Rr); (!isfinite(rn) || rn > 1e10) && break; push!(rr, rn / max(r0, 1e-300)); rn < 1e-12 && break
        end
        fin = filter(x -> isfinite(x) && x > 0, rr); push!(fs, isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end]))))
    end
    mean(fs)
end
stat(cycle!, s, d) = (Xc = copy(s.Xstar); cycle!(Xc); norm(ee_A(Xc, Lp, μ, ones(N))) / max(norm(ee_A(s.Xstar, Lp, μ, ones(N))), 1e-30))

println("swiss roll, N=$N.  cycle factor (avg 3 seeds), stationarity from X*:")
println("d   variant       raw     recomb   stationarity")
println("-"^50)
for d in (1, 2)
    s = setup(d)
    nd(X) = ee_two_level_P!(X, Lp, s.LpH, s.P, s.R, s.mass, μ)
    df(X) = ee_two_level_deflated!(X, Lp, s.LpH, s.P, s.R, s.mass, μ, s.Q)
    @printf("%d   non-deflated  %.3f   %.3f    %.1f\n", d, factor(nd, s, d; recomb = false), factor(nd, s, d; recomb = true), stat(nd, s, d))
    @printf("%d   DEFLATED      %.3f   %.3f    %.2f\n", d, factor(df, s, d; recomb = false), factor(df, s, d; recomb = true), stat(df, s, d))
end
