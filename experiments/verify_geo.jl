# Independent verification of the workflow winner (bamg_ls caliber-5 LS over mass-aware TVs). The
# reported 0.27 used recombination; measure the RAW (no-recomb) two-grid factor + stationarity, which
# is what a multilevel V-cycle actually needs. Compare to caliber-1 and the mock-cycle ideal (0.066).
# Run: julia --project=. experiments/verify_geo.jl

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
ag = aggregate(Lp; X_ext = TV, rng = MersenneTwister(5)); agg = ag.aggregate; nc = ag.n_coarse
mass = Float64[count(==(I), agg) for I in 1:nc]

function bamg_ls_interpolation(agg, TV, A::SparseMatrixCSC, tvw; c = 5, gcap = 3.0, ridge = 1e-8)
    n = length(agg); ncc = maximum(agg); rows = rowvals(A); vals = nonzeros(A); W = Diagonal(sqrt.(tvw))
    seed = zeros(Int, ncc); for i in 1:n; a = agg[i]; seed[a] == 0 && (seed[a] = i); end
    isseed = falses(n); for a in 1:ncc; isseed[seed[a]] = true; end
    Ip = Int[]; Jp = Int[]; Vp = Float64[]; coup = Dict{Int,Float64}()
    for i in 1:n
        if isseed[i]; push!(Ip, i); push!(Jp, agg[i]); push!(Vp, 1.0); continue; end
        empty!(coup)
        for k in nzrange(A, i); r = rows[k]; r == i && continue; a = agg[r]; coup[a] = get(coup, a, 0.0) + abs(vals[k]); end
        coup[agg[i]] = get(coup, agg[i], 0.0) + 1e30
        cand = collect(keys(coup))
        length(cand) > c && (cand = cand[partialsortperm([coup[a] for a in cand], 1:c; rev = true)])
        m = length(cand)
        m == 1 && (push!(Ip, i); push!(Jp, cand[1]); push!(Vp, 1.0); continue)
        Ss = [seed[a] for a in cand]; M = W * TV[Ss, :]'; xi = W * (@view TV[i, :])
        G = M' * M; G .+= (ridge * (tr(G) / m + 1e-300)) .* Matrix(I, m, m); b = M' * xi
        p = try ([2G ones(m); ones(m)' 0.0] \ [2 .* b; 1.0])[1:m] catch; fill(1.0 / m, m) end
        if any(!isfinite, p) || maximum(abs, p) > gcap
            push!(Ip, i); push!(Jp, agg[i]); push!(Vp, 1.0)
        else
            for (idx, a) in enumerate(cand); abs(p[idx]) > 1e-8 && (push!(Ip, i); push!(Jp, a); push!(Vp, p[idx])); end
        end
    end
    sparse(Ip, Jp, Vp, n, ncc), sparse(collect(1:ncc), seed, ones(ncc), ncc, n)
end

function fac(P, R, LpH; recomb)
    X = Xstar .+ 0.05 .* randn(MersenneTwister(3), 1, N); Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]; fs = Float64[]
    for _ in 1:25
        r0 = norm(ee_A(X, Lp, μ, ones(N))); ee_two_level_P!(X, Lp, LpH, P, R, mass, μ); Rr = ee_A(X, Lp, μ, ones(N))
        if recomb
            push!(Xs, copy(X)); push!(Rs, copy(Rr)); length(Xs) > 5 && (popfirst!(Xs); popfirst!(Rs))
            length(Xs) ≥ 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, μ, ones(N)); norm(Ra) < norm(Rr) && (X = Xa; Rr = Ra; Xs[end] = copy(X); Rs[end] = copy(Rr)))
        end
        rn = norm(Rr); (!isfinite(rn) || rn > 1e10) && break; push!(fs, rn / max(r0, 1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, fs); isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end])))
end
stat(P, R, LpH) = (Xc = copy(Xstar); ee_two_level_P!(Xc, Lp, LpH, P, R, mass, μ); norm(ee_A(Xc, Lp, μ, ones(N))) / max(norm(ee_A(Xstar, Lp, μ, ones(N))), 1e-30))

tvw = [norm(TV[:, k])^2 / max(norm(Lp * TV[:, k])^2, 1e-30) for k in 1:8]
P, R = bamg_ls_interpolation(agg, TV, Lp, tvw; c = 5, gcap = 3.0); LpH = galerkin_coarse_operator(Lp, P)
P1, R1, _ = piecewise_constant_interpolation(agg); LpH1 = galerkin_coarse_operator(Lp, P1)
laggedGS(e) = gauss_seidel!(e, Lp; b = μ .* (Lm * e), sweeps = 1)
μmock = cr_shrinkage(N, laggedGS, agg, nc; ν = 2, sweeps = 60, warmup = 20, rng = MersenneTwister(11))

println("interpolation    recomb   RAW(no-recomb)   stationarity")
println("-"^58)
@printf("bamg_ls c=5      %.3f    %.3f            %.2f\n", fac(P, R, LpH; recomb = true), fac(P, R, LpH; recomb = false), stat(P, R, LpH))
@printf("caliber-1        %.3f    %.3f            %.2f\n", fac(P1, R1, LpH1; recomb = true), fac(P1, R1, LpH1; recomb = false), stat(P1, R1, LpH1))
@printf("mock-cycle ideal (smoother + perfect coarse projection): %.3f\n", μmock)
