# BAMG least-squares caliber-2/3 interpolation over the MASS-AWARE test vectors TV
# (algebraic higher-order, LAMG style) for the two-level EE FAS cycle on a swiss-roll kNN graph.
#
# Variant (1): existing selective caliber-2  caliber2_interpolation(agg, TV, Lp; τ) sweeping τ.
# Variant (2): general caliber-3 LS — each non-seed node fits weights (sum=1) over its 3 nearest
#              aggregates (by |Lp| coupling) minimizing Σ_k (TV[i,k]-Σ_J p_J TV[seed_J,k])² via KKT,
#              guard |p|>3 → caliber-1. Seeds caliber-1, R = seed injection (R·P=I).
#
# Run:  cd /Users/oren/code/elastic-embedding && julia --project=. experiments/geo_bamg_ls_caliber2.jl
# Reference:  caliber-1=0.856, prior-geometric=0.558, ideal(mock)=0.066.

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

N = 1200
t = 1.5*pi .* (1 .+ 2 .* rand(MersenneTwister(1), N)); h = 21 .* rand(MersenneTwister(2), N)
Dd = zeros(3, N); Dd[1,:] = t .* cos.(t); Dd[2,:] = h; Dd[3,:] = t .* sin.(t)
B, w, _ = knn_affinity_graph(Dd, 8); Lp = weighted_laplacian(B, w)
F = eigen(Symmetric(Matrix(Lp))); mu = 2 * F.values[2] / N
Xstar = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, mu; iters=2000, gtol=1e-11)[1]
Lm = build_Lminus_dense(Xstar; σ2=1.0)
TV = zeros(N, 8)
for k in 1:8
    v = randn(MersenneTwister(40+k), N); v .-= mean(v)
    for _ in 1:4; gauss_seidel!(v, Lp; b = mu .* (Lm*v), sweeps=1); end
    v .-= mean(v); TV[:,k] = v ./ norm(v)
end
ag = aggregate(Lp; X_ext=TV, rng=MersenneTwister(5)); agg = ag.aggregate; nc = ag.n_coarse
mass = Float64[count(==(I), agg) for I in 1:nc]

# smoothness weights ω_k = ‖x_k‖²/‖Lp x_k‖² per test vector (Brandt LS weighting)
tvw = [ (nx = norm(@view TV[:,k]); nax = norm(Lp * (@view TV[:,k])); (nx*nx) / max(nax*nax, 1e-30) ) for k in 1:8 ]

# ---- general caliber-c LS interpolation over TV, c nearest aggregates by |Lp| coupling ----
# For each non-seed i: pick the c aggregates most strongly |Lp|-coupled to i (summed |Lp[i,r]| over
# members r of that aggregate). Fit p (Σp=1) minimizing Σ_k ω_k (TV[i,k]-Σ_J p_J TV[seed_J,k])² via KKT.
# Guard |p|>gcap → caliber-1. Seeds caliber-1; R = seed injection (R·P=I).
function bamg_ls_interpolation(agg::AbstractVector{Int}, TV::AbstractMatrix, A::SparseMatrixCSC,
                               tvw::AbstractVector; c::Int = 3, gcap::Float64 = 3.0, ridge::Float64 = 1e-8)
    n = length(agg); ncc = maximum(agg); K = size(TV, 2)
    seed = zeros(Int, ncc); @inbounds for i in 1:n; a = agg[i]; seed[a] == 0 && (seed[a] = i); end
    isseed = falses(n); for a in 1:ncc; isseed[seed[a]] = true; end
    rows = rowvals(A); vals = nonzeros(A)
    W = Diagonal(sqrt.(tvw))                                     # per-TV sqrt weights
    Ip = Int[]; Jp = Int[]; Vp = Float64[]; n_up = 0
    coup = Dict{Int,Float64}()
    @inbounds for i in 1:n
        if isseed[i]; push!(Ip, i); push!(Jp, agg[i]); push!(Vp, 1.0); continue; end
        empty!(coup)
        for k in nzrange(A, i)
            r = rows[k]; r == i && continue
            a = agg[r]; coup[a] = get(coup, a, 0.0) + abs(vals[k])
        end
        # own aggregate must be a candidate too (self-coupling proxy = large)
        aown = agg[i]; coup[aown] = get(coup, aown, 0.0) + 1e30
        cand = collect(keys(coup))
        if length(cand) > c
            perm = partialsortperm([coup[a] for a in cand], 1:c; rev = true)
            cand = cand[perm]
        end
        m = length(cand)
        if m == 1
            push!(Ip, i); push!(Jp, cand[1]); push!(Vp, 1.0); continue
        end
        Ss = [seed[a] for a in cand]
        M = W * TV[Ss, :]'                                       # K × m weighted seed test-values
        xi = W * (@view TV[i, :])                                # K weighted target
        G = M' * M                                               # m × m
        G .+= (ridge * (tr(G) / m + 1e-300)) .* Matrix(I, m, m)
        b = M' * xi
        p = try
            ([2G ones(m); ones(m)' 0.0] \ [2 .* b; 1.0])[1:m]
        catch
            fill(1.0 / m, m)
        end
        if any(!isfinite, p) || maximum(abs, p) > gcap
            push!(Ip, i); push!(Jp, agg[i]); push!(Vp, 1.0)
        else
            wrote = 0
            for (idx, a) in enumerate(cand)
                if abs(p[idx]) > 1e-8
                    push!(Ip, i); push!(Jp, a); push!(Vp, p[idx]); wrote += 1
                end
            end
            wrote > 1 && (n_up += 1)
        end
    end
    P = sparse(Ip, Jp, Vp, n, ncc)
    R = sparse(collect(1:ncc), seed, ones(ncc), ncc, n)
    return P, R, n_up
end

function mu2lvl(P, R, LpH)
    X = Xstar .+ 0.05 .* randn(MersenneTwister(3), 1, N); Xs=Matrix{Float64}[]; Rs=Matrix{Float64}[]; fs=Float64[]
    for _ in 1:20
        r0 = norm(ee_A(X, Lp, mu, ones(N))); ee_two_level_P!(X, Lp, LpH, P, R, mass, mu); Rr = ee_A(X, Lp, mu, ones(N))
        push!(Xs, copy(X)); push!(Rs, copy(Rr)); length(Xs) > 5 && (popfirst!(Xs); popfirst!(Rs))
        length(Xs) >= 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, mu, ones(N)); norm(Ra) < norm(Rr) && (X=Xa; Rr=Ra; Xs[end]=copy(X); Rs[end]=copy(Rr)))
        rn = norm(Rr); (!isfinite(rn) || rn > 1e8) && break; push!(fs, rn/max(r0,1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, fs); isempty(fin) ? Inf : exp(mean(log.(fin[max(1,end-4):end])))
end

function statcheck(P, R, LpH)
    Xc = copy(Xstar); ee_two_level_P!(Xc, Lp, LpH, P, R, mass, mu)
    norm(ee_A(Xc, Lp, mu, ones(N))) / max(norm(ee_A(Xstar, Lp, mu, ones(N))), 1e-30)
end

println("N=$N  n_coarse=$nc  mu=$(round(mu, sigdigits=4))")
println("Reference: caliber-1=0.856  prior-geometric=0.558  ideal(mock)=0.066\n")

# ---- Variant (1): selective caliber-2, sweep τ ----
println("=== Variant (1): caliber2_interpolation(agg, TV, Lp; τ), sweep τ ===")
best1 = Inf
for tau in (0.5, 0.35, 0.25, 0.15, 0.1)
    P, R, _, nup = caliber2_interpolation(agg, TV, Lp; τ = tau)
    LpH = galerkin_coarse_operator(Lp, P)
    f = mu2lvl(P, R, LpH); s = statcheck(P, R, LpH)
    global best1 = min(best1, f)
    @printf("  τ=%.2f  n_upgraded=%4d  factor=%.4f  stationarity=%.2e\n", tau, nup, f, s)
end

# ---- Variant (2): general caliber-c LS over TV ----
println("\n=== Variant (2): general BAMG LS over TV, sweep c (nearest aggregates) ===")
best2 = Inf; best2c = 0; best2stat = NaN
for c in (4, 5, 6, 7, 8, 10)
    P, R, nup = bamg_ls_interpolation(agg, TV, Lp, tvw; c = c, gcap = 3.0)
    LpH = galerkin_coarse_operator(Lp, P)
    f = mu2lvl(P, R, LpH); s = statcheck(P, R, LpH)
    if f < best2; global best2 = f; global best2c = c; global best2stat = s; end
    @printf("  c=%d  n_upgraded=%4d  factor=%.4f  stationarity=%.2e\n", c, nup, f, s)
end

# also sweep the extrapolation guard cap at the best c
println("\n=== Variant (2b): sweep guard cap gcap at c=$best2c ===")
for gcap in (2.5, 3.0, 3.5, 4.0, 4.5)
    P, R, nup = bamg_ls_interpolation(agg, TV, Lp, tvw; c = best2c, gcap = gcap)
    LpH = galerkin_coarse_operator(Lp, P)
    f = mu2lvl(P, R, LpH); s = statcheck(P, R, LpH)
    global best2 = min(best2, f)
    @printf("  gcap=%.1f  n_upgraded=%4d  factor=%.4f  stationarity=%.2e\n", gcap, nup, f, s)
end

@printf("\nBEST variant(1) caliber-2 factor = %.4f\n", best1)
@printf("BEST variant(2) caliber-c LS factor = %.4f  (c=%d, stat=%.2e)\n", best2, best2c, best2stat)
@printf("OVERALL BEST factor = %.4f\n", min(best1, best2))

open(joinpath(@__DIR__, "geo_bamg_ls_caliber2_results.txt"), "w") do io
    @printf(io, "BEST_V1_caliber2=%.4f\nBEST_V2_caliberc=%.4f\nbest2c=%d\nbest2stat=%.3e\nOVERALL_BEST=%.4f\n",
            best1, best2, best2c, best2stat, min(best1, best2))
end
flush(stdout)
