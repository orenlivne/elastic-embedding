# averaging_coarsevar_plus_sparsify:
# Attack two suspects behind the geometric two-level bottleneck:
#   (i)  COARSE-VARIABLE mismatch: the mock (0.066) uses aggregate AVERAGES as the coarse variable,
#        but a seed-injection R makes the coarse variable a single seed value. Use AVERAGING R = R0
#        from piecewise_constant_interpolation so the coarse variable is the aggregate average.
#   (ii) coarse-operator DENSIFICATION from a caliber>1 P: a smoothed-aggregation P spreads support,
#        Galerkin P'LpP densifies. Counter it with LAMG's sparsify_lump on the coarse operator.
#
# P = smoothed aggregation:  P = P0 - omega * (Dinv .* (Lp*P0)),  P0 = caliber-1 indicator, Dinv=1/diag(Lp)
# R in { R0 (averaging), Rseed (seed injection) }.   LpH = sparsify_lump(P'LpP, tol).
# Sweep omega and tol; report factor + stationarity for each combo. Best combo at the end.
#
# Reference factors: caliber-1 = 0.856, prior-geometric = 0.558, ideal(mock) = 0.066.
# Run: cd /Users/oren/code/elastic-embedding && julia --project=. experiments/geo_averaging_coarsevar_plus_sparsify.jl

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

# ---- caliber-1 pieces ----
P0, R0, Q0 = piecewise_constant_interpolation(agg)      # P0 indicator, R0 = averaging restriction
# seed-injection restriction (as in geometric_interpolation): R[I, seed_I]=1
seed = zeros(Int, nc); for i in 1:N; a = agg[i]; seed[a] == 0 && (seed[a] = i); end
Rseed = sparse(collect(1:nc), seed, ones(nc), nc, N)

Dinv = 1.0 ./ diag(Lp)
# smoothed-aggregation prolongation for a given omega
smoothed_P(omega) = P0 .- omega .* (Dinv .* (Lp * P0))

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

function stat_ratio(P, R, LpH)
    Xc = copy(Xstar); ee_two_level_P!(Xc, Lp, LpH, P, R, mass, mu)
    norm(ee_A(Xc, Lp, mu, ones(N))) / max(norm(ee_A(Xstar, Lp, mu, ones(N))), 1e-30)
end

omegas = (0.4, 0.55, 0.67, 0.8, 1.0)
tols   = (0.0, 0.01, 0.05, 0.1)

# Moore-Penrose left inverse of the SMOOTHED P: R = (PᵀP)⁻¹Pᵀ  ⇒  R·P = I exactly (FAS-safe).
# This is the "aggregate-average of the smoothed basis" coarse variable and fixes the RP≠I blowup.
function pinv_R(P)
    Pd = Matrix(P); G = Pd' * Pd
    sparse((G + 1e-10*I) \ (Pd'))
end

# baseline: caliber-1 with the two R's (no smoothing), tol=0
@printf("BASELINES (caliber-1 P0, tol=0):\n")
for (rname, R) in (("R0-avg", R0), ("Rseed", Rseed))
    LpH0 = galerkin_coarse_operator(Lp, sparse(P0))
    @printf("  P0 x %-6s  factor=%.4f  stat=%.2e  RP~I=%.2e\n", rname,
            mu2lvl(sparse(P0), R, LpH0), stat_ratio(sparse(P0), R, LpH0), norm(Matrix(R*P0) - I))
end

@printf("\nSMOOTHED-AGGREGATION SWEEP (factor / stat):\n")
best = let best = (factor=Inf, omega=NaN, tol=NaN, rname="", stat=NaN)
    for omega in omegas
        P = sparse(smoothed_P(omega))
        A0 = galerkin_coarse_operator(Lp, P)
        Rvariants = (("R0-avg", R0), ("Rseed", Rseed), ("Rpinv", pinv_R(P)))
        for (rname, R) in Rvariants
            rp = norm(Matrix(R*P) - I)
            for tol in tols
                LpH = LAMG.sparsify_lump(A0, tol)
                f = mu2lvl(P, R, LpH); s = stat_ratio(P, R, LpH)
                @printf("  omega=%.2f %-6s tol=%.2f | factor=%.4f stat=%.2e RP-I=%.2e nnz(LpH)=%d\n",
                        omega, rname, tol, f, s, rp, nnz(LpH))
                if isfinite(f) && f < best.factor && isfinite(s) && s < 5.0
                    best = (factor=f, omega=omega, tol=tol, rname=rname, stat=s)
                end
            end
        end
    end
    best
end

@printf("\nBEST (stat<5): factor=%.4f  omega=%.2f  tol=%.2f  R=%s  stat=%.2e\n",
        best.factor, best.omega, best.tol, best.rname, best.stat)
@printf("Reference: caliber-1=0.856  prior-geometric=0.558  ideal(mock)=0.066\n")
