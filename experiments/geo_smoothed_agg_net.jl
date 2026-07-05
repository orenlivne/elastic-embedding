# Mass-aware SMOOTHED AGGREGATION with the NET operator.
# Tentative prolongator P0 = piecewise_constant (caliber-1). Smooth it with the dense net operator
# A = L+ - mu*L- (the EE Hessian at Xstar) so the prolongator aligns with the mass-shaped low modes
# (the slow ones on geometric/manifold graphs). k smoothing sweeps of damped Jacobi:
#     P <- P - omega * Dinv .* (A * P),   Dinv = 1 ./ max(diag(A), 0.1*max(diag(Lp)))
# rho(Dinv*A) measured = 1.48 (so omega up to ~1.35 stable; SA-optimal ~ 4/(3*rho) ~ 0.9).
# Restriction consistent with the smoothed P: Rls = (P'P)^{-1} P'  (so R*P == I, FAS-safe).
#   Compared against R0 (caliber-1 averaging) and Pt = P'.
# LpH = galerkin_coarse_operator(Lp, P). Note: on this harness even caliber-1/geometric give
#   stationarity_ratio ~ 60-70 (inherent to the BB coarse solve leaving a residual), so ratio ~1e2
#   is NORMAL here, not a P/R bug (verified: caliber-1 stat=68.7, geometric stat=66.0).
# Reference: caliber-1=0.856, prior-geo=0.558, mock=0.066.
# RESULT (measured): BEST factor = 0.8129 at k=1, omega=0.67, R=Rls (least-squares restriction).
#   Full k=1 sweep: omega 0.30->0.85 factors 0.844/0.881/0.813/0.886/0.881; R0~Rls, Pt worse.
#   k=2 (extra smoothing sweep) trended WORSE (0.869 at omega=0.30). So smoothed aggregation with the
#   dense NET operator does NOT beat prior-geo (0.558); it barely improves on caliber-1 (0.856).
#   WHY: one/two damped-Jacobi sweeps widen the prolongator support but do NOT reproduce affine
#   embedding modes exactly the way the affine-LS geometric interpolation does, which is what a smooth
#   1-D manifold needs. The net operator's near-null constant is preserved (good) but the smoothed
#   columns only approximately span the low modes, leaving the manifold staircase largely intact.
# Run: cd /Users/oren/code/elastic-embedding && julia --project=. experiments/geo_smoothed_agg_net.jl

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

# ==== BUILD P (n x nc), R (nc x n), LpH ====
P0, R0, _ = piecewise_constant_interpolation(agg)   # caliber-1 tentative prolongator
A = Matrix(Lp) .- mu .* Lm                            # dense net operator (n x n)
dLp = maximum(diag(Lp))
Dinv = 1.0 ./ max.(diag(A), 0.1 * dLp)               # guarded inverse diagonal

# k damped-Jacobi smoothing sweeps of the tentative prolongator with the net operator A.
smooth_P(omega, ksweeps) = begin
    Pd = Matrix(P0)
    for _ in 1:ksweeps
        Pd = Pd .- omega .* (Dinv .* (A * Pd))
    end
    Pd
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

best = Inf; bestcfg = ""
for ksweeps in (1, 2)
    for omega in (0.3, 0.5, 0.67, 0.85, 1.0)
        Pd = smooth_P(omega, ksweeps)
        P = sparse(Pd)
        LpH = galerkin_coarse_operator(Lp, P)
        Rls = sparse((Pd' * Pd) \ Pd')              # least-squares restriction, R*P == I
        for (rname, R) in (("R0", R0), ("Pt", sparse(P')), ("Rls", Rls))
            f = mu2lvl(P, R, LpH)
            Xc = copy(Xstar); ee_two_level_P!(Xc, Lp, LpH, P, R, mass, mu)
            stat = norm(ee_A(Xc, Lp, mu, ones(N))) / max(norm(ee_A(Xstar, Lp, mu, ones(N))), 1e-30)
            @printf("k=%d omega=%.2f R=%-3s factor=%.4f stationarity_ratio=%.2e\n", ksweeps, omega, rname, f, stat)
            if isfinite(f) && f < best; global best = f; global bestcfg = @sprintf("k=%d omega=%.2f R=%s", ksweeps, omega, rname); end
        end
    end
end
@printf("BEST factor=%.4f  cfg=%s\n", best, bestcfg)
