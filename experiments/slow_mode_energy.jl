# Brandt diagnostic: WHY is the smooth slow mode slow? Compute its LOCAL energy — is the energy
# carried by the stiffness L⁺ or the nonlinear mass μL⁻? Is it localized? And which aggregates TRAP
# energy that caliber-1 (constant-per-aggregate) cannot see — and do those coincide with high mass?
# d=1 for a clean scalar mode. Run: julia --project=. experiments/slow_mode_energy.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

rng = MersenneTwister(1)
s = 150; n, edges, w, _ = sbm_graph(fill(s, 4), 10.0 / s, 1.0 / s; rng = rng)
B, w = incidence(n, edges, w); Lp = weighted_laplacian(B, w)
μstar = eigen(Symmetric(Matrix(Lp))).values[2] / n
lag = aggregate(Lp; rng = MersenneTwister(5)); agg = lag.aggregate; nc = lag.n_coarse
mass = Float64[count(==(I), agg) for I in 1:nc]
P = sparse(1:n, agg, ones(n), n, nc); LpH = Matrix(P' * Lp * P)
μ = 2μstar
@printf("N=%d, aggregates=%d, μ=%.4g\n", n, nc, μ)

Xstar = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), 1), Lp, μ; iters = 1500, gtol = 1e-11)[1]
Lm = build_Lminus_dense(Xstar; σ2 = 1.0)                 # frozen repulsive Laplacian at X*
A = Matrix(Lp) .- μ .* Lm                                # chord operator

# extract slowest mode (unaccelerated cycle, translation-deflated)
e = 0.01 .* randn(MersenneTwister(2), 1, n); e .-= mean(e)
for _ in 1:150
    global e
    X = Xstar .+ e; ee_two_level!(X, Lp, LpH, agg, nc, mass, μ)
    e = X .- Xstar; e .-= mean(e); ne = norm(e); ne < 1e-13 && break; e ./= ne
end
ev = vec(e)
Xv = vec(Xstar)

# global energies (chord): stiffness eᵀL⁺e  vs  mass μ eᵀL⁻e
S = ev' * (Lp * ev); M = μ * (ev' * (Lm * ev))
@printf("\nGLOBAL ENERGY of slow mode:  stiffness S=%.4g   mass M=%.4g   M/S=%.2f  → %s\n",
        S, M, M / S, M > S ? "NONLINEAR (mass) dominant" : "stiffness dominant")

# per-node local energy (edge-split), stiffness and mass
sloc = zeros(n); mloc = zeros(n)
for (k, (i, j)) in enumerate(edges)
    de = (ev[i] - ev[j])^2; sloc[i] += w[k] * de / 2; sloc[j] += w[k] * de / 2
end
@inbounds for i in 1:n-1, j in i+1:n
    wt = exp(-(Xv[i] - Xv[j])^2); me = μ * wt * (ev[i] - ev[j])^2
    mloc[i] += me / 2; mloc[j] += me / 2
end

# caliber-1 trapped energy: residual rc = e − aggregate-mean(e) (what constant-per-aggregate misses)
ē = zeros(nc); cnt = zeros(Int, nc)
for i in 1:n; ē[agg[i]] += ev[i]; cnt[agg[i]] += 1; end
ē ./= cnt; rc = ev .- ē[agg]
Src = rc' * (Lp * rc); Mrc = μ * (rc' * (Lm * rc))
@printf("CALIBER-1 TRAPPED FRACTION (within-aggregate residual energy / total):  stiffness %.1f%%   mass %.1f%%\n",
        100Src / S, 100Mrc / M)

# per-aggregate trapped stiffness energy (intra-aggregate edges) and per-aggregate mass energy
trapS = zeros(nc)
for (k, (i, j)) in enumerate(edges)
    agg[i] == agg[j] && (trapS[agg[i]] += w[k] * (ev[i] - ev[j])^2)
end
aggM = zeros(nc); for i in 1:n; aggM[agg[i]] += mloc[i]; end   # mass energy summed per aggregate
aggS = zeros(nc); for i in 1:n; aggS[agg[i]] += sloc[i]; end
@printf("\nTrapped stiffness energy concentration: top-5 aggregates hold %.0f%% of trapped energy (of %d aggs)\n",
        100 * sum(sort(trapS; rev = true)[1:5]) / (sum(trapS) + 1e-300), nc)
@printf("Correlation(per-agg trapped-stiffness, per-agg mass-energy) = %.2f\n", cor(trapS, aggM))
@printf("Correlation(per-agg total-energy, per-agg size)            = %.2f\n\n", cor(aggS .+ aggM, Float64.(cnt)))

println("Top-6 aggregates by trapped stiffness energy:  agg  size  trapS   aggMass  aggStiff")
for I in sortperm(trapS; rev = true)[1:6]
    @printf("                                               %-4d %-4d  %.3g   %.3g   %.3g\n",
            I, cnt[I], trapS[I], aggM[I], aggS[I])
end

# FIX TEST: aggregate on the FULL operator A=L⁺−μL⁻ low modes (mass-aware test vectors via X_ext)
println("\nFIX TEST — does MASS-AWARE aggregation (X_ext = low modes of A) trap less of the slow mode?")
Fa = eigen(Symmetric(A))
for K in (2, 4, 8)
    lag2 = aggregate(Lp; X_ext = Fa.vectors[:, 1:K], rng = MersenneTwister(5))
    a2 = lag2.aggregate; nc2 = lag2.n_coarse
    ē2 = zeros(nc2); c2 = zeros(Int, nc2); for i in 1:n; ē2[a2[i]] += ev[i]; c2[a2[i]] += 1; end; ē2 ./= c2
    rc2 = ev .- ē2[a2]; trapped2 = 100 * (rc2' * (Lp * rc2)) / S
    @printf("  A-modes K=%d: aggs=%-3d  trapped stiffness of slow mode = %.1f%%  (L⁺-only agg was 31.2%%)\n",
            K, nc2, trapped2)
end

# CLOSE THE LOOP: actual two-level FACTOR with L⁺-only vs mass-aware aggregation
mk(a, k) = (m = Float64[count(==(I), a) for I in 1:k]; Pm = sparse(1:n, a, ones(n), n, k); (agg = a, nc = k, mass = m, LpH = Matrix(Pm' * Lp * Pm)))
function factor(Ag, d; cyc = 18, win = 3)
    X = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(Lp), d), Lp, μ; iters = 1500, gtol = 1e-11)[1] .+ 0.05 .* randn(MersenneTwister(3), d, n)
    Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]; fs = Float64[]
    for _ in 1:cyc
        r0 = norm(ee_A(X, Lp, μ, ones(n))); ee_two_level!(X, Lp, Ag.LpH, Ag.agg, Ag.nc, Ag.mass, μ); R = ee_A(X, Lp, μ, ones(n))
        push!(Xs, copy(X)); push!(Rs, copy(R)); length(Xs) > win && (popfirst!(Xs); popfirst!(Rs))
        length(Xs) ≥ 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, μ, ones(n)); norm(Ra) < norm(R) && (X = Xa; R = Ra; Xs[end] = copy(X); Rs[end] = copy(R)))
        rn = norm(R); (!isfinite(rn) || rn > 1e8) && return Inf; push!(fs, rn / max(r0, 1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, fs); isempty(fin) ? Inf : exp(mean(log.(fin[max(1, end-4):end])))
end
# practical mass-aware test vectors: random vectors relaxed on the FULL operator by lagged-GS (O(m) each)
function lagged_tvs(K, ν)
    TV = zeros(n, K)
    for k in 1:K
        v = randn(MersenneTwister(40 + k), n); v .-= mean(v)
        for _ in 1:ν; gauss_seidel!(v, Lp; b = μ .* (Lm * v), sweeps = 1); end
        v .-= mean(v); TV[:, k] = v ./ norm(v)
    end
    TV
end
lagG = aggregate(Lp; X_ext = lagged_tvs(4, 4), rng = MersenneTwister(5))
lagA = aggregate(Lp; X_ext = Fa.vectors[:, 1:4], rng = MersenneTwister(5))
println("\nACTUAL TWO-LEVEL FACTOR (d=1, r=2, recomb win3):")
@printf("  L⁺-only aggregation:              %.3f\n", factor(mk(agg, nc), 1))
@printf("  mass-aware (A eigenvectors K=4):  %.3f\n", factor(mk(lagA.aggregate, lagA.n_coarse), 1))
@printf("  mass-aware (lagged-GS TVs, K=4):  %.3f   ← practical, O(m) test vectors\n", factor(mk(lagG.aggregate, lagG.n_coarse), 1))
