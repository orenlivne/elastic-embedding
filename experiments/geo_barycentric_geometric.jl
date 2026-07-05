# Barycentric geometric interpolation on the INTRINSIC manifold geometry.
# Geometry Y_geo = bottom Laplacian eigenvectors (rows 2:1+g) that unroll the swiss roll.
# For each non-seed fine node i: find g+1 nearest seeds forming a simplex around geom[:,i];
# solve the (g+1)x(g+1) affine system [Yseeds; ones'] * bary = [geom_i; 1] for exact barycentric
# coords (sum=1). If any bary < -0.05 or > 1.05 (outside simplex) or singular, retry with the next
# nearest seed set; else fall back to caliber-1. Clamp tiny negatives to 0, renormalize.
# Seeds get caliber-1 (P[seed,agg]=1); R = seed injection (R.P = I). LpH = Galerkin.
# Sweep g in {2,3,4}. Reference: caliber-1=0.856, prior-geometric=0.558, ideal(mock)=0.066.
#
# Run: cd /Users/oren/code/elastic-embedding && julia --project=. experiments/geo_barycentric_geometric.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using LAMG, Random, Printf, LinearAlgebra, SparseArrays, Statistics

# ---- problem setup (exact protocol) ----
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

# ---- BUILD P (n x nc), R (nc x n), LpH via barycentric geometric interpolation ----
# geom: g x N intrinsic manifold coordinates (bottom Laplacian eigenvectors, skip constant mode).
function build_barycentric(agg, geom; nc, ntry=6, lo=-0.05, hi=1.05)
    g, n = size(geom)
    # seed = first member per aggregate
    seed = zeros(Int, nc)
    @inbounds for i in 1:n; a = agg[i]; seed[a] == 0 && (seed[a] = i); end
    isseed = falses(n); for a in 1:nc; isseed[seed[a]] = true; end
    Yseed = geom[:, seed]                      # g x nc seed positions in manifold coords
    Ip = Int[]; Jp = Int[]; Vp = Float64[]
    nfallback = 0; dist = zeros(nc)
    ncand = min(g + 1 + ntry, nc)              # candidate pool of nearest seeds
    @inbounds for i in 1:n
        if isseed[i]
            push!(Ip, i); push!(Jp, agg[i]); push!(Vp, 1.0); continue
        end
        yi = @view geom[:, i]
        for J in 1:nc
            s = 0.0; for k in 1:g; d = yi[k] - Yseed[k, J]; s += d*d; end; dist[J] = s
        end
        nn = partialsortperm(dist, 1:ncand)    # nearest candidate seeds (ascending)
        got = false
        # try successive (g+1)-subsets: greedy nearest, then swap in farther candidates
        # base subset = first g+1 nearest; retries replace the LAST (farthest) member.
        for r in 0:(ncand - (g+1))
            sel = Vector{Int}(undef, g+1)
            for j in 1:g; sel[j] = nn[j]; end
            sel[g+1] = nn[g+1+r]
            Ys = Yseed[:, sel]                  # g x (g+1)
            A = [Ys; ones(1, g+1)]              # (g+1) x (g+1) affine matrix
            b = vcat(collect(yi), 1.0)
            bary = try
                A \ b
            catch
                continue
            end
            (any(!isfinite, bary)) && continue
            if all(x -> x >= lo && x <= hi, bary)
                # clamp tiny negatives to 0, renormalize to sum 1
                for j in 1:(g+1); bary[j] < 0 && (bary[j] = 0.0); end
                sb = sum(bary); sb <= 1e-300 && continue
                bary ./= sb
                for j in 1:(g+1)
                    bj = bary[j]
                    if abs(bj) > 1e-12
                        push!(Ip, i); push!(Jp, sel[j]); push!(Vp, bj)
                    end
                end
                got = true
                break
            end
        end
        if !got
            nfallback += 1
            push!(Ip, i); push!(Jp, agg[i]); push!(Vp, 1.0)
        end
    end
    P = sparse(Ip, Jp, Vp, n, nc)
    R = sparse(collect(1:nc), seed, ones(nc), nc, n)
    return P, R, nfallback
end

# ---- two-level factor measurement (protocol boilerplate) ----
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

function sweep()
    best = (factor=Inf, g=0, stat=NaN, fb=0, hi=0.0, ntry=0)
    # Primary sweep: g in {2,3,4} at default simplex tolerance.
    println("=== primary g sweep (hi=1.05, ntry=6) ===")
    for g in (2, 3, 4)
        geom = Matrix(F.vectors[:, 2:1+g]')        # g x N
        P, R, nfb = build_barycentric(agg, geom; nc=nc)
        LpH = galerkin_coarse_operator(Lp, P)
        Xc = copy(Xstar); ee_two_level_P!(Xc, Lp, LpH, P, R, mass, mu)
        stat = norm(ee_A(Xc, Lp, mu, ones(N))) / max(norm(ee_A(Xstar, Lp, mu, ones(N))), 1e-30)
        f = mu2lvl(P, R, LpH)
        @printf("g=%d  factor=%.4f  stationarity=%.2e  fallback=%d/%d (%.1f%%)\n",
                g, f, stat, nfb, N, 100nfb/N)
        f < best.factor && (best = (factor=f, g=g, stat=stat, fb=nfb, hi=1.05, ntry=6))
    end
    # Secondary sweep at the winning intrinsic dimension g=2: relax the simplex tolerance
    # (larger hi admits mild extrapolation -> fewer caliber-1 fallbacks) and widen the
    # candidate seed pool (ntry) so a valid enclosing simplex is more often found.
    println("\n=== secondary tuning at g=2 (simplex tol `hi` and pool `ntry`) ===")
    geom = Matrix(F.vectors[:, 2:3]')
    for hi in (1.05, 1.2, 1.5, 2.0), ntry in (6, 12)
        P, R, nfb = build_barycentric(agg, geom; nc=nc, hi=hi, ntry=ntry, lo=1.0 - hi)
        LpH = galerkin_coarse_operator(Lp, P)
        Xc = copy(Xstar); ee_two_level_P!(Xc, Lp, LpH, P, R, mass, mu)
        stat = norm(ee_A(Xc, Lp, mu, ones(N))) / max(norm(ee_A(Xstar, Lp, mu, ones(N))), 1e-30)
        f = mu2lvl(P, R, LpH)
        @printf("g=2 hi=%.2f ntry=%2d  factor=%.4f  stationarity=%.2e  fallback=%d/%d (%.1f%%)\n",
                hi, ntry, f, stat, nfb, N, 100nfb/N)
        f < best.factor && (best = (factor=f, g=2, stat=stat, fb=nfb, hi=hi, ntry=ntry))
    end
    @printf("\nBEST: g=%d hi=%.2f ntry=%d  factor=%.4f  stationarity=%.2e  fallback=%d/%d (%.1f%%)\n",
            best.g, best.hi, best.ntry, best.factor, best.stat, best.fb, N, 100best.fb/N)
    @printf("Reference: caliber-1=0.856, prior-geometric=0.558, ideal(mock)=0.066\n")
    best
end
sweep()
