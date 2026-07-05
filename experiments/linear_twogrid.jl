# DECISIVE: is the raw-0.99 two-grid on geometric graphs a fixable COARSE-SOLVE bug, or fundamental?
# Build the LINEARIZED operator J = L+ - mu*L-(X*) (symmetric) and run a standard linear two-grid with
# the SAME frozen-Gaussian lagged-GS smoother but an EXACT (dense pinv) coarse solve of JH = P'JP.
# If factor -> ~0.066 (the mock ideal), the nonlinear BB coarse solve was the culprit (FIXABLE).
# If factor stays ~0.9, the coarse OPERATOR/INTERPOLATION is fundamentally deficient (not fixable by
# a better solver). Test caliber-1 and the bamg_ls geometric P. Run: julia --project=. experiments/linear_twogrid.jl

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

J = Matrix(Lp) .- μ .* Lm            # linearized (chord) operator, symmetric, kills the constant

# linear two-grid factor with EXACT coarse solve (pinv handles the constant-null + indefiniteness)
function tg_exact(P; ν1 = 1, ν2 = 2, sweeps = 60, warmup = 25)
    JH = Matrix(P' * (J * P)); JHi = pinv(JH)
    smooth!(e) = (rhs = μ .* (Lm * e); gauss_seidel!(e, Lp; b = rhs, sweeps = 1); e)
    e = randn(MersenneTwister(7), N); e .-= mean(e); e ./= norm(e); rs = Float64[]
    for _ in 1:sweeps
        for _ in 1:ν1; smooth!(e); end
        e .+= P * (JHi * (-(P' * (J * e))))        # exact coarse correction
        for _ in 1:ν2; smooth!(e); end
        f = norm(e); push!(rs, f); e ./= max(f, 1e-300)
    end
    exp(mean(log.(rs[warmup+1:end] .+ 1e-300)))
end

# bamg_ls interpolation (the workflow's best interpolation) for comparison
function bamg_ls(agg, TV, A, tvw; c = 5, gcap = 3.0, ridge = 1e-8)
    n = length(agg); ncc = maximum(agg); rows = rowvals(A); vals = nonzeros(A); W = Diagonal(sqrt.(tvw))
    seed = zeros(Int, ncc); for i in 1:n; a = agg[i]; seed[a] == 0 && (seed[a] = i); end
    isseed = falses(n); for a in 1:ncc; isseed[seed[a]] = true; end
    Ip = Int[]; Jp = Int[]; Vp = Float64[]; coup = Dict{Int,Float64}()
    for i in 1:n
        if isseed[i]; push!(Ip, i); push!(Jp, agg[i]); push!(Vp, 1.0); continue; end
        empty!(coup); for k in nzrange(A, i); r = rows[k]; r == i && continue; coup[agg[r]] = get(coup, agg[r], 0.0) + abs(vals[k]); end
        coup[agg[i]] = get(coup, agg[i], 0.0) + 1e30; cand = collect(keys(coup))
        length(cand) > c && (cand = cand[partialsortperm([coup[a] for a in cand], 1:c; rev = true)]); m = length(cand)
        m == 1 && (push!(Ip, i); push!(Jp, cand[1]); push!(Vp, 1.0); continue)
        Ss = [seed[a] for a in cand]; M = W * TV[Ss, :]'; xi = W * (@view TV[i, :]); G = M' * M
        G .+= (ridge * (tr(G) / m + 1e-300)) .* Matrix(I, m, m)
        p = try ([2G ones(m); ones(m)' 0.0] \ [2 .* (M' * xi); 1.0])[1:m] catch; fill(1.0 / m, m) end
        if any(!isfinite, p) || maximum(abs, p) > gcap; push!(Ip, i); push!(Jp, agg[i]); push!(Vp, 1.0)
        else; for (idx, a) in enumerate(cand); abs(p[idx]) > 1e-8 && (push!(Ip, i); push!(Jp, a); push!(Vp, p[idx])); end; end
    end
    sparse(Ip, Jp, Vp, n, ncc)
end

tvw = [norm(TV[:, k])^2 / max(norm(Lp * TV[:, k])^2, 1e-30) for k in 1:8]
P1, _, _ = piecewise_constant_interpolation(agg)
Pg = bamg_ls(agg, TV, Lp, tvw; c = 5)
laggedGS(e) = gauss_seidel!(e, Lp; b = μ .* (Lm * e), sweeps = 1)
μmock = cr_shrinkage(N, laggedGS, agg, nc; ν = 2, sweeps = 60, warmup = 20, rng = MersenneTwister(11))

@printf("LINEARIZED two-grid with EXACT coarse solve (swiss roll):\n")
@printf("  caliber-1:   %.3f\n", tg_exact(P1))
@printf("  bamg_ls c=5: %.3f\n", tg_exact(Pg))
@printf("  mock ideal:  %.3f\n\n", μmock)

# confirm indefiniteness of J = L+ - mu*L-
ej = eigen(Symmetric(J)).values
@printf("J spectrum: #negative=%d, smallest 4 |eig| = %.2e %.2e %.2e %.2e (INDEFINITE/near-singular ⇒ Galerkin correction unstable)\n\n",
        count(<(−1e-9), ej), sort(abs.(ej))[1], sort(abs.(ej))[2], sort(abs.(ej))[3], sort(abs.(ej))[4])

# DEFLATED coarse correction: project out the bottom-K eigenvectors of J each cycle (stable, like the mock)
function tg_deflated(P, K; ν1 = 1, ν2 = 2, sweeps = 60, warmup = 25)
    JH = Matrix(P' * (J * P)); JHi = pinv(JH)
    V = eigen(Symmetric(J)).vectors[:, 1:K]                      # bottom-K modes to deflate
    defl(e) = (e .-= V * (V' * e); e)
    smooth!(e) = (rhs = μ .* (Lm * e); gauss_seidel!(e, Lp; b = rhs, sweeps = 1); e)
    e = randn(MersenneTwister(7), N); defl(e); e ./= norm(e); rs = Float64[]
    for _ in 1:sweeps
        for _ in 1:ν1; smooth!(e); end
        e .+= P * (JHi * (-(P' * (J * e)))); defl(e)            # coarse correction + deflation
        for _ in 1:ν2; smooth!(e); end
        defl(e); f = norm(e); push!(rs, f); e ./= max(f, 1e-300)
    end
    exp(mean(log.(rs[warmup+1:end] .+ 1e-300)))
end
@printf("DEFLATED coarse correction (deflate bottom-K near-null modes of J), caliber-1 P:\n")
for K in (1, 2, 4, 8)
    @printf("  K=%d: %.3f\n", K, tg_deflated(P1, K))
end
@printf("\nIf deflating the few near-null modes recovers ~mock -> confirmed: it's INDEFINITE-operator coarse-\n")
@printf("correction instability; fix = deflation of the ~d embedding modes (not interpolation).\n")
