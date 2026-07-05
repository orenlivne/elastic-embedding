# Smoothed Aggregation (Vanek-Mandel-Brezina) on L+ for the two-level EE FAS cycle on a
# swiss-roll kNN graph. Tentative caliber-1 prolongator P0 is smoothed by one weighted-Jacobi
# sweep on L+ : P = P0 - omega .* (Dinv .* (Lp*P0)). Sweep omega x {R0, P'} restriction.
# Reference: caliber-1=0.856, prior-geometric=0.558, ideal(mock)=0.066.  Beat 0.558; target ~0.1.
#
# Run: cd /Users/oren/code/elastic-embedding && julia --project=. experiments/geo_smoothed_agg_Lplus.jl

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

function mu2lvl(P, R, LpH, massc)
    X = Xstar .+ 0.05 .* randn(MersenneTwister(3), 1, N); Xs=Matrix{Float64}[]; Rs=Matrix{Float64}[]; fs=Float64[]
    for _ in 1:20
        r0 = norm(ee_A(X, Lp, mu, ones(N))); ee_two_level_P!(X, Lp, LpH, P, R, massc, mu); Rr = ee_A(X, Lp, mu, ones(N))
        push!(Xs, copy(X)); push!(Rs, copy(Rr)); length(Xs) > 5 && (popfirst!(Xs); popfirst!(Rs))
        length(Xs) >= 2 && (Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, Lp, mu, ones(N)); norm(Ra) < norm(Rr) && (X=Xa; Rr=Ra; Xs[end]=copy(X); Rs[end]=copy(Rr)))
        rn = norm(Rr); (!isfinite(rn) || rn > 1e8) && break; push!(fs, rn/max(r0,1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, fs); isempty(fin) ? Inf : exp(mean(log.(fin[max(1,end-4):end])))
end

# ==== Smoothed aggregation on L+ ====
P0, R0, _ = piecewise_constant_interpolation(agg)
Dinv = 1.0 ./ diag(Lp)                                 # weighted-Jacobi diagonal inverse

# spectral radius of Dinv.*Lp (Jacobi iteration matrix scaling), for reference
DL = Diagonal(Dinv) * Lp
rho = eigmax(Symmetric(Matrix(DL)))
@printf("spectral radius of Dinv.*Lp = %.4f (omega should be < 2/rho = %.4f)\n", rho, 2/rho)

function smoothed_P(omega; nu::Int=1)
    P = P0
    for _ in 1:nu
        P = P .- omega .* (Dinv .* (Lp * P))           # weighted-Jacobi smoothing of columns
    end
    SparseMatrixCSC(P)
end

# ---- Seed-anchored SA: smooth interior rows but RESTORE seed rows to identity so that
#      seed-injection R gives R*P = I EXACTLY (FAS-stationary, as in geometric_interpolation). ----
seed = zeros(Int, nc); for i in 1:N; a = agg[i]; seed[a] == 0 && (seed[a] = i); end
isseed = falses(N); for a in 1:nc; isseed[seed[a]] = true; end
Rinj = sparse(collect(1:nc), seed, ones(nc), nc, N)    # seed injection restriction (R*P0 = I)

function smoothed_P_anchored(omega; nu::Int=1)
    P = Matrix(smoothed_P(omega; nu=nu))
    for a in 1:nc                                      # restore seed rows to unit e_a
        P[seed[a], :] .= 0.0; P[seed[a], a] = 1.0
    end
    SparseMatrixCSC(sparse(P))
end

function sweep()
    omegas = [0.3, 0.5, 0.67, 0.85, 1.0]
    best_factor = Inf; best_desc = ""
    # Plain textbook SA (as assigned): smooth all rows; R=R0 vs R=P'
    @printf("\n--- plain smoothed aggregation (all rows smoothed) ---\n")
    @printf("%-7s %-14s %-14s\n", "omega", "R=R0", "R=P'")
    for omega in omegas
        P = smoothed_P(omega)
        LpH = galerkin_coarse_operator(Lp, P)
        massP = vec(sum(P, dims=1))                                    # consistent coarse repulsion mass
        f_R0 = try mu2lvl(P, R0, LpH, massP) catch e; Inf end          # caliber-1 averaging restriction
        Rt = SparseMatrixCSC(sparse(P'))
        f_Pt = try mu2lvl(P, Rt, LpH, massP) catch e; Inf end          # Galerkin restriction R = P'
        @printf("%-7.2f %-14.4f %-14.4f\n", omega, f_R0, f_Pt)
        if f_R0 < best_factor; best_factor = f_R0; best_desc = @sprintf("plain omega=%.2f R=R0", omega); end
        if f_Pt < best_factor; best_factor = f_Pt; best_desc = @sprintf("plain omega=%.2f R=P'", omega); end
    end
    # Seed-anchored SA: seed rows = identity, R = seed injection => R*P = I exactly (FAS-safe)
    @printf("\n--- seed-anchored smoothed aggregation (R*P=I via seed injection) ---\n")
    @printf("%-7s %-14s\n", "omega", "R=Rinj")
    for omega in omegas
        P = smoothed_P_anchored(omega)
        LpH = galerkin_coarse_operator(Lp, P)
        massP = vec(sum(P, dims=1))
        f = try mu2lvl(P, Rinj, LpH, massP) catch e; Inf end
        @printf("%-7.2f %-14.4f\n", omega, f)
        if f < best_factor; best_factor = f; best_desc = @sprintf("anchored omega=%.2f R=Rinj", omega); end
    end
    # Multi-sweep prolongator smoothing (nu weighted-Jacobi sweeps): plain R=R0 and anchored R=Rinj
    @printf("\n--- multi-sweep smoothing: nu weighted-Jacobi sweeps ---\n")
    @printf("%-7s %-6s %-14s %-14s\n", "omega", "nu", "plain R=R0", "anchored Rinj")
    for omega in [0.5, 0.67, 0.85], nu in [2, 3, 4]
        Pp = smoothed_P(omega; nu=nu); LpHp = galerkin_coarse_operator(Lp, Pp); mp = vec(sum(Pp,dims=1))
        fp = try mu2lvl(Pp, R0, LpHp, mp) catch e; Inf end
        Pa = smoothed_P_anchored(omega; nu=nu); LpHa = galerkin_coarse_operator(Lp, Pa); ma = vec(sum(Pa,dims=1))
        fa = try mu2lvl(Pa, Rinj, LpHa, ma) catch e; Inf end
        @printf("%-7.2f %-6d %-14.4f %-14.4f\n", omega, nu, fp, fa)
        if fp < best_factor; best_factor = fp; best_desc = @sprintf("plain omega=%.2f nu=%d R=R0", omega, nu); end
        if fa < best_factor; best_factor = fa; best_desc = @sprintf("anchored omega=%.2f nu=%d R=Rinj", omega, nu); end
    end
    best_factor, best_desc
end
best_factor, best_desc = sweep()

# Stationarity check for the BEST configuration (reparse omega + rebuild P,R)
bo = parse(Float64, match(r"omega=([0-9.]+)", best_desc).captures[1])
bnu = (m = match(r"nu=([0-9]+)", best_desc)) === nothing ? 1 : parse(Int, m.captures[1])
anchored = occursin("anchored", best_desc)
Pb = anchored ? smoothed_P_anchored(bo; nu=bnu) : smoothed_P(bo; nu=bnu)
LpHb = galerkin_coarse_operator(Lp, Pb)
Rb = anchored ? Rinj : (occursin("R=R0", best_desc) ? R0 : SparseMatrixCSC(sparse(Pb')))
massb = vec(sum(Pb, dims=1))
Xc = copy(Xstar); ee_two_level_P!(Xc, Lp, LpHb, Pb, Rb, massb, mu)
stat = norm(ee_A(Xc, Lp, mu, ones(N))) / max(norm(ee_A(Xstar, Lp, mu, ones(N))), 1e-30)

# R.P approx I diagnostic for best
RP = Rb * Pb
rp_err = norm(Matrix(RP) - I(nc)) / sqrt(nc)

@printf("\nBEST: %s  factor=%.4f\n", best_desc, best_factor)
@printf("stationarity_ratio=%.2e   ||R*P - I||/sqrt(nc)=%.3e\n", stat, rp_err)
@printf("references: caliber-1=0.856  prior-geometric=0.558  ideal=0.066\n")
