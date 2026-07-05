# Two-level FAS for genuine EE. 1-2 cycle, iterate recombination (window 3) on by default.
# Tests: (1) stationarity, (2) acceleration on/off, (3) size independence, (4) SLOWEST-MODE analysis
# (is the slow error smooth ⇒ interpolation problem, or oscillatory ⇒ relaxation problem?).
# Connected fixed-degree 4-block SBM (inter-degree ≈3 so ν₂>0 at all N). Run: julia experiments/two_level.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using Random, Printf, LinearAlgebra, SparseArrays, Statistics

function build_problem(s; rng = MersenneTwister(1))
    n, edges, w, _ = sbm_graph(fill(s, 4), 10.0 / s, 1.0 / s; rng = rng)   # intra≈10, inter≈3
    B, w = incidence(n, edges, w); Lp = weighted_laplacian(B, w)
    ν2 = eigen(Symmetric(Matrix(Lp))).values[2]
    ν2 < 1e-6 && error("graph disconnected at s=$s (ν₂=$ν2)")
    agg, nc = affinity_aggregation(Lp; ntv = 6, maxsize = 4, θ = 0.15, rng = MersenneTwister(5))
    mass = Float64[count(==(I), agg) for I in 1:nc]
    P = sparse(1:n, agg, ones(n), n, nc)
    (n = n, m = length(edges), Lp = Lp, μstar = ν2 / n, agg = agg, nc = nc, mass = mass, LpH = Matrix(P' * Lp * P))
end

function measure_factor(X0, pr, μ; cycles = 18, recombine = true, window = 3)
    X = copy(X0); Xs = Matrix{Float64}[]; Rs = Matrix{Float64}[]; facs = Float64[]
    for _ in 1:cycles
        r0 = norm(ee_A(X, pr.Lp, μ, ones(pr.n)))
        ee_two_level!(X, pr.Lp, pr.LpH, pr.agg, pr.nc, pr.mass, μ)     # 1-2 cycle (module default)
        R = ee_A(X, pr.Lp, μ, ones(pr.n))
        if recombine
            push!(Xs, copy(X)); push!(Rs, copy(R))
            length(Xs) > window && (popfirst!(Xs); popfirst!(Rs))
            if length(Xs) ≥ 2
                Xa = ee_diis(Xs, Rs); Ra = ee_A(Xa, pr.Lp, μ, ones(pr.n))
                if norm(Ra) < norm(R); X = Xa; R = Ra; Xs[end] = copy(X); Rs[end] = copy(R); end
            end
        end
        rn = norm(R); (!isfinite(rn) || rn > 1e8) && return Inf
        push!(facs, rn / max(r0, 1e-300)); rn < 1e-12 && break
    end
    fin = filter(x -> isfinite(x) && x > 0, facs); isempty(fin) && return Inf
    exp(mean(log.(fin[max(1, end-4):end])))
end

equilibrium(pr, μ, d) = ee_minimize(1e-2 .* laplacian_eigenmaps(Matrix(pr.Lp), d), pr.Lp, μ; iters = 1500, gtol = 1e-11)[1]

# deflate the embedding symmetries (translation for all d; rotation for d=2) — exact zero modes
function deflate_sym!(e, Xstar)
    d = size(e, 1)
    for a in 1:d; e[a, :] .-= mean(@view e[a, :]); end          # translation
    if d == 2
        g = vcat((-Xstar[2, :])', (Xstar[1, :])'); g ./= (norm(g) + 1e-300)  # rotation generator
        e .-= dot(vec(g), vec(e)) .* g
    end
    e
end

# slowest NON-symmetry mode of the UNACCELERATED cycle (power iteration), + its smoothness character
function slow_mode(pr, μ, d)
    Xstar = equilibrium(pr, μ, d)
    e = 0.01 .* randn(MersenneTwister(2), d, pr.n); deflate_sym!(e, Xstar)
    for _ in 1:120
        X = Xstar .+ e; ee_two_level!(X, pr.Lp, pr.LpH, pr.agg, pr.nc, pr.mass, μ)
        e = X .- Xstar; deflate_sym!(e, Xstar); ne = norm(e); ne < 1e-13 && break; e ./= ne
    end
    X = Xstar .+ 1e-4 .* e; ee_two_level!(X, pr.Lp, pr.LpH, pr.agg, pr.nc, pr.mass, μ)
    ef = X .- Xstar; deflate_sym!(ef, Xstar); fac = norm(ef) / 1e-4
    F = eigen(Symmetric(Matrix(pr.Lp))); VK = F.vectors[:, 1:min(20, pr.n)]
    lowfrac = 0.0; rq = 0.0
    for a in 1:d
        ea = e[a, :]; na = dot(ea, ea) + 1e-300
        lowfrac += dot(VK' * ea, VK' * ea) / na; rq += dot(ea, pr.Lp * ea) / na
    end
    (fac, lowfrac / d, rq / d, F.values[2], F.values[end])
end

# ---------- setup ----------
pr = build_problem(150)
@printf("GRAPH: 4-block SBM N=%d, m=%d, avg deg≈%.1f, aggregates=%d (%.2f), μ*=%.4g\n\n",
        pr.n, pr.m, 2pr.m / pr.n, pr.nc, pr.nc / pr.n, pr.μstar)

# (1) stationarity
μ = 2pr.μstar; Xstar = equilibrium(pr, μ, 1); rb = norm(ee_A(Xstar, pr.Lp, μ, ones(pr.n)))
Xc = copy(Xstar); ee_two_level!(Xc, pr.Lp, pr.LpH, pr.agg, pr.nc, pr.mass, μ)
@printf("(1) STATIONARITY (d=1,r=2): ‖A(X*)‖ %.2e → %.2e ; ‖ΔX‖/‖X*‖=%.1e  [%s]\n\n",
        rb, norm(ee_A(Xc, pr.Lp, μ, ones(pr.n))), norm(Xc - Xstar) / norm(Xstar),
        norm(ee_A(Xc, pr.Lp, μ, ones(pr.n))) ≤ 5rb ? "OK" : "BAD")

# (2) acceleration
println("(2) ACCELERATION (r=1.5, 1-2 cycle):  d   no-recomb   recomb(win3)")
for d in (1, 2)
    local μ2 = 1.5pr.μstar; local Xp = equilibrium(pr, μ2, d) .+ 0.05 .* randn(MersenneTwister(3), d, pr.n)
    @printf("                                      %d   %8.3f   %8.3f\n", d,
            measure_factor(Xp, pr, μ2; recombine = false), measure_factor(Xp, pr, μ2; recombine = true))
end

# (3) size independence
println("\n(3) SIZE INDEPENDENCE (d=2, r=2, recomb):  N      aggs    factor")
for s in (75, 150, 225, 300)
    local p = build_problem(s); local μ3 = 2p.μstar
    local Xp = equilibrium(p, μ3, 2) .+ 0.05 .* randn(MersenneTwister(3), 2, p.n)
    @printf("                                           %-6d %6d  %.3f\n", p.n, p.nc, measure_factor(Xp, p, μ3))
end

# (4) slowest-mode analysis
println("\n(4) SLOWEST-MODE ANALYSIS (r=2):")
println("d   slow-factor   energy-frac in bottom-20 modes   Rayleigh q   [ν₂ .. ν_max]   diagnosis")
for d in (1, 2)
    local μ4 = 2pr.μstar; fac, lf, rq, ν2, νmax = slow_mode(pr, μ4, d)
    diag = lf > 0.6 ? "SMOOTH ⇒ interpolation" : "OSCILLATORY ⇒ relaxation"
    @printf("%d   %8.3f   %26.2f   %10.3g   [%.3g .. %.3g]   %s\n", d, fac, lf, rq, ν2, νmax, diag)
end
