# Does the naive freeze-Gaussian / solve-Laplacian / repeat (SCF / lagged Picard) iteration slow down
# near the critical μ = λ/σ²?  The scheme (d=1 for a clean first bifurcation):
#
#     x_new = μ (L⁺)⁺ L⁻(x_old) x_old          # solve L⁺ x_new = μ L⁻(x_old) x_old, x ⟂ 1
#
# fixed point ⇒ L⁺x = μL⁻(x)x ⇒ X(L⁺−μL⁻)=0 (the Euler–Lagrange eqn). At collapse L⁻(0)=N·I on ⟂1,
# so the map near 0 amplifies the Fiedler mode by ρ = μN/ν₂; the first bifurcation is μ* = ν₂/N.
# We sweep μ/μ* and count outer iterations to a steady embedding.
#
# Run:  julia experiments/scf_critical_slowdown.jl

include(joinpath(@__DIR__, "..", "src", "ElasticEmbedding.jl"))
using .ElasticEmbedding
using Random, Printf, LinearAlgebra, SparseArrays, Statistics

rng = MersenneTwister(1)
n, edges, w, block = sbm_graph([200, 200], 0.06, 0.003; rng = rng)
B, w = incidence(n, edges, w)
Lp = Matrix(weighted_laplacian(B, w))
N = n

# ν₂ (algebraic connectivity) and the first critical coefficient μ* = ν₂/N
evals = eigen(Symmetric(Lp)).values
ν2 = evals[2]
μstar = ν2 / N
@printf("N=%d, ν₂=%.4g, μ* = ν₂/N = %.4g\n\n", N, ν2, μstar)

# Deflated L⁺ solver: (L⁺ + (1/N)11ᵀ) y = b  gives y = (L⁺)⁺ b for b ⟂ 1 (factor once)
Lreg = cholesky(Symmetric(Lp .+ (1.0 / N)))
solveLplus(b) = (y = Lreg \ (b .- mean(b)); y .- mean(y))

function Lminus_times(x, σ2)                 # returns L⁻(x) x  (d=1), O(N²)
    out = zeros(N)
    @inbounds for i in 1:N
        s = 0.0
        for j in 1:N
            i == j && continue
            wt = exp(-(x[i] - x[j])^2 / σ2)
            s += wt * (x[i] - x[j])          # (L⁻x)_i = Σ_j w̃_ij (x_i − x_j)
        end
        out[i] = s
    end
    out
end

function scf(μ; σ2 = 1.0, tol = 1e-6, maxit = 4000)
    x = 1e-3 .* randn(MersenneTwister(9), N); x .-= mean(x)
    for it in 1:maxit
        xn = solveLplus(μ .* Lminus_times(x, σ2))
        nx = norm(x); nxn = norm(xn)
        nxn < 1e-9 && return (:collapse, it, nxn)
        Δ = norm(xn .- x) / max(nx, 1e-12)
        x = xn
        Δ < tol && return (:converged, it, norm(x))
    end
    (:maxit, maxit, norm(x))
end

println("μ/μ*     outcome      outer iters   ‖x‖(scale)")
println("-"^52)
for r in [0.9, 0.98, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0]
    status, iters, nx = scf(r * μstar)
    @printf("%-7.2f  %-11s  %8d      %.3g\n", r, string(status), iters, nx)
end

println("\nExpectation: μ<μ* collapses to 0; just above μ* the outer count blows up (critical")
println("slowing down, iters ~ 1/log(μ/μ*)); well above μ* it drops. The INNER L⁺ solves stay cheap —")
println("it is the OUTER Picard count that explodes at the bifurcation.")
