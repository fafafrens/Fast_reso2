# Validation of DecaySpectra.jl
#
# 1. Number conservation: N_b = B · N_a follows from the normalization of the
#    decay map (Eq. 4 / Eq. 49), for any parent spectrum.
# 2. Fourier modes: the cos(nϕ)-modulated-input trick (Eq. 41) must reconstruct
#    the direct ϕ-dependent spectrum (Eq. 32) of a modulated source.
# 3. Three-body number conservation, Eq. (59).
# 3b. Four-body number conservation + cluster-mass permutation symmetry (§3.1).
# 4. Quadrature convergence: default Gauss–Legendre orders vs doubled.

include("DecaySpectra.jl")
using .DecaySpectra
using Logging
using QuadGK   # test harness only: outer adaptive integrals
using Printf

const T = 0.150   # temperature (GeV) of the static thermal test source

# static Boltzmann source: F(qT, η, ϕ) = exp(−E/T),  E = mT cosh(η)
thermal(m) = (qT, η, ϕ) -> exp(-hypot(m, qT) * cosh(η) / T)

# N = ∫ d³q/E F = ∫ qT dqT dη dϕ F   (Eq. 13)
number(F; rtol = 1e-6) =
    2π * quadgk(qT -> quadgk(η -> qT * F(qT, η, 0.0), -Inf, Inf; rtol)[1],
                0, Inf; rtol)[1]

# ---------------------------------------------------------------- test 1
println("== 1. Two-body number conservation:  ρ → ππ, static thermal source ==")
ρ = TwoBodyDecay(0.7752, 0.1396, 0.1396; B = 1.0)
Fa = thermal(0.7752)
Na = number(Fa)
Nb = 2π * quadgk(pT -> quadgk(ηp -> pT * ρ(Fa, pT, ηp, 0.0),
                              -Inf, Inf; rtol = 1e-4)[1],
                 0, Inf; rtol = 1e-4)[1]
@printf("   N_a = %.6e   N_b = %.6e   ratio N_b/(B N_a) = %.6f\n", Na, Nb, Nb / Na)
@assert isapprox(Nb, Na; rtol = 1e-3)
println("   PASS")

# ---------------------------------------------------------------- test 2
println("== 2. Fourier modes vs direct spectrum:  v2-modulated source ==")
v2 = 0.1
Fa_mod(qT, η, ϕ) = Fa(qT, η, ϕ) * (1 + 2v2 * cos(2ϕ))
Fa_n0(qT, η, ϕ) = Fa(qT, η, 0.0)                 # n = 0 mode as cos(0·ϕ) input
Fa_n2(qT, η, ϕ) = v2 * Fa(qT, η, 0.0) * cos(2ϕ)  # n = ±2 mode input
for (pT, ηp) in ((0.3, 0.0), (0.8, 0.5), (1.5, -1.0))
    b0 = ρ(Fa_n0, pT, ηp, 0.0)
    b2 = ρ(Fa_n2, pT, ηp, 0.0)
    for ϕp in (0.0, π / 3, π / 2)
        direct = ρ(Fa_mod, pT, ηp, ϕp)
        recon = b0 + 2b2 * cos(2ϕp)
        @assert isapprox(direct, recon; rtol = 1e-6)
    end
    @printf("   pT=%.2f ηp=%+.1f:  F_b0=%.4e  v2_b=F_b2/F_b0=%.4f\n", pT, ηp, b0, b2 / b0)
end
println("   PASS")

# ---------------------------------------------------------------- test 3
println("== 3. Three-body number conservation:  ω → πππ ==")
ω = ThreeBodyDecay(0.7827, 0.1396, 0.1396, 0.1350; B = 1.0)
Faω = thermal(0.7827)
Naω = number(Faω)
Nbω = 2π * quadgk(pT -> quadgk(ηp -> pT * ω(Faω, pT, ηp, 0.0),
                               -Inf, Inf; rtol = 1e-3)[1],
                  0, Inf; rtol = 1e-3)[1]
@printf("   N_a = %.6e   N_b = %.6e   ratio N_b/(B N_a) = %.6f\n", Naω, Nbω, Nbω / Naω)
@assert isapprox(Nbω, Naω; rtol = 5e-3)
println("   PASS")

# ---------------------------------------------------------------- test 3b
println("== 3b. Four-body number conservation + cluster symmetry:  a → ππππ ==")
d4 = FourBodyDecay(1.3, 0.1396, 0.1396, 0.1396, 0.1350; B = 1.0)
Fa4 = thermal(1.3)
Na4 = number(Fa4)
Nb4 = 2π * quadgk(pT -> quadgk(ηp -> pT * d4(Fa4, pT, ηp, 0.0),
                              -Inf, Inf; rtol = 1e-3)[1],
                 0, Inf; rtol = 1e-3)[1]
@printf("   N_a = %.6e   N_b = %.6e   ratio N_b/(B N_a) = %.6f\n", Na4, Nb4, Nb4 / Na4)
@assert isapprox(Nb4, Na4; rtol = 5e-3)
# ρ₃ is symmetric under permuting the three cluster masses (any particle may be
# peeled first, Eq. rho3); the constructor does NOT sort them, so permuting the
# inputs and getting the same spectrum genuinely tests the inner σ-integral.
v0 = d4(Fa4, 0.5, 0.3, 0.2)
dev = maximum(abs(FourBodyDecay(1.3, 0.1396, p...)(Fa4, 0.5, 0.3, 0.2) / v0 - 1)
              for p in ([0.1350, 0.1396, 0.1396], [0.1396, 0.1350, 0.1396]))
@printf("   cluster-mass permutation: max rel dev = %.1e\n", dev)
@assert dev < 1e-3
println("   PASS")

# ---------------------------------------------------------------- test 3c
println("== 3c. Finite parent width: BroadParent (Breit–Wigner average) ==")
let mρ = 0.7752, mπ = 0.1396, mp = 0.938, g = collect(range(0, 6, length = 300))
    # functor number conservation (ρ→ππ, ρ broad) — the BroadParent as a functor
    ρw = BroadParent(mπ, mπ, BreitWigner(mρ, 0.149; mmin = 2mπ); B = 1.0, nma = 8, nq = 64, nθ = 16)
    Faw = thermal(mρ)
    Nbw = 2π * quadgk(pT -> quadgk(ηp -> pT * ρw(Faw, pT, ηp, 0.0), -Inf, Inf; rtol = 2e-3)[1],
                      0, Inf; rtol = 2e-3)[1]
    @printf("   functor N_b/(B N_a) = %.5f\n", Nbw / number(Faw))
    @assert isapprox(Nbw, number(Faw); rtol = 5e-3)
    # matrix: a single node at the pole reproduces the sharp matrix
    Wsharp = decaymatrices(TwoBodyDecay(mρ, mπ, mπ), [0], g, g)[1]
    W1 = decaymatrices(BroadParent(mπ, mπ, BreitWigner(mρ, 1e-3); nma = 1), [0], g, g)[1]
    @assert maximum(abs.(W1 .- Wsharp)) < 1e-3 * maximum(abs.(Wsharp))
    # narrow width → sharp spectrum (number-weighted)
    Fa = [exp(-hypot(mρ, q) / T) for q in g]
    relw(F, R) = sum(abs.(F .- R) .* g) / sum(abs.(R) .* g)
    dn = relw((decaymatrices(BroadParent(mπ, mπ, BreitWigner(mρ, 1e-3; mmin = 2mπ); nma = 24), [0], g, g)[1]) * Fa,
              Wsharp * Fa)
    @assert dn < 5e-3
    # channel reopening: pole below the (π,p) threshold, spectral tail above it
    clsd = maximum(abs.(decaymatrices(BroadParent(mπ, mp, BreitWigner(1.0, 1e-3); nma = 32), [0], g, g)[1]))
    opnd = maximum(abs.(decaymatrices(BroadParent(mπ, mp, BreitWigner(1.0, 0.2);  nma = 32), [0], g, g)[1]))
    @assert clsd == 0.0 && opnd > 0.0
    # threshold-cut window: the parent-mass quadrature must sample the OPEN part.
    # Sampling the full window and discarding closed nodes left a narrowly-open
    # channel under-sampled — or with no node at all: nma=8 against a threshold at
    # 1.3 GeV with support to 1.4 GeV caught zero nodes and returned a zero matrix.
    let bwc = BreitWigner(1.0, 0.1)                    # window [0.6, 1.4]
        W8  = decaymatrices(BroadParent(0.65, 0.65, bwc; nma = 8),  [0], g, g)[1]
        W32 = decaymatrices(BroadParent(0.65, 0.65, bwc; nma = 32), [0], g, g)[1]
        ref = decaymatrices(BroadParent(0.65, 0.65, bwc; nma = 96), [0], g, g)[1]
        @assert maximum(abs.(W8)) > 0                  # was exactly 0.0 (silent zero)
        r8  = sum(abs.(W8  .- ref)) / sum(abs.(ref))   # narrow open sliver [1.3, 1.4]:
        r32 = sum(abs.(W32 .- ref)) / sum(abs.(ref))   # converges, just not fast
        @assert r32 < r8 && r32 < 0.1
        @printf("   threshold-cut window: nma=8 non-zero, rel dev %.2f→%.2f (nma 8→32)\n", r8, r32)
    end
    # fully closed: the channel threshold sits above the whole spectral window, so
    # parentmassnodes returns no node and the MassAverage carries empty arrays.
    # Both evaluation paths must agree on zero — the functor used to throw on the
    # empty reduction while the matrix path returned all-zero.
    let dclosed = BroadParent(0.60, 0.60, BreitWigner(1.0, 0.05; mmin = 0.85, mmax = 1.15))
        @assert isempty(dclosed.coef)                  # threshold 1.20 > mmax 1.15
        Wc = decaymatrices(dclosed, [0, 2], g, g)
        @assert all(all(iszero, W) for W in Wc)
        @assert dclosed(thermal(1.0), 0.9, 0.2, 0.3) === 0.0    # was: ArgumentError
        @printf("   fully-closed channel: matrix all-zero and functor 0.0 (agree)\n")
    end
    @printf("   nma=1==sharp, narrow→sharp %.1e, reopening (closed=0, open≠0): OK\n", dn)
    println("   PASS")
end

# ---------------------------------------------------------------- test 4
println("== 4. Gauss–Legendre convergence: default orders vs doubled ==")
ρfine = TwoBodyDecay(0.7752, 0.1396, 0.1396; nq = 256, nθ = 64)
ωfine = ThreeBodyDecay(0.7827, 0.1396, 0.1396, 0.1350; nµ = 64, nq = 256, nθ = 64)
for (pT, ηp) in ((0.2, 0.0), (0.3, 0.0), (1.2, 0.8))
    r2 = abs(ρ(Fa, pT, ηp, 0.4) / ρfine(Fa, pT, ηp, 0.4) - 1)
    r3 = abs(ω(Faω, pT, ηp, 0.4) / ωfine(Faω, pT, ηp, 0.4) - 1)
    @printf("   pT=%.2f ηp=%+.1f:  2-body rel.diff=%.1e   3-body rel.diff=%.1e\n",
            pT, ηp, r2, r3)
    @assert r2 < 5e-7 && r3 < 5e-7
end
println("   PASS")

# ---------------------------------------------------------------- test 5
println("== 5. LagrangeWeights: exactness, explicit vs generic, O(h^P) order ==")
using .DecaySpectra.LagrangeWeights   # the copy embedded in DecaySpectra
linterp(x, F, x1, h, P) = begin
    j0, w = lagrangeweights(x, x1, h, length(F), P)
    sum(w[m] * F[j0 + m - 1] for m in eachindex(w))
end
for P in 2:7                      # partition of unity + node exactness
    local F = rand(21)
    for x in (0.537, 0.013, 1.987)
        @assert abs(sum(lagrangeweights(x, 0.0, 0.1, 21, P)[2]) - 1) < 1e-12
    end
    for i in 3:19
        @assert abs(linterp((i - 1) * 0.1, F, 0.0, 0.1, P) - F[i]) < 1e-11
    end
end
gw(ξ, P) = ntuple(m -> prod((ξ - k) / (m - 1 - k) for k in 0:P-1 if k != m - 1), P)
for P in 2:4, ξ in 0.0:0.13:(P - 1)   # explicit formulas == generic product
    @assert all(abs.(collect(LagrangeWeights.lweights(ξ, Val(P))) .- collect(gw(ξ, P))) .< 1e-12)
end
xs5 = [2 * (k / 20)^2 for k in 0:20]  # non-uniform (stretched) grid version
for P in 2:6
    for x in (0.017, 0.31, 1.93)      # partition of unity
        @assert abs(sum(lagrangeweights(x, xs5, P)[2]) - 1) < 1e-12
    end
    local poly = x -> sum(x^d * (0.3 + 0.1d) for d in 0:P-1)
    local F = poly.(xs5)              # exact for degree < P on any spacing
    for x in 0.011:0.0173:1.99
        j0, w = lagrangeweights(x, xs5, P)
        @assert abs(sum(w[m] * F[j0 + m - 1] for m in 1:P) - poly(x)) < 1e-10
    end
end
for P in (2, 3, 4, 6), x in (0.537, 0.013, 1.987)   # vector path == uniform path
    j0a, wa = lagrangeweights(x, 0.0, 0.1, 21, P)
    j0b, wb = lagrangeweights(x, collect(0.0:0.1:2.0), P)
    @assert j0a == j0b && all(abs.(collect(wa) .- collect(wb)) .< 1e-12)
end
f5(x) = exp(-x) * cos(3x)
for P in (2, 4, 6)                # convergence order ~ 2^P per halving of h
    local errs = map((0.1, 0.05)) do h
        local N = round(Int, 2.0 / h) + 1
        local F = [f5((i - 1) * h) for i in 1:N]
        maximum(abs(linterp(x, F, 0.0, h, P) - f5(x)) for x in 0.001:0.00373:1.995)
    end
    ratio = errs[1] / errs[2]
    @printf("   P=%d:  error(h=0.1)=%.2e  ratio per halving=%.1f  (theory %d)\n",
            P, errs[1], ratio, 2^P)
    @assert ratio > 0.7 * 2^P
end
# derivative weights: sum-to-zero, polynomial-derivative exactness, non-uniform
dinterp(x, F, x1, h, P) = begin
    j0, w = lagrangederiv(x, x1, h, length(F), P)
    sum(w[m] * F[j0 + m - 1] for m in eachindex(w))
end
for P in 2:6
    @assert abs(sum(lagrangederiv(0.537, 0.0, 0.1, 21, P)[2])) < 1e-11   # d(const)=0
    local poly = x -> sum(x^d * (0.3 + 0.1d) for d in 0:P-1)
    local dpoly = x -> sum(d * x^(d - 1) * (0.3 + 0.1d) for d in 1:P-1)
    local Fu = poly.(0.0:0.05:2.0)
    @assert maximum(abs(dinterp(x, Fu, 0.0, 0.05, P) - dpoly(x)) for x in 0.05:0.03:1.9) < 1e-8
    local Fn = poly.(xs5)                     # non-uniform (stretched) grid
    @assert maximum(begin j0, w = lagrangederiv(x, xs5, P)
                          abs(sum(w[m] * Fn[j0 + m - 1] for m in 1:P) - dpoly(x)) end
                    for x in 0.05:0.03:1.9) < 1e-7
end
# second derivative: zero-sum, exactness (needs P ≥ 3), lagrangeall consistency
d2interp(x, F, x1, h, P) = begin
    j0, w = lagrangederiv2(x, x1, h, length(F), P)
    sum(w[m] * F[j0 + m - 1] for m in eachindex(w))
end
@assert all(lagrangederiv2(0.5, 0.0, 0.1, 21, Val(2))[2] .== 0.0)      # line: no curvature
for P in 3:6
    @assert abs(sum(lagrangederiv2(0.537, 0.0, 0.1, 21, P)[2])) < 1e-9
    local poly = x -> sum(x^d * (0.3 + 0.1d) for d in 0:P-1)
    local d2poly = x -> sum(d * (d - 1) * x^(d - 2) * (0.3 + 0.1d) for d in 2:P-1)
    local Fu = poly.(0.0:0.05:2.0)
    @assert maximum(abs(d2interp(x, Fu, 0.0, 0.05, P) - d2poly(x)) for x in 0.06:0.03:1.9) < 1e-6
    # lagrangeall returns value + 1st + 2nd matching the standalone calls
    local j0, w, dw, d2w = lagrangeall(0.537, 0.0, 0.1, 21, Val(P))
    @assert all(collect(w) .≈ collect(lagrangeweights(0.537, 0.0, 0.1, 21, Val(P))[2])) &&
            all(collect(dw) .≈ collect(lagrangederiv(0.537, 0.0, 0.1, 21, Val(P))[2])) &&
            all(collect(d2w) .≈ collect(lagrangederiv2(0.537, 0.0, 0.1, 21, Val(P))[2]))
end
println("   1st + 2nd derivative weights + lagrangeall: sum-to-zero, exactness OK")
println("   PASS")

# ---------------------------------------------------------------- test 6
println("== 6. DecayMatrix vs functor (boost-invariant source) ==")
pT6 = 0.1:0.1:1.5
qg6 = range(0.0, 4.2, length = 121)
FBIρ(qT) = exp(-hypot(0.7752, qT) / T)
FBIω(qT) = exp(-hypot(0.7827, qT) / T)
M0 = DecayMatrix(ρ, 0, pT6, qg6)
M2 = DecayMatrix(ρ, 2, pT6, qg6)
Mω = DecayMatrix(ω, 0, pT6, qg6)
d0 = maximum(abs.(M0(FBIρ) ./ [ρ((q, η, ϕ) -> FBIρ(q), pT, 0.0, 0.0) for pT in pT6] .- 1))
d2 = maximum(abs.(M2(q -> FBIρ(q)) ./
                  [ρ((q, η, ϕ) -> FBIρ(q) * cos(2ϕ), pT, 0.0, 0.0) for pT in pT6] .- 1))
dω = maximum(abs.(Mω(FBIω) ./ [ω((q, η, ϕ) -> FBIω(q), pT, 0.0, 0.0) for pT in pT6] .- 1))
@printf("   max rel dev:  ρ n=0 %.1e   ρ n=2 %.1e   ω n=0 %.1e\n", d0, d2, dω)
@assert d0 < 5e-4 && d2 < 5e-4 && dω < 5e-4
# Float64 ranges stay compact and select the uniform stencil. An equivalent
# vector takes the general non-uniform stencil, whose last bits may differ.
@assert M0.pT === pT6 && M0.q === qg6
qv6 = collect(qg6)
Mv = DecayMatrix(ρ, 0, pT6, qv6)
@assert Mv.q isa Vector{Float64} && Mv.q == qv6 && Mv.q !== qv6
@assert maximum(abs.(M0.W .- Mv.W)) < 1e-10 * maximum(abs.(M0.W))
# A genuinely non-uniform grid remains supported and is stored as a copied vector.
qn6 = [0.0, 0.03, 0.11, 0.26, 0.50, 0.85, 1.30, 2.0, 3.0, 4.2]
Mn = DecayMatrix(ρ, 0, range(0.1, 0.3, length = 3), qn6)
@assert Mn.q isa Vector{Float64} && Mn.q == qn6 && Mn.q !== qn6
@assert size(Mn.W) == (3, length(qn6)) && all(isfinite, Mn.W)
println("   PASS")

# ---------------------------------------------------------------- test 7
println("== 7. CascadeTables: keys, cascade vs direct, round trip, updates ==")
isdefined(Main, :CascadeTables) || include("CascadeTables.jl")
using .CascadeTables

# key invariants: partner order canonicalized, observed slot preserved
@assert ChannelKey(1.0, 0.2, [0.5, 0.3]) == ChannelKey(1.0, 0.2, [0.3, 0.5])
@assert ChannelKey(1.0, 0.2, [0.5]).md == 0.0
@assert ChannelKey(1.0, 0.2, [0.5]) != ChannelKey(1.0, 0.5, [0.2])

# parser rejects padding violations: a nonzero daughter slot past the declared
# count means daughters would be silently dropped (an under-declared 4-body mode,
# a lost radiative photon). Must fail loudly, not parse a corrupt channel.
let read = CascadeTables.ParticleList.read_decaylist, hdr = "1 x 1.0 0.14 0 1 0 0 0 0 0 12\n"
    good = tempname(); write(good, hdr * "1 2 1.0 211 -211 0 0 0\n")
    @assert read(good) isa Tuple                                # 0-padded: fine
    bad = tempname(); write(bad, hdr * "1 2 1.0 211 -211 211 0 0\n")   # slot 3 nonzero
    threw = false
    try; read(bad); catch; threw = true; end
    @assert threw
    rm.((good, bad); force = true)
end

# synthetic decay list: ρ → ππ (BR 0.6) and ω → ππϖ (BR 1.0)
mρ, mπ, mω, mϖ = 0.7752, 0.1396, 0.7827, 0.135
parts7 = Dict(1 => ParticleInfo(1, "rho", mρ, 0.0, 1.0, 0, 0),   # width 0 ⇒ sharp cascade
              2 => ParticleInfo(2, "pi", mπ, 0.0, 1.0, 0, 1),
              3 => ParticleInfo(3, "om", mω, 0.0, 1.0, 0, 0),
              4 => ParticleInfo(4, "pi0", mϖ, 0.0, 1.0, 0, 0))
byp7 = Dict(1 => [DecayChannel(1, 0.6, [2, 2])],
            3 => [DecayChannel(3, 1.0, [2, 2, 4])])
dl7 = DecayList(parts7, byp7, "synthetic", "deadbeef")

g7 = range(0.0, 3.0, length = 61)
ord7 = (nq = 32, nθ = 8, nµ = 8)
tabfile = tempname()
tab7 = CascadeTable(dl7; grid = g7, orders = ord7, modes = [0, 2], cache = tabfile)
@assert length(tab7.W) == 3            # ρ→π|π, ω→π|(πϖ), ω→ϖ|(ππ)
@assert tab7.grid === g7 && tab7.grid isa AbstractRange{Float64}
@assert tab7.modes === (0, 2)

prim7(p, n, grid) = (n == 0 ? 1.0 : 0.05) * [exp(-hypot(p.mass, q) / T) for q in grid]
res7 = runcascade(tab7, dl7; primary = prim7)

# direct reference with Main.DecaySpectra (independent module instance)
gv = collect(g7)
Fρ = prim7(parts7[1], 0, gv); Fω = prim7(parts7[3], 0, gv)
for (m, n) in ((1, 0), (2, 2))
    local Wρ = DecayMatrix(TwoBodyDecay(mρ, mπ, mπ; nq = 32, nθ = 8), n, gv, gv).W
    local Wω = DecayMatrix(ThreeBodyDecay(mω, mπ, mπ, mϖ; nµ = 8, nq = 32, nθ = 8), n, gv, gv).W
    # mode-n parent inputs are prim7(·, n, ·) = (n==0 ? 1 : 0.05)·Boltzmann;
    # ρ→ππ feeds both slots (×2 on BR 0.6), ω→ππϖ feeds the two π slots (×2)
    local s = n == 0 ? 1.0 : 0.05
    local expect = prim7(parts7[2], n, gv) .+ 1.2 .* (Wρ * (s .* Fρ)) .+ 2.0 .* (Wω * (s .* Fω))
    local dev = maximum(abs.(res7.F[2][m] .- expect) ./ (abs.(expect) .+ 1e-300))
    @printf("   mode n=%d: cascade vs direct max rel dev = %.1e\n", n, dev)
    @assert dev < 1e-5                 # Float32 matrix storage rounding
end

# round trip: loading must reproduce exactly
tab7native = loadtables(tabfile)
@assert typeof(tab7native.grid) === typeof(tab7.grid)
@assert tab7native.modes === (0, 2)
@assert tab7native.grid == tab7.grid && tab7native.W == tab7.W
tab7b = CascadeTable(dl7; grid = g7, orders = ord7, modes = [0, 2], cache = tabfile)
res7b = runcascade(tab7b, dl7; primary = prim7)
@assert tab7b.grid isa AbstractRange{Float64}
@assert res7b.F[2][1] == res7.F[2][1] && res7b.F[2][2] == res7.F[2][2]

# lenient update: a new channel builds only its missing key; prune removes it
# (copies — dl7 references the original dicts and must stay unchanged)
parts7c = copy(parts7); byp7c = copy(byp7)
parts7c[5] = ParticleInfo(5, "X", 1.6, 0.1, 1.0, 0, 0)
byp7c[5] = [DecayChannel(5, 0.5, [1, 2])]    # X → ρπ
dl7c = DecayList(parts7c, byp7c, "synthetic", "cafebabe")
tab7c = CascadeTable(dl7c; grid = g7, orders = ord7, modes = [0, 2], cache = tabfile)
@assert length(tab7c.W) == 5           # + X→ρ|π and X→π|ρ
@assert prune!(tab7c, dl7) == 2        # back to the original three
rm(tabfile; force = true)
println("   PASS")

# ---------------------------------------------------------------- test 7b
println("== 7b. Channel filtering: ChannelFilter selection, combinators, reuse ==")
# dl7 = ρ→ππ (2-body, pid 1) and ω→ππϖ (3-body, pid 3)
let
    parentids(selection) = Set(keys(filter(selection, dl7).byparent))
    # Channel properties and boolean composition.
    @assert parentids(nbody(2)) == Set([1])
    @assert parentids(nbody(3)) == Set([3])
    @assert parentids(nbody(2) | nbody(3)) == Set([1, 3])
    @assert isempty(parentids(nbody(2) & nbody(3)))
    @assert parentids(!nbody(3)) == Set([1])
    @assert parentids(brabove(0.8)) == Set([3])

    # Parent properties.
    @assert parentids(fromparent("om")) == Set([3])
    @assert parentids(fromid(1)) == Set([1])
    @assert isempty(parentids(parentmass(0.0, 0.5)))
    @assert parentids(mesonicparent()) == Set([1, 3])
    @assert parentids(parentstrangeness(0)) == Set([1, 3])
    @assert parentids(parentisospin(0)) == Set([1, 3])
    @assert isempty(parentids(baryonicparent()))
    @assert isempty(parentids(strangeparent()))
    @assert isempty(parentids(charmparent()))

    # Daughter properties.
    @assert parentids(todaughter("pi0")) == Set([3])
    @assert parentids(toid(4)) == Set([3])
    @assert parentids(daughtermass(0.139, 0.140)) == Set([1])
end

# an all-true filter reproduces the full cascade bit-for-bit, REUSING tab7 in
# memory (build once, filter freely — no rebuild)
allpass = filter(channelfilter((p, ch, ds) -> true), dl7)
@assert allpass.parts === dl7.parts
@assert allpass.byparent == dl7.byparent
@assert allpass.path == dl7.path && allpass.sha == dl7.sha
resall = runcascade(tab7, allpass; primary = prim7)
@assert resall.F[2][1] == res7.F[2][1] && resall.F[2][2] == res7.F[2][2]
# restricting to 2-body drops the ω feed-down: mode-0 pion yield falls but stays
# ≥ primary (feed-down is non-negative at mode 0, and 2-body ⊆ full)
res2 = runcascade(tab7, filter(nbody(2), dl7); primary = prim7)
@assert all(res2.F[2][1] .≤ res7.F[2][1]) && all(res2.F[2][1] .≥ prim7(parts7[2], 0, gv))
println("   PASS")

# ---------------------------------------------------------------- test 7c
println("== 7c. Cascade with a broad parent (finite width, Part B wiring) ==")
let g = range(0, 3, length = 61), ord = (nq = 32, nθ = 8, nµ = 8, nσ = 8, nma = 8)
    prim(p, n, gr) = (n == 0 ? 1.0 : 0.0) .* [exp(-hypot(p.mass, q) / T) for q in gr]
    πi = ParticleInfo(2, "pi", 0.1396, 0.0, 1.0, 0, 1)
    byp = Dict(1 => [DecayChannel(1, 1.0, [2, 2])])                         # R → ππ
    dl0 = DecayList(Dict(1 => ParticleInfo(1, "R", 0.9, 0.0,  1.0, 0, 0), 2 => πi), byp, "s", "s0")
    dlb = DecayList(Dict(1 => ParticleInfo(1, "R", 0.9, 0.12, 1.0, 0, 0), 2 => πi), byp, "b", "b0")
    r0 = runcascade(CascadeTable(dl0; grid = g, orders = ord, modes = [0]), dl0; primary = prim)
    rb = runcascade(CascadeTable(dlb; grid = g, orders = ord, modes = [0]), dlb; primary = prim)
    # compare the feed-down (F − Fprim); the total F is diluted by the primary pion
    f0 = r0.F[2][1] .- r0.Fprim[2][1]
    fb = rb.F[2][1] .- rb.Fprim[2][1]
    dif = maximum(abs.(fb .- f0)) / maximum(abs.(f0))
    @assert dif > 1e-2        # the Γ=0.12 parent takes the BroadParent path and smears the pion
    @printf("   broad-parent pion feed-down differs from sharp by %.1e\n", dif)
    println("   PASS")
end

# ---------------------------------------------------------------- test 7d
println("== 7d. Reopened channel feeding a heavier daughter (sweep ordering) ==")
# Broad P (pole 1.0, Γ=0.1 → tail to 1.4) → D(1.1) x(0.15): closed at the pole
# (Σm=1.25), reopened by the tail. D(1.1) is HEAVIER than P, so a pole-mass sweep
# processes D first and drops the P→D→(a,b) feed-down. cascadeorder must place P
# before D (feeder first).
let g = range(0, 3, length = 61), ord = (nq = 32, nθ = 8, nµ = 8, nσ = 8, nma = 8)
    parts = Dict(1 => ParticleInfo(1, "P", 1.0, 0.1, 1.0, 0, 0),
                 2 => ParticleInfo(2, "D", 1.1, 0.0, 1.0, 0, 0),
                 3 => ParticleInfo(3, "x", 0.15, 0.0, 1.0, 0, 0),
                 4 => ParticleInfo(4, "a", 0.5, 0.0, 1.0, 0, 0),
                 5 => ParticleInfo(5, "b", 0.5, 0.0, 1.0, 0, 0))
    byp = Dict(1 => [DecayChannel(1, 1.0, [2, 3])], 2 => [DecayChannel(2, 1.0, [4, 5])])
    dl = DecayList(parts, byp, "syn", "h0")
    ord_ = CascadeTables.cascadeorder(dl)
    @assert findfirst(==(1), ord_) < findfirst(==(2), ord_)   # P processed before D
    tab = CascadeTable(dl; grid = g, orders = ord, modes = [0], cache = nothing)
    prim(p, n, gr) = (n == 0 && p.id == 1) ? [exp(-hypot(p.mass, q) / 0.15) for q in gr] : zeros(length(gr))
    res = runcascade(tab, dl; primary = prim)
    planned = runcascade(CascadePlan(tab, dl); primary = prim)
    for particle_id in keys(res.F), mode_index in eachindex(tab.modes)
        reference = res.F[particle_id][mode_index]
        error = maximum(abs.(planned.F[particle_id][mode_index] .- reference)) /
                max(maximum(abs.(reference)), 1.0)
        @assert error ≤ 1e-12
        @assert planned.Fprim[particle_id][mode_index] ==
                res.Fprim[particle_id][mode_index]
    end
    fda = sum(res.F[4][1] .- res.Fprim[4][1])                 # P→D→a downstream feed-down
    @assert fda > 0                                           # was exactly 0.0 under mass-order
    @printf("   P before D; downstream P→D→a feed-down = %.2e (was 0)\n", fda)
    println("   PASS")
end

# ---------------------------------------------------------------- test 7e
println("== 7e. Flat CascadePlan: grouping, reuse, threading, validation ==")

function assert_plan_matches(candidate, reference)
    @assert Set(keys(candidate.F)) == Set(keys(reference.F))
    @assert Set(keys(candidate.Fprim)) == Set(keys(reference.Fprim))
    for particle_id in keys(reference.F)
        @assert length(candidate.F[particle_id]) ==
                length(reference.F[particle_id])
        for mode_index in eachindex(reference.F[particle_id])
            expected = reference.F[particle_id][mode_index]
            actual = candidate.F[particle_id][mode_index]
            error = maximum(abs.(actual .- expected)) /
                    max(maximum(abs.(expected)), 1.0)
            @assert error ≤ 1e-12
            @assert candidate.Fprim[particle_id][mode_index] ==
                    reference.Fprim[particle_id][mode_index]
        end
    end
    return nothing
end

# The ordinary two-mode topology covers repeated daughter slots (rho → pi pi)
# and verifies that the compiled arrays form gap-free flat ranges.
plan7 = CascadePlan(tab7, dl7)
planned7 = runcascade(plan7; primary = prim7)
assert_plan_matches(planned7, res7)
@assert plan7.ids == sort!(collect(keys(dl7.parts)))
@assert [index for step in plan7.steps for index in step.groups] ==
        collect(eachindex(plan7.groups))
@assert [index for group in plan7.groups for index in group.targets] ==
        collect(eachindex(plan7.targets))

# A plan keeps references to table matrices; no matrix payload is copied.
matrix_pool7 = [
    matrix
    for matrices in values(tab7.W)
    for matrix in matrices
]
@assert all(
    any(matrix === table_matrix for table_matrix in matrix_pool7)
    for group in plan7.groups
    for matrix in group.matrices
)

# Mode tasks preserve the serial operation order within every mode.
threaded7 = runcascade(plan7; primary = prim7, parallel = :modes)
@assert threaded7.F == planned7.F
@assert threaded7.Fprim == planned7.Fprim

# Reuse with a different callback must not retain or mutate run state.
prim7alt(p, n, grid) =
    (1.0 + p.id / 10) * (n == 0 ? 0.7 : -0.03) *
    [exp(-hypot(p.mass, q) / (T * 1.1)) for q in grid]
planned7_snapshot = deepcopy(planned7.F)
alternate7 = runcascade(plan7; primary = prim7alt)
alternate7_reference = runcascade(tab7, dl7; primary = prim7alt)
assert_plan_matches(alternate7, alternate7_reference)
@assert planned7.F == planned7_snapshot

# The plan itself is immutable during execution and may serve concurrent calls.
task7a = Threads.@spawn runcascade(plan7; primary = prim7)
task7b = Threads.@spawn runcascade(plan7; primary = prim7alt)
concurrent7a, concurrent7b = fetch(task7a), fetch(task7b)
@assert concurrent7a.F == planned7.F
@assert concurrent7a.Fprim == planned7.Fprim
@assert concurrent7b.F == alternate7.F
@assert concurrent7b.Fprim == alternate7.Fprim

# A filtered topology compiles against the already-built full table.
two_body7 = filter(nbody(2), dl7)
two_body_plan7 = CascadePlan(tab7, two_body7)
assert_plan_matches(
    runcascade(two_body_plan7; primary = prim7),
    runcascade(tab7, two_body7; primary = prim7),
)
empty_decay_list7 =
    filter(channelfilter((_parent, _channel, _daughters) -> false), dl7)
empty_plan7 = CascadePlan(tab7, empty_decay_list7)
@assert isempty(empty_plan7.steps)
@assert isempty(empty_plan7.groups)
@assert isempty(empty_plan7.targets)
empty_result7 = runcascade(empty_plan7; primary = prim7)
@assert empty_result7.F == empty_result7.Fprim

# Execution no longer consults the keyed table after plan construction.
snapshot_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    tab7.modes,
    copy(tab7.meta),
    copy(tab7.W),
)
snapshot_plan7 = CascadePlan(snapshot_table7, dl7)
snapshot_result7 = runcascade(snapshot_plan7; primary = prim7)
empty!(snapshot_table7.W)
@assert runcascade(snapshot_plan7; primary = prim7).F == snapshot_result7.F

# Mode position, rather than the numerical mode value, selects a matrix.
reversed_W7 = Dict{ChannelKey,Vector{Matrix{Float32}}}(
    key => reverse(matrices) for (key, matrices) in tab7.W
)
reversed_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    reverse(tab7.modes),
    copy(tab7.meta),
    reversed_W7,
)
reversed_plan7 = CascadePlan(reversed_table7, dl7)
reversed_serial7 = runcascade(reversed_plan7; primary = prim7)
assert_plan_matches(
    reversed_serial7,
    runcascade(reversed_table7, dl7; primary = prim7),
)
@assert runcascade(
    reversed_plan7;
    primary = prim7,
    parallel = :modes,
).F == reversed_serial7.F

one_mode_W7 = Dict{ChannelKey,Vector{Matrix{Float32}}}(
    key => [matrices[1]] for (key, matrices) in tab7.W
)
one_mode_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    (tab7.modes[1],),
    copy(tab7.meta),
    one_mode_W7,
)
one_mode_plan7 = CascadePlan(one_mode_table7, dl7)
one_mode_serial7 = runcascade(one_mode_plan7; primary = prim7)
assert_plan_matches(
    one_mode_serial7,
    runcascade(one_mode_table7, dl7; primary = prim7),
)
@assert runcascade(
    one_mode_plan7;
    primary = prim7,
    parallel = :modes,
).F == one_mode_serial7.F

# Invalid execution options fail before the callback is invoked.
for invalid_parallel in (:threads, "modes", 2)
    let threw = false
        try
            runcascade(
                plan7;
                primary = (_args...) -> error("primary must not be called"),
                parallel = invalid_parallel,
            )
        catch error
            threw =
                error isa ArgumentError &&
                occursin(":serial or :modes", sprint(showerror, error))
        end
        @assert threw
    end
end

let threw = false
    try
        runcascade(
            plan7;
            primary = (_particle, _mode, grid) -> zeros(length(grid) - 1),
        )
    catch error
        threw = error isa DimensionMismatch
    end
    @assert threw
end

# Missing or malformed table payloads are rejected while compiling the plan.
missing_W7 = copy(tab7.W)
missing_key7 = first(keys(missing_W7))
delete!(missing_W7, missing_key7)
missing_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    tab7.modes,
    copy(tab7.meta),
    missing_W7,
)
let threw = false
    try
        CascadePlan(missing_table7, dl7)
    catch error
        threw = error isa KeyError && error.key == missing_key7
    end
    @assert threw
end

malformed_W7 = copy(tab7.W)
malformed_key7 = first(keys(malformed_W7))
malformed_W7[malformed_key7] = malformed_W7[malformed_key7][1:1]
malformed_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    tab7.modes,
    copy(tab7.meta),
    malformed_W7,
)
let threw = false
    try
        CascadePlan(malformed_table7, dl7)
    catch error
        threw = error isa DimensionMismatch
    end
    @assert threw
end

bad_shape_W7 = copy(tab7.W)
bad_shape_key7 = first(keys(bad_shape_W7))
bad_shape_matrices7 = copy(bad_shape_W7[bad_shape_key7])
bad_shape_matrices7[1] =
    zeros(Float32, length(tab7.grid) - 1, length(tab7.grid))
bad_shape_W7[bad_shape_key7] = bad_shape_matrices7
bad_shape_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    tab7.modes,
    copy(tab7.meta),
    bad_shape_W7,
)
let threw = false
    try
        CascadePlan(bad_shape_table7, dl7)
    catch error
        threw = error isa DimensionMismatch
    end
    @assert threw
end

# Equal-mass daughters share one transfer. Repeated slots/channels accumulate
# independently for each dense destination.
let
    grid = range(0.0, 2.0, length = 17)
    orders = (nq = 8, nθ = 4, nµ = 4)
    parts = Dict(
        10 => ParticleInfo(10, "P", 1.5, 0.0, 1.0, 0, 0),
        11 => ParticleInfo(11, "Q", 1.5, 0.0, 1.0, 0, 0),
        20 => ParticleInfo(20, "a", 0.2, 0.0, 1.0, 0, 0),
        21 => ParticleInfo(21, "b", 0.2, 0.0, 1.0, 0, 0),
        30 => ParticleInfo(30, "x", 0.3, 0.0, 1.0, 0, 0),
    )
    byparent = Dict(
        10 => [
            DecayChannel(10, 0.2, [20, 30]),
            DecayChannel(10, 0.3, [21, 30]),
            DecayChannel(10, 0.1, [20, 30]),
        ],
        11 => [DecayChannel(11, 0.4, [21, 30])],
    )
    decay_list = DecayList(parts, byparent, "grouped", "grouped")
    table = CascadeTable(
        decay_list;
        grid,
        orders,
        modes = [0, 2],
        cache = nothing,
    )
    plan = CascadePlan(table, decay_list)
    @assert plan.ids[[step.parent for step in plan.steps]] == [10, 11]
    @assert length(plan.groups) == 4
    @assert sort([length(group.targets) for group in plan.groups]) ==
            [1, 1, 1, 2]
    @assert plan.groups[first(plan.steps[1].groups)].matrices[1] ===
            plan.groups[first(plan.steps[2].groups)].matrices[1]

    shared_group = only(
        group for group in plan.groups if length(group.targets) == 2
    )
    shared_weights = Dict(
        plan.ids[plan.targets[index].daughter] =>
            plan.targets[index].weight
        for index in shared_group.targets
    )
    @assert isapprox(shared_weights[20], 0.3; atol = 1e-15)
    @assert isapprox(shared_weights[21], 0.3; atol = 1e-15)

    primary(p, n, gr) =
        (n == 0 ? 1.0 : 0.04) *
        [exp(-hypot(p.mass, q) / T) for q in gr]
    assert_plan_matches(
        runcascade(plan; primary),
        runcascade(table, decay_list; primary),
    )
end

# A cycle retains the legacy deterministic mass-descending approximate sweep.
let
    grid = range(0.0, 1.0, length = 7)
    parts = Dict(
        101 => ParticleInfo(101, "P", 1.0, 0.1, 1.0, 0, 0),
        102 => ParticleInfo(102, "Q", 1.0, 0.1, 1.0, 0, 0),
        103 => ParticleInfo(103, "x", 0.1, 0.0, 1.0, 0, 0),
    )
    byparent = Dict(
        101 => [DecayChannel(101, 0.2, [102, 103])],
        102 => [DecayChannel(102, 0.3, [101, 103])],
    )
    decay_list = DecayList(parts, byparent, "cycle", "cycle")
    matrices = Dict{ChannelKey,Vector{Matrix{Float32}}}()
    CascadeTables.eachchannelkey(decay_list) do _parent, _channel, _daughter, key
        get!(matrices, key) do
            [fill(0.002f0, length(grid), length(grid))]
        end
    end
    table = CascadeTable(
        grid,
        tab7.orders,
        (0,),
        Dict{Symbol,Any}(),
        matrices,
    )
    cycle_log = IOBuffer()
    plan = with_logger(SimpleLogger(cycle_log, Logging.Warn)) do
        CascadePlan(table, decay_list)
    end
    @assert occursin("cycle", lowercase(String(take!(cycle_log))))
    @assert plan.ids[[step.parent for step in plan.steps]] == [101, 102]
    @assert length(plan.groups) == 4
    primary(p, _n, gr) = fill(p.id == 101 ? 1.0 : 0.1, length(gr))
    reference = with_logger(SimpleLogger(IOBuffer(), Logging.Error)) do
        runcascade(table, decay_list; primary)
    end
    assert_plan_matches(
        runcascade(plan; primary),
        reference,
    )
end

println("   PASS")

# ---------------------------------------------------------------- test 8
println("== 8. Public API contract: shapes, types, dispatch, invariants ==")
# Pins the boost-invariant public interface so refactors (and the planned
# rapidity-tensor extension) cannot silently drift it. The math is covered
# above; this asserts the *contract* every extension must preserve.
let g = range(0.0, 4.0, length = 40), pT8 = collect(0.1:0.1:1.5)
    rules8 = Rules(; nq = 8, nθ = 4, nµ = 4, nσ = 4, nma = 4)
    @assert fieldtype(typeof(rules8), :µ) === typeof(rules8.µ)
    @assert fieldtype(typeof(rules8), :σ) === typeof(rules8.σ)

    ρ8  = TwoBodyDecay(0.7752, 0.1396, 0.1396)
    ω8  = ThreeBodyDecay(0.7827, 0.1396, 0.1396, 0.1350)
    d48 = FourBodyDecay(1.3, 0.1396, 0.1396, 0.1396, 0.1350)
    bp8 = BroadParent(0.1396, 0.1396, BreitWigner(0.7752, 0.05); nma = 6)

    # every decay constructor collapses to the one unified dispatch type
    @assert all(x isa MassAverage for x in (ρ8, ω8, d48, bp8))
    # node arrays are aligned; two-body is one node, µ-averages are many
    for x in (ρ8, ω8, d48, bp8)
        @assert length(x.ma) == length(x.Eb) == length(x.coef) ≥ 1
    end
    @assert length(ρ8.coef) == 1 && length(ω8.coef) > 1

    # decaymatrices(::Vector) : Vector{Matrix{Float64}}, one (npT × nq) per mode
    W = decaymatrices(ρ8, [0, 2], pT8, g)
    @assert W isa Vector{Matrix{Float64}} && length(W) == 2
    @assert all(size(w) == (length(pT8), length(g)) for w in W)

    # T kwarg selects the storage eltype and is exactly the rounded Float64 result
    W32 = decaymatrices(ρ8, [0, 2], pT8, g; T = Float32)
    @assert W32 isa Vector{Matrix{Float32}} && all(W32[m] == Float32.(W[m]) for m in 1:2)

    # single-mode DecayMatrix wraps the fused core: .W ≡ decaymatrices((n,))[1]
    M = DecayMatrix(ρ8, 0, pT8, g)
    @assert M.W isa Matrix{Float64} && size(M.W) == (length(pT8), length(g))
    @assert M.n == 0 && M.pT == pT8 && M.pT !== pT8 && M.q === g
    @assert M.W == decaymatrices(ρ8, (0,), pT8, g)[1]

    # Non-Float64 ranges are copied element-by-element rather than reconstructed
    # from endpoints, preserving their actual rounded sample values.
    pT32 = 0.1f0:0.1f0:0.3f0
    qi = 0:1:4
    Mi = DecayMatrix(ρ8, 0, pT32, qi; P = Val(2))
    @assert Mi.pT isa Vector{Float64} && Mi.pT == collect(Float64, pT32)
    @assert Mi.q isa Vector{Float64} && Mi.q == collect(Float64, qi)
    @assert Mi.W == decaymatrices(ρ8, (0,), pT32, qi; P = Val(2))[1]
    Wi = (zeros(length(pT32), length(qi)),)
    decaymatrices!(Wi, ρ8, (0,), pT32, qi; P = Val(2))
    @assert Wi[1] == Mi.W
    W17 = decaymatrices(ρ8, collect(0:16), pT32, qi; P = Val(2))
    @assert W17[1] == decaymatrices(ρ8, [0], pT32, qi; P = Val(2))[1]

    # the parallel assembly is deterministic: threaded ≡ serial, bitwise
    @assert decaymatrices(ω8, [0, 2], pT8, g; threaded = true) ==
            decaymatrices(ω8, [0, 2], pT8, g; threaded = false)

    # Val keeps the stencil width known to the compiler in the innermost loop.
    @assert size(decaymatrices(ρ8, [0], pT8, g; P = Val(6))[1]) == (length(pT8), length(g))

    # mode argument is robust: any Integer eltype is accepted (was a MethodError
    # deep in the assembly for non-Int), and a negative Fourier mode reuses the
    # |n| Chebyshev factor (T is even in n; a negative n used to return T = Z)
    @assert decaymatrices(ρ8, Int32[0, 2], pT8, g) == W        # was: throws
    @assert decaymatrices(ρ8, Int8[0, 2],  pT8, g) == W
    @assert decaymatrices(ρ8, [-2], pT8, g)[1] == decaymatrices(ρ8, [2], pT8, g)[1]

    # functor apply contract: M(F) == M.W * F.(q)  (the reference the rapidity
    # collapse w[k,j,0]+2Σ_{d≥1}w[k,j,d] must reproduce for an η-independent source)
    F8(q) = exp(-hypot(0.7752, q) / T)
    @assert M(F8) == M.W * F8.(M.q)

    # BreitWigner: callable spectral function, positive on a window around the pole
    bw = BreitWigner(0.7752, 0.05)
    @assert bw isa BreitWigner && bw(0.7752) > 0 && bw.mmin < 0.7752 < bw.mmax

    # very broad state: the default mmin = mR − kΓ would go negative, and a
    # negative mmin squares inside the ψ map — the parent-mass window silently
    # inverted into the far upper tail (all nodes ABOVE the pole). Clamped to 0.
    bwv = BreitWigner(0.475, 0.55)                 # f0(500)-like, mR − 4Γ < 0
    @assert bwv.mmin == 0.0
    masv, wsv, _ = DecaySpectra.parentmassnodes(bwv, DecaySpectra.MassRule(16))
    @assert sum(masv .* wsv) / sum(wsv) < bwv.mR + bwv.Γ   # mean near pole, not tail

    # BreitWigner validates its domain. Each of these used to be accepted and to
    # yield an *empty* parent-mass average — a channel whose matrix is all zeros,
    # feeding nothing downstream with nothing to indicate why.
    for bad in (() -> BreitWigner(1.0,  0.0),                        # Γ = 0: ψ map divides by 0
                () -> BreitWigner(1.0, -0.1),                        # Γ < 0: inverts the window
                () -> BreitWigner(-1.0, 0.1),                        # pole outside the support
                () -> BreitWigner(NaN,  0.1),                        # non-finite
                () -> BreitWigner(Inf,  0.1),
                () -> BreitWigner(1.0, 0.1; mmin = 1.5, mmax = 0.5), # reversed
                () -> BreitWigner(1.0, 0.1; mmin = 1.0, mmax = 1.0), # empty
                () -> Rules(; nq = 0), () -> Rules(; nµ = -1))       # orders
        threw = false
        try; bad(); catch e; threw = e isa ArgumentError; end
        @assert threw
    end

    # constructors reject non-open decays: at exact threshold p_b = 0 makes
    # coef = Inf (the functor then returns NaN) — must fail loud instead
    for bad in (() -> TwoBodyDecay(0.14 + 0.14, 0.14, 0.14),
                () -> ThreeBodyDecay(0.14 + 0.14 + 0.14, 0.14, 0.14, 0.14),
                () -> FourBodyDecay(4 * 0.14, 0.14, 0.14, 0.14, 0.14))
        threw = false
        try; bad(); catch; threw = true; end
        @assert threw
    end

    # the functor is type-stable (an Any-leak here would poison any assembly);
    # kwarg-free, so its inferred return type is stable across Julia versions
    @assert only(Base.return_types(ρ8, (typeof(F8), Float64, Float64, Float64))) == Float64
    # the fused core returns a concrete Vector{Matrix{Float64}} (exact, runtime —
    # inferring a kwarg function's return type is version-sensitive, so check the
    # value's type rather than Base.return_types)
    @assert typeof(decaymatrices(ρ8, (0,), pT8, g)) == Vector{Matrix{Float64}}

    # decaymatrices! promises to write EVERY entry, so callers may hand it a
    # dirty destination (a reused staging buffer, an Mmap-backed matrix). Pre-fill
    # with NaN, not 0: a gap in the row partition would read back as 0 and hide,
    # whereas a NaN survives and fails loudly.
    let np = length(pT8), nq = length(g)
        for thr in (false, true)
            Ws = (fill(NaN, np, nq), fill(NaN, np, nq))
            decaymatrices!(Ws, ρ8, (0, 2), pT8, g; threaded = thr)
            @assert !any(isnan, Ws[1]) && !any(isnan, Ws[2])   # full coverage
            @assert collect(Ws) == decaymatrices(ρ8, [0, 2], pT8, g; threaded = thr)
        end
        # a channel with no nodes must still overwrite the destination with zeros
        dcl = BroadParent(0.60, 0.60, BreitWigner(1.0, 0.05; mmin = 0.85, mmax = 1.15))
        Wz = (fill(NaN, np, nq),)
        decaymatrices!(Wz, dcl, (0,), pT8, g)
        @assert all(iszero, Wz[1])
        # shape is checked, not assumed — of the output and of the scratch
        for bad in (() -> decaymatrices!((fill(NaN, np, nq),), ρ8, (0, 2), pT8, g),
                    () -> decaymatrices!((fill(NaN, np + 1, nq), fill(NaN, np + 1, nq)),
                                         ρ8, (0, 2), pT8, g),
                    () -> decaymatrices!(Wz, ρ8, (0,), pT8, g;             # buffers too short
                                         scratch = rowscratch((0,), nq - 1)),
                    () -> decaymatrices!(Wz, ρ8, (0,), pT8, g;             # wrong buffer count
                                         scratch = rowscratch((0, 2), nq)))
            threw = false
            try; bad(); catch e; threw = e isa DimensionMismatch; end
            @assert threw
        end

        # a caller-supplied scratch removes the last allocation: the row buffers.
        # Reused across calls (each row zeroes its buffer first), so the result must
        # not depend on what the scratch held on entry.
        pin(Ws, d, mt, pT, gr, sc) =
            decaymatrices!(Ws, d, mt, pT, gr; threaded = false, scratch = sc)
        let Ws = (zeros(np, nq), zeros(np, nq)), sc = rowscratch((0, 2), nq)
            ref = decaymatrices(ρ8, [0, 2], pT8, g)
            pin(Ws, ρ8, (0, 2), pT8, g, sc)                 # compile
            @assert collect(Ws) == ref
            fill!(sc[1], NaN); fill!(sc[2], NaN)            # dirty it
            pin(Ws, ρ8, (0, 2), pT8, g, sc)
            @assert collect(Ws) == ref                      # reuse is exact
            # measured 0 B here; a small bound absorbs kwarg-lowering differences
            # across Julia versions while still proving the buffers are gone
            @assert (@allocated pin(Ws, ρ8, (0, 2), pT8, g, sc)) < 128
        end
    end
end
println("   PASS")

println("\nAll tests passed.")
