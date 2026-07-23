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
# range grid and vector grid must assemble the identical matrix
Mv = DecayMatrix(ρ, 0, pT6, collect(qg6))
@assert maximum(abs.(M0.W .- Mv.W)) < 1e-10 * maximum(abs.(M0.W))
println("   PASS")

# ---------------------------------------------------------------- test 7
println("== 7. CascadeTables: keys, cascade vs direct, round trip, updates ==")
isdefined(Main, :CascadeTables) || include("CascadeTables.jl")
using .CascadeTables

# key invariants: partner order canonicalized, observed slot preserved
@assert ChannelKey(1.0, 0.2, [0.5, 0.3]) == ChannelKey(1.0, 0.2, [0.3, 0.5])
@assert ChannelKey(1.0, 0.2, [0.5]).md == 0.0
@assert ChannelKey(1.0, 0.2, [0.5]) != ChannelKey(1.0, 0.5, [0.2])

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
tab7b = CascadeTable(dl7; grid = g7, orders = ord7, modes = [0, 2], cache = tabfile)
res7b = runcascade(tab7b, dl7; primary = prim7)
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
@assert  haskey(filter(nbody(2), dl7).byparent, 1) && !haskey(filter(nbody(2), dl7).byparent, 3)
@assert !haskey(filter(nbody(3), dl7).byparent, 1) &&  haskey(filter(nbody(3), dl7).byparent, 3)
# combinators compose booleanly
@assert haskey(filter(nbody(2) | nbody(3), dl7).byparent, 1)     # union keeps both parents
@assert isempty(filter(nbody(2) & nbody(3), dl7).byparent)       # no channel is both
@assert haskey(filter(!nbody(3), dl7).byparent, 1) && !haskey(filter(!nbody(3), dl7).byparent, 3)
# name / id / mass selectors
@assert  haskey(filter(fromparent("om"), dl7).byparent, 3) && !haskey(filter(fromparent("om"), dl7).byparent, 1)
@assert length(filter(fromid(1), dl7).byparent) == 1 && haskey(filter(fromid(1), dl7).byparent, 1)
@assert isempty(filter(parentmass(0.0, 0.5), dl7).byparent)      # both parents heavier
# an all-true filter reproduces the full cascade bit-for-bit, REUSING tab7 in
# memory (build once, filter freely — no rebuild)
allpass = filter(channelfilter((p, ch, ds) -> true), dl7)
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

# ---------------------------------------------------------------- test 8
println("== 8. Public API contract: shapes, types, dispatch, invariants ==")
# Pins the boost-invariant public interface so refactors (and the planned
# rapidity-tensor extension) cannot silently drift it. The math is covered
# above; this asserts the *contract* every extension must preserve.
let g = range(0.0, 4.0, length = 40), pT8 = collect(0.1:0.1:1.5)
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
    @assert M.n == 0 && M.pT == pT8 && M.q == collect(g)
    @assert M.W == decaymatrices(ρ8, (0,), pT8, g)[1]

    # the parallel assembly is deterministic: threaded ≡ serial, bitwise
    @assert decaymatrices(ω8, [0, 2], pT8, g; threaded = true) ==
            decaymatrices(ω8, [0, 2], pT8, g; threaded = false)

    # functor apply contract: M(F) == M.W * F.(q)  (the reference the rapidity
    # collapse w[k,j,0]+2Σ_{d≥1}w[k,j,d] must reproduce for an η-independent source)
    F8(q) = exp(-hypot(0.7752, q) / T)
    @assert M(F8) == M.W * F8.(M.q)

    # BreitWigner: callable spectral function, positive on a window around the pole
    bw = BreitWigner(0.7752, 0.05)
    @assert bw isa BreitWigner && bw(0.7752) > 0 && bw.mmin < 0.7752 < bw.mmax

    # hot paths are type-stable (an Any-leak here would poison any assembly)
    @assert only(Base.return_types(ρ8, (typeof(F8), Float64, Float64, Float64))) == Float64
    @assert isconcretetype(only(Base.return_types(decaymatrices,
                (MassAverage, Tuple{Int}, Vector{Float64}, Vector{Float64}))))
end
println("   PASS")

println("\nAll tests passed.")
