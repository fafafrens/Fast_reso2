# Validation of DecaySpectra.jl
#
# 1. Number conservation: N_b = B · N_a follows from the normalization of the
#    decay map (Eq. 4 / Eq. 49), for any parent spectrum.
# 2. Fourier modes: the cos(nϕ)-modulated-input trick (Eq. 41) must reconstruct
#    the direct ϕ-dependent spectrum (Eq. 32) of a modulated source.
# 3. Three-body number conservation, Eq. (59).
# 3b. Four-body number conservation + cluster-mass permutation symmetry (§3.1).
# 4. Quadrature convergence: default Gauss–Legendre orders vs doubled.

const REPO = normpath(joinpath(@__DIR__, ".."))  # repository root for data/ and tools/

using FastReso2
const DecaySpectra = FastReso2.DecaySpectra
const CascadeTables = FastReso2.CascadeTables
const ThermalModel = FastReso2.ThermalModel
using Logging
using LinearAlgebra
using Serialization
using QuadGK   # test harness only: outer adaptive integrals
using Printf
using Test: @inferred

const T = 0.15   # temperature (GeV) of the static thermal test source

# static Boltzmann source: F(qT, η, ϕ) = exp(−E/T),  E = mT cosh(η)
thermal(m) = (qT, η, ϕ) -> exp(-hypot(m, qT) * cosh(η) / T)

# N = ∫ d³q/E F = ∫ qT dqT dη dϕ F   (Eq. 13)
number(F; rtol = 1.0e-6) =
    2π * quadgk(
    qT -> quadgk(η -> qT * F(qT, η, 0.0), -Inf, Inf; rtol)[1],
    0, Inf; rtol
)[1]

# ---------------------------------------------------------------- test 1
println("== 1. Two-body number conservation:  ρ → ππ, static thermal source ==")
ρ = TwoBodyDecay(0.7752, 0.1396, 0.1396; B = 1.0)
Fa = thermal(0.7752)
Na = number(Fa)
Nb = 2π * quadgk(
    pT -> quadgk(
        ηp -> pT * ρ(Fa, pT, ηp, 0.0),
        -Inf, Inf; rtol = 1.0e-4
    )[1],
    0, Inf; rtol = 1.0e-4
)[1]
@printf("   N_a = %.6e   N_b = %.6e   ratio N_b/(B N_a) = %.6f\n", Na, Nb, Nb / Na)
@assert isapprox(Nb, Na; rtol = 1.0e-3)
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
        @assert isapprox(direct, recon; rtol = 1.0e-6)
    end
    @printf("   pT=%.2f ηp=%+.1f:  F_b0=%.4e  v2_b=F_b2/F_b0=%.4f\n", pT, ηp, b0, b2 / b0)
end

# Callable evaluation is serial unless explicitly requested, and opting into
# threads must not change the callback's accepted numeric result type.
complex_parent(qT, η, ϕ) = complex(Fa(qT, η, ϕ), 0.25 * Fa(qT, η, ϕ))
complex_serial = ρ(complex_parent, 0.8, 0.2, 0.1)
complex_threaded = ρ(complex_parent, 0.8, 0.2, 0.1; threaded = true)
@assert complex_serial isa ComplexF64
@assert complex_threaded isa ComplexF64
@assert isapprox(complex_threaded, complex_serial; rtol = 1.0e-13)
println("   PASS")

# ---------------------------------------------------------------- test 3
println("== 3. Three-body number conservation:  ω → πππ ==")
ω = ThreeBodyDecay(0.7827, 0.1396, 0.1396, 0.135; B = 1.0)
Faω = thermal(0.7827)
Naω = number(Faω)
Nbω = 2π * quadgk(
    pT -> quadgk(
        ηp -> pT * ω(Faω, pT, ηp, 0.0),
        -Inf, Inf; rtol = 1.0e-3
    )[1],
    0, Inf; rtol = 1.0e-3
)[1]
@printf("   N_a = %.6e   N_b = %.6e   ratio N_b/(B N_a) = %.6f\n", Naω, Nbω, Nbω / Naω)
@assert isapprox(Nbω, Naω; rtol = 5.0e-3)
println("   PASS")

# ---------------------------------------------------------------- test 3b
println("== 3b. Four-body number conservation + cluster symmetry:  a → ππππ ==")
d4 = FourBodyDecay(1.3, 0.1396, 0.1396, 0.1396, 0.135; B = 1.0)
Fa4 = thermal(1.3)
Na4 = number(Fa4)
Nb4 = 2π * quadgk(
    pT -> quadgk(
        ηp -> pT * d4(Fa4, pT, ηp, 0.0),
        -Inf, Inf; rtol = 1.0e-3
    )[1],
    0, Inf; rtol = 1.0e-3
)[1]
@printf("   N_a = %.6e   N_b = %.6e   ratio N_b/(B N_a) = %.6f\n", Na4, Nb4, Nb4 / Na4)
@assert isapprox(Nb4, Na4; rtol = 5.0e-3)
# ρ₃ is symmetric under permuting the three cluster masses (any particle may be
# peeled first, Eq. rho3); the constructor does NOT sort them, so permuting the
# inputs and getting the same spectrum genuinely tests the inner σ-integral.
v0 = d4(Fa4, 0.5, 0.3, 0.2)
dev = maximum(
    abs(FourBodyDecay(1.3, 0.1396, p...)(Fa4, 0.5, 0.3, 0.2) / v0 - 1)
        for p in ([0.135, 0.1396, 0.1396], [0.1396, 0.135, 0.1396])
)
@printf("   cluster-mass permutation: max rel dev = %.1e\n", dev)
@assert dev < 1.0e-3
println("   PASS")

# ---------------------------------------------------------------- test 3c
println("== 3c. Finite parent width: BroadParent (Breit–Wigner average) ==")
let mρ = 0.7752, mπ = 0.1396, mp = 0.938, g = collect(range(0, 6, length = 300))
    # functor number conservation (ρ→ππ, ρ broad) — the BroadParent as a functor
    ρw = BroadParent(mπ, mπ, BreitWigner(mρ, 0.149; mmin = 2mπ); B = 1.0, nma = 8, nq = 64, nθ = 16)
    Faw = thermal(mρ)
    Nbw = 2π * quadgk(
        pT -> quadgk(ηp -> pT * ρw(Faw, pT, ηp, 0.0), -Inf, Inf; rtol = 2.0e-3)[1],
        0, Inf; rtol = 2.0e-3
    )[1]
    @printf("   functor N_b/(B N_a) = %.5f\n", Nbw / number(Faw))
    @assert isapprox(Nbw, number(Faw); rtol = 5.0e-3)
    # matrix: a single node at the pole reproduces the sharp matrix
    Wsharp = decaymatrices(TwoBodyDecay(mρ, mπ, mπ), [0], g, g)[1]
    W1 = decaymatrices(BroadParent(mπ, mπ, BreitWigner(mρ, 1.0e-3); nma = 1), [0], g, g)[1]
    @assert maximum(abs.(W1 .- Wsharp)) < 1.0e-3 * maximum(abs.(Wsharp))
    # narrow width → sharp spectrum (number-weighted)
    Fa = [exp(-hypot(mρ, q) / T) for q in g]
    relw(F, R) = sum(abs.(F .- R) .* g) / sum(abs.(R) .* g)
    dn = relw(
        (decaymatrices(BroadParent(mπ, mπ, BreitWigner(mρ, 1.0e-3; mmin = 2mπ); nma = 24), [0], g, g)[1]) * Fa,
        Wsharp * Fa
    )
    @assert dn < 5.0e-3
    # channel reopening: pole below the (π,p) threshold, spectral tail above it
    clsd = maximum(abs.(decaymatrices(BroadParent(mπ, mp, BreitWigner(1.0, 1.0e-3); nma = 32), [0], g, g)[1]))
    opnd = maximum(abs.(decaymatrices(BroadParent(mπ, mp, BreitWigner(1.0, 0.2); nma = 32), [0], g, g)[1]))
    @assert clsd == 0.0 && opnd > 0.0
    # threshold-cut window: the parent-mass quadrature must sample the OPEN part.
    # Sampling the full window and discarding closed nodes left a narrowly-open
    # channel under-sampled — or with no node at all: nma=8 against a threshold at
    # 1.3 GeV with support to 1.4 GeV caught zero nodes and returned a zero matrix.
    let bwc = BreitWigner(1.0, 0.1)                    # window [0.6, 1.4]
        W8 = decaymatrices(BroadParent(0.65, 0.65, bwc; nma = 8), [0], g, g)[1]
        W32 = decaymatrices(BroadParent(0.65, 0.65, bwc; nma = 32), [0], g, g)[1]
        ref = decaymatrices(BroadParent(0.65, 0.65, bwc; nma = 96), [0], g, g)[1]
        @assert maximum(abs.(W8)) > 0                  # was exactly 0.0 (silent zero)
        r8 = sum(abs.(W8 .- ref)) / sum(abs.(ref))   # narrow open sliver [1.3, 1.4]:
        r32 = sum(abs.(W32 .- ref)) / sum(abs.(ref))   # converges, just not fast
        @assert r32 < r8 && r32 < 0.1
        @printf("   threshold-cut window: nma=8 non-zero, rel dev %.2f→%.2f (nma 8→32)\n", r8, r32)
    end
    # fully closed: the channel threshold sits above the whole spectral window, so
    # parentmassnodes returns no node and the MassAverage carries empty arrays.
    # Both evaluation paths must agree on zero — the functor used to throw on the
    # empty reduction while the matrix path returned all-zero.
    let dclosed = BroadParent(0.6, 0.6, BreitWigner(1.0, 0.05; mmin = 0.85, mmax = 1.15))
        @assert isempty(dclosed.coef)                  # threshold 1.20 > mmax 1.15
        Wc = decaymatrices(dclosed, [0, 2], g, g)
        @assert all(all(iszero, W) for W in Wc)
        @assert dclosed(thermal(1.0), 0.9, 0.2, 0.3) === 0.0    # was: ArgumentError
        @printf("   fully-closed channel: matrix all-zero and functor 0.0 (agree)\n")
    end
    @printf("   nma=1==sharp, narrow→sharp %.1e, reopening (closed=0, open≠0): OK\n", dn)
    println("   PASS")
end

# ---------------------------------------------------------------- test 3d
println("== 3d. SpectralShape: generic parentmassnodes reproduces the BreitWigner fast path ==")
let rules = Rules(nma = 16)
    # The generic path weights ψ nodes by jacobian(s)·ϱ(√s); for a constant-
    # width BreitWigner that product is √s/(2 mR), the fast path's weight times
    # a constant. The constant cancels in BroadParent's weight/normalization
    # ratio, so the two paths must build identical decays. `invoke` forces the
    # generic method for a shape whose dispatch target is the specialized one.
    for (bw, threshold) in (
            (BreitWigner(0.7752, 0.149; mmin = 2 * 0.1396), 2 * 0.1396),
            (BreitWigner(0.475, 0.55; mmin = 0.05), 2 * 0.1396),   # threshold cuts in
            (BreitWigner(1.0, 0.2), 1.0777),                       # reopened: pole below cut
        )
        fast = DecaySpectra.parentmassnodes(bw, rules.ma, threshold)
        generic = invoke(
            DecaySpectra.parentmassnodes,
            Tuple{SpectralShape, DecaySpectra.MassRule, Any},
            bw, rules.ma, threshold,
        )
        @assert fast[1] == generic[1]                        # identical mass nodes
        @assert length(fast[2]) == length(generic[2])
        if !isempty(fast[2])
            # weight/normalization is the physical quantity; 2 mR cancels there
            fast_ratios = fast[2] ./ fast[3]
            generic_ratios = generic[2] ./ generic[3]
            deviation = maximum(abs.(fast_ratios .- generic_ratios) ./
                                max.(abs.(fast_ratios), 1.0e-300))
            @assert deviation < 1.0e-12 "generic path deviates: $deviation"
        end
    end
    # and through BroadParent itself: the constant-width decay is unchanged
    ρfast = BroadParent(0.1396, 0.1396, BreitWigner(0.7752, 0.149; mmin = 2 * 0.1396), rules)
    @assert !isempty(ρfast.coef)

    # Exact closure. BroadParent scales each node by weight/normalization, so
    # daughters per decay is sum(weights)/normalization — an *identity* whenever
    # the open part is the whole support, for every shape and either rule. It
    # was 1 - 8.9e-5 while the normalization was swept on Gauss-Legendre and the
    # open nodes on the cosine rule, which is a silent loss of a guarantee the
    # specialized method has always had.
    for shape in (
            BreitWigner(0.7752, 0.1491; mmin = 2 * 0.1396),          # no cut, GL
            MassDependentBW(0.7752, 0.1491, 2 * 0.1396),             # no cut, cosine
            MassDependentBW(0.475, 0.55, 2 * 0.1396),                # sigma
            MassDependentBW(1.232, 0.117, 1.0777),                   # Delta
        )
        _, weights, normalization =
            DecaySpectra.parentmassnodes(shape, rules.ma, DecaySpectra.support(shape)[1])
        @assert sum(weights) == normalization "closure is not exact: $(sum(weights)/normalization)"
    end
    # When the threshold does cut, the deficit is the physics, not a defect:
    # the closed part normalizes in and emits nothing (channel reopening).
    let leaky = BreitWigner(0.475, 0.55; mmin = 0.05)
        _, weights, normalization =
            DecaySpectra.parentmassnodes(leaky, rules.ma, 2 * 0.1396)
        @assert 0.95 < sum(weights) / normalization < 0.99
    end
    println("   generic ≡ fast on nodes to 1e-12; closure exact when nothing is cut: PASS")
end

# ---------------------------------------------------------------- test 3e
println("== 3e. MassDependentBW: threshold, sharp limit, convergence, closure ==")
let mπ = 0.1396, mρ = 0.7752, g = collect(range(0, 6, length = 300))
    # (a) The shape is exactly zero below its threshold and positive above.
    σshape = MassDependentBW(0.475, 0.55, 2mπ)
    @assert σshape(2mπ - 1.0e-9) == 0.0 && σshape(2mπ) == 0.0
    @assert σshape(0.3) > 0 && σshape(0.475) > 0
    # mth = 0 degenerates to a constant width: Γ(m) = Γ0 exactly, so the shape
    # equals m²Γ0/((m²−mR²)² + m²Γ0²) — check against the explicit formula.
    flat = MassDependentBW(1.0, 0.2, 0.0; mmin = 0.5, mmax = 1.5)
    for m in (0.6, 1.0, 1.4)
        @assert isapprox(flat(m), m^2 * 0.2 / ((m^2 - 1)^2 + m^2 * 0.04); rtol = 1.0e-14)
    end
    # a pole at or below threshold must be rejected, not silently misweighted
    caught = try MassDependentBW(1.0, 0.2, 1.0); false catch e; e isa ArgumentError end
    @assert caught

    # (b) Γ0 → 0 recovers the sharp two-body matrix (number-weighted), through
    # the generic parentmassnodes path and its cosine threshold rule.
    Wsharp = decaymatrices(TwoBodyDecay(mρ, mπ, mπ), [0], g, g)[1]
    Wnarrow = decaymatrices(
        BroadParent(mπ, mπ, MassDependentBW(mρ, 1.0e-3, 2mπ); nma = 24), [0], g, g)[1]
    Fa = [exp(-hypot(mρ, q) / 0.145) for q in g]
    relw(F, R) = sum(abs.(F .- R) .* g) / sum(abs.(R) .* g)
    dnarrow = relw(Wnarrow * Fa, Wsharp * Fa)
    @assert dnarrow < 5.0e-3

    # (c) nma convergence on the σ, the broadest state in any table.
    Wref = decaymatrices(BroadParent(mπ, mπ, MassDependentBW(0.475, 0.55, 2mπ); nma = 96), [0], g, g)[1]
    r16 = relw(decaymatrices(BroadParent(mπ, mπ, MassDependentBW(0.475, 0.55, 2mπ); nma = 16), [0], g, g)[1] * Fa, Wref * Fa)
    r32 = relw(decaymatrices(BroadParent(mπ, mπ, MassDependentBW(0.475, 0.55, 2mπ); nma = 32), [0], g, g)[1] * Fa, Wref * Fa)
    @assert r32 < r16 && r32 < 5.0e-3

    # (d) Defect-1 closure. A constant-width σ normalizes in below-threshold
    # weight that emits nothing, so daughters-per-decay falls short of the
    # branching ratios; the threshold-vanishing shape has no such weight. Count
    # daughters per parent through the functor in a static thermal cell.
    #
    # The constant shape uses mmin = 0.05, the floor `decayfor` applies in
    # production — NOT mmin = 2mπ, which would pre-cut the leak this test
    # exists to demonstrate (the plan doc's 2.56% of the σ's weight).
    Faσ = thermal(0.475)
    countdaughters(shape) = 2π * quadgk(
        pT -> quadgk(ηp -> pT * BroadParent(mπ, mπ, shape; B = 1.0, nma = 16,
                                            nq = 64, nθ = 16)(Faσ, pT, ηp, 0.0),
                     -Inf, Inf; rtol = 2.0e-3)[1],
        0, Inf; rtol = 2.0e-3)[1] / number(Faσ)
    leaked = countdaughters(BreitWigner(0.475, 0.55; mmin = 0.05))
    closed = countdaughters(MassDependentBW(0.475, 0.55, 2mπ))
    @printf("   daughters/decay: constant-Γ %.4f (leaks ~2.5%%), Γ(m) %.4f (must be 1)\n",
            leaked, closed)
    @assert abs(closed - 1) < 1.0e-2
    @assert closed - leaked > 0.015   # the leak is the 2.56% the plan doc measured
    println("   PASS")
end

# ---------------------------------------------------------------- test 4
println("== 4. Gauss–Legendre convergence: default orders vs doubled ==")
ρfine = TwoBodyDecay(0.7752, 0.1396, 0.1396; nq = 256, nθ = 64)
ωfine = ThreeBodyDecay(0.7827, 0.1396, 0.1396, 0.135; nµ = 64, nq = 256, nθ = 64)
for (pT, ηp) in ((0.2, 0.0), (0.3, 0.0), (1.2, 0.8))
    r2 = abs(ρ(Fa, pT, ηp, 0.4) / ρfine(Fa, pT, ηp, 0.4) - 1)
    r3 = abs(ω(Faω, pT, ηp, 0.4) / ωfine(Faω, pT, ηp, 0.4) - 1)
    @printf(
        "   pT=%.2f ηp=%+.1f:  2-body rel.diff=%.1e   3-body rel.diff=%.1e\n",
        pT, ηp, r2, r3
    )
    @assert r2 < 5.0e-7 && r3 < 5.0e-7
end
println("   PASS")

# ---------------------------------------------------------------- test 5
println("== 5. LagrangeWeights: exactness, explicit vs generic, O(h^P) order ==")
using FastReso2.DecaySpectra.LagrangeWeights
linterp(x, F, x1, h, P) = begin
    j0, w = lagrangeweights(x, x1, h, length(F), P)
    sum(w[m] * F[j0 + m - 1] for m in eachindex(w))
end
for P in 2:7                      # partition of unity + node exactness
    local F = rand(21)
    for x in (0.537, 0.013, 1.987)
        @assert abs(sum(lagrangeweights(x, 0.0, 0.1, 21, P)[2]) - 1) < 1.0e-12
    end
    for i in 3:19
        @assert abs(linterp((i - 1) * 0.1, F, 0.0, 0.1, P) - F[i]) < 1.0e-11
    end
end
gw(ξ, P) = ntuple(m -> prod((ξ - k) / (m - 1 - k) for k in 0:(P - 1) if k != m - 1), P)
for P in 2:4, ξ in 0.0:0.13:(P - 1)   # explicit formulas == generic product
    @assert all(abs.(collect(LagrangeWeights.lweights(ξ, Val(P))) .- collect(gw(ξ, P))) .< 1.0e-12)
end
xs5 = [2 * (k / 20)^2 for k in 0:20]  # non-uniform (stretched) grid version
for P in 2:6
    for x in (0.017, 0.31, 1.93)      # partition of unity
        @assert abs(sum(lagrangeweights(x, xs5, P)[2]) - 1) < 1.0e-12
    end
    local poly = x -> sum(x^d * (0.3 + 0.1d) for d in 0:(P - 1))
    local F = poly.(xs5)              # exact for degree < P on any spacing
    for x in 0.011:0.0173:1.99
        j0, w = lagrangeweights(x, xs5, P)
        @assert abs(sum(w[m] * F[j0 + m - 1] for m in 1:P) - poly(x)) < 1.0e-10
    end
end
for P in (2, 3, 4, 6), x in (0.537, 0.013, 1.987)   # vector path == uniform path
    j0a, wa = lagrangeweights(x, 0.0, 0.1, 21, P)
    j0b, wb = lagrangeweights(x, collect(0.0:0.1:2.0), P)
    @assert j0a == j0b && all(abs.(collect(wa) .- collect(wb)) .< 1.0e-12)
end
f5(x) = exp(-x) * cos(3x)
for P in (2, 4, 6)                # convergence order ~ 2^P per halving of h
    local errs = map((0.1, 0.05)) do h
        local N = round(Int, 2.0 / h) + 1
        local F = [f5((i - 1) * h) for i in 1:N]
        maximum(abs(linterp(x, F, 0.0, h, P) - f5(x)) for x in 0.001:0.00373:1.995)
    end
    ratio = errs[1] / errs[2]
    @printf(
        "   P=%d:  error(h=0.1)=%.2e  ratio per halving=%.1f  (theory %d)\n",
        P, errs[1], ratio, 2^P
    )
    @assert ratio > 0.7 * 2^P
end
# derivative weights: sum-to-zero, polynomial-derivative exactness, non-uniform
dinterp(x, F, x1, h, P) = begin
    j0, w = lagrangederiv(x, x1, h, length(F), P)
    sum(w[m] * F[j0 + m - 1] for m in eachindex(w))
end
for P in 2:6
    @assert abs(sum(lagrangederiv(0.537, 0.0, 0.1, 21, P)[2])) < 1.0e-11   # d(const)=0
    local poly = x -> sum(x^d * (0.3 + 0.1d) for d in 0:(P - 1))
    local dpoly = x -> sum(d * x^(d - 1) * (0.3 + 0.1d) for d in 1:(P - 1))
    local Fu = poly.(0.0:0.05:2.0)
    @assert maximum(abs(dinterp(x, Fu, 0.0, 0.05, P) - dpoly(x)) for x in 0.05:0.03:1.9) < 1.0e-8
    local Fn = poly.(xs5)                     # non-uniform (stretched) grid
    @assert maximum(
        begin
                j0, w = lagrangederiv(x, xs5, P)
                abs(sum(w[m] * Fn[j0 + m - 1] for m in 1:P) - dpoly(x))
            end
            for x in 0.05:0.03:1.9
    ) < 1.0e-7
end
# second derivative: zero-sum, exactness (needs P ≥ 3), lagrangeall consistency
d2interp(x, F, x1, h, P) = begin
    j0, w = lagrangederiv2(x, x1, h, length(F), P)
    sum(w[m] * F[j0 + m - 1] for m in eachindex(w))
end
@assert all(lagrangederiv2(0.5, 0.0, 0.1, 21, Val(2))[2] .== 0.0)      # line: no curvature
for P in 3:6
    @assert abs(sum(lagrangederiv2(0.537, 0.0, 0.1, 21, P)[2])) < 1.0e-9
    local poly = x -> sum(x^d * (0.3 + 0.1d) for d in 0:(P - 1))
    local d2poly = x -> sum(d * (d - 1) * x^(d - 2) * (0.3 + 0.1d) for d in 2:(P - 1))
    local Fu = poly.(0.0:0.05:2.0)
    @assert maximum(abs(d2interp(x, Fu, 0.0, 0.05, P) - d2poly(x)) for x in 0.06:0.03:1.9) < 1.0e-6
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
d2 = maximum(
    abs.(
        M2(q -> FBIρ(q)) ./
            [ρ((q, η, ϕ) -> FBIρ(q) * cos(2ϕ), pT, 0.0, 0.0) for pT in pT6] .- 1
    )
)
dω = maximum(abs.(Mω(FBIω) ./ [ω((q, η, ϕ) -> FBIω(q), pT, 0.0, 0.0) for pT in pT6] .- 1))
@printf("   max rel dev:  ρ n=0 %.1e   ρ n=2 %.1e   ω n=0 %.1e\n", d0, d2, dω)
@assert d0 < 5.0e-4 && d2 < 5.0e-4 && dω < 5.0e-4
# Float64 ranges stay compact and select the uniform stencil. An equivalent
# vector takes the general non-uniform stencil, whose last bits may differ.
@assert M0.pT === pT6 && M0.q === qg6
qv6 = collect(qg6)
Mv = DecayMatrix(ρ, 0, pT6, qv6)
@assert Mv.q isa Vector{Float64} && Mv.q == qv6 && Mv.q !== qv6
@assert maximum(abs.(M0.W .- Mv.W)) < 1.0e-10 * maximum(abs.(M0.W))
# A genuinely non-uniform grid remains supported and is stored as a copied vector.
qn6 = [0.0, 0.03, 0.11, 0.26, 0.5, 0.85, 1.3, 2.0, 3.0, 4.2]
Mn = DecayMatrix(ρ, 0, range(0.1, 0.3, length = 3), qn6)
@assert Mn.q isa Vector{Float64} && Mn.q == qn6 && Mn.q !== qn6
@assert size(Mn.W) == (3, length(qn6)) && all(isfinite, Mn.W)
println("   PASS")

# ---------------------------------------------------------------- test 7
println("== 7. CascadeTables: keys, cascade vs direct, round trip, updates ==")
include(joinpath(REPO, "tools", "convert_decay_data.jl"))
using .DecayDataConverter

# key invariants: partner order canonicalized, observed slot preserved
@assert ChannelKey(1.0, 0.2, [0.5, 0.3]) == ChannelKey(1.0, 0.2, [0.3, 0.5])
@assert ChannelKey(1.0, 0.2, [0.5]).md == 0.0
@assert ChannelKey(1.0, 0.2, [0.5]) != ChannelKey(1.0, 0.5, [0.2])
@assert (@inferred ChannelKey(2, 1.0f0, 0.2, 0, 0, 0.3f0, 0, 0)) isa ChannelKey
@assert (@inferred ChannelKey(1.0f0, 0.2, Float32[0.5, 0.3])) isa ChannelKey
for invalid_key in (
        () -> ChannelKey(9, 1.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0),
        () -> ChannelKey(2, 1.0, 0.2, -0.1, 0.0, 0.3, 0.0, 0.0),
        () -> ChannelKey(2, 1.0, 0.2, 0.0, 0.3, 0.3, 0.0, 0.0),
        () -> ChannelKey(3, 1.0, 0.2, 0.1, 0.3, 0.5, 0.4, 0.0),
    )
    threw = false
    try
        invalid_key()
    catch error
        threw = error isa ArgumentError
    end
    @assert threw
end

# RowBandMatrix keeps one contiguous source-column range per output row.
# Offsets delimit each row in the flat payload and make scalar access O(1).
# Packed matrices are read-only; generic array operations allocate dense output.
let
    dense = Float32[
        0 1 0 2 0
        3 0 0 0 0
        0 0 0 0 0
        0 0 4 5 6
    ]
    packed = CascadeTables.RowBandMatrix(dense)

    @assert packed isa AbstractMatrix{Float32}
    @assert size(packed) == size(dense)
    @assert eltype(packed) == Float32
    @assert packed.column_starts == UInt8[2, 1, 1, 3]
    @assert packed.value_offsets == UInt16[1, 4, 5, 5, 8]
    @assert [
        CascadeTables._rowband(packed, row)
            for row in axes(packed, 1)
    ] == [
        2:4,
        1:1,
        1:0,
        3:5,
    ]
    @assert packed.values == Float32[1, 0, 2, 3, 4, 5, 6]
    @assert packed[1, 1] == 0
    @assert packed[1, 2] == 1
    @assert packed[1, 3] == 0
    @assert packed[4, 5] == 6
    @assert packed[:, 3] == dense[:, 3]
    @assert sum(packed) == sum(dense)
    @assert Matrix(packed) == dense
    @assert Matrix(copy(packed)) == dense

    dense_similar = similar(packed)
    @assert dense_similar isa Matrix{Float32}
    @assert size(dense_similar) == size(packed)
    copyto!(dense_similar, packed)
    @assert dense_similar == dense

    zero_matrix = zero(packed)
    @assert zero_matrix isa Matrix{Float32}
    @assert size(zero_matrix) == size(packed)
    @assert all(iszero, zero_matrix)

    mapped_matrix = map(value -> value + 1, packed)
    @assert mapped_matrix isa Matrix{Float32}
    @assert mapped_matrix == dense .+ 1

    similar_float64 = similar(packed, Float64)
    @assert similar_float64 isa Matrix{Float64}
    @assert size(similar_float64) == size(packed)

    resized_similar = similar(packed, Float64, (3, 7))
    @assert resized_similar isa Matrix{Float64}
    @assert size(resized_similar) == (3, 7)

    mutation_threw = false
    try
        packed[3, 5] = 7
    catch
        mutation_threw = true
    end
    @assert mutation_threw
    @assert Matrix(packed) == dense

    big_packed = CascadeTables.RowBandMatrix(
        BigInt[
            0 1 0
            2 0 0
        ]
    )
    @assert map(value -> value + 1, big_packed) ==
        Matrix(big_packed) .+ 1

    packed32 = CascadeTables.RowBandMatrix(dense, UInt32)
    @assert eltype(packed32.column_starts) == UInt32
    @assert eltype(packed32.value_offsets) == UInt16
    @assert Matrix(packed32) == dense

    packed32_offsets =
        CascadeTables.RowBandMatrix(dense, UInt32, UInt32)
    @assert eltype(packed32_offsets.column_starts) == UInt32
    @assert eltype(packed32_offsets.value_offsets) == UInt32
    @assert Matrix(packed32_offsets) == dense

    special_values = Float32[
        0 -0.0 0
        NaN Inf 0
    ]
    special_packed =
        CascadeTables.RowBandMatrix(special_values)
    @assert isequal(Matrix(special_packed), special_values)

    no_columns =
        CascadeTables.RowBandMatrix(zeros(Float32, 2, 0))
    no_rows =
        CascadeTables.RowBandMatrix(zeros(Float32, 0, 3))
    @assert size(Matrix(no_columns)) == (2, 0)
    @assert size(Matrix(no_rows)) == (0, 3)
    @assert isempty(no_columns.values)
    @assert isempty(no_rows.values)
    @assert no_columns.value_offsets == UInt16[1, 1, 1]
    @assert no_rows.value_offsets == UInt16[1]

    dimension_typed =
        CascadeTables.RowBandMatrix(zeros(Float32, 256, 256))
    @assert eltype(dimension_typed.column_starts) == UInt16
    @assert eltype(dimension_typed.value_offsets) == UInt32

    let threw = false
        try
            CascadeTables.RowBandMatrix(
                zeros(Float32, 1, 256),
                UInt8,
            )
        catch error
            threw = error isa ArgumentError
        end
        @assert threw
    end

    let threw = false
        try
            CascadeTables.RowBandMatrix(
                ones(Float32, 1, 255),
                UInt16,
                UInt8,
            )
        catch error
            threw = error isa ArgumentError
        end
        @assert threw
    end

    for invalid_constructor in (
            () -> CascadeTables.RowBandMatrix(
                -1,
                UInt16[],
                UInt16[1],
                Float32[],
            ),
            () -> CascadeTables.RowBandMatrix(
                3,
                UInt16[0],
                UInt16[1, 3],
                zeros(Float32, 2),
            ),
            () -> CascadeTables.RowBandMatrix(
                3,
                UInt16[3],
                UInt16[1, 3],
                zeros(Float32, 2),
            ),
            () -> CascadeTables.RowBandMatrix(
                3,
                UInt16[1],
                UInt16[1, 2],
                zeros(Float32, 2),
            ),
            () -> CascadeTables.RowBandMatrix(
                3,
                UInt16[1, 1],
                UInt16[1, 3, 2],
                zeros(Float32, 1),
            ),
            () -> CascadeTables.RowBandMatrix(
                typemax(Int),
                Int[typemax(Int)],
                UInt16[1, 3],
                zeros(Float32, 2),
            ),
            () -> CascadeTables.RowBandMatrix(
                3,
                UInt64[typemax(UInt64)],
                UInt16[1, 2],
                zeros(Float32, 1),
            ),
            () -> CascadeTables.RowBandMatrix(
                3,
                UInt8[1, 1],
                UInt64[1, typemax(UInt64), 2],
                zeros(Float32, 1),
            ),
        )
        threw = false
        try
            invalid_constructor()
        catch error
            threw = error isa ArgumentError
        end
        @assert threw
    end
end

# The canonical reader resolves independently loaded particle and decay files.
# AZYHYDRO parsing belongs to the standalone converter and round-trips through
# that one runtime format.
let
    combined_path = tempname()
    converted_particle_path = tempname()
    converted_decay_path = tempname()
    particle_path = tempname()
    decay_path = tempname()
    empty_particle_path = tempname()
    empty_decay_path = tempname()
    unresolved_decay_path = tempname()
    write(
        combined_path,
        "1 P 1.0 0.1 3 0 0 0 0 0 0 1\n" *
            "1 2 1.0 2 3 0 0 0\n" *
            "2 a 0.1 0.0 1 0 0 0 0 0 1 0\n" *
            "3 b 0.1 0.0 1 0 0 0 0 0 -1 0\n",
    )
    write(
        particle_path,
        "1 P 1.0 0.1 3 0 0 0 0 0 0 parent-alias\n" *
            "2 a 0.1 0.0 1 0 0 0 0 0 1\n" *
            "3 b 0.1 0.0 1 0 0 0 0 0 -1\n",
    )
    write(decay_path, "1 2 1.0 2 3\n")
    write(empty_particle_path, "")
    write(empty_decay_path, "")
    write(unresolved_decay_path, "1 2 1.0 2 999\n")

    catalog = readparticles(particle_path)
    network = readdecays(decay_path)
    split = DecayList(catalog, network)
    empty_split = DecayList(
        readparticles(empty_particle_path),
        readdecays(empty_decay_path),
    )
    convert_azyhydro(
        [combined_path],
        converted_particle_path,
        converted_decay_path,
    )
    threw = false
    try
        convert_azyhydro(
            [combined_path],
            converted_particle_path,
            converted_decay_path,
        )
    catch error
        threw = error isa ArgumentError &&
            occursin("--force", sprint(showerror, error))
    end
    @assert threw
    original_combined = read(combined_path, String)
    threw = false
    try
        convert_azyhydro(
            [combined_path],
            combined_path,
            tempname();
            force = true,
        )
    catch error
        threw = error isa ArgumentError &&
            occursin("overwrite an input", sprint(showerror, error))
    end
    @assert threw
    @assert read(combined_path, String) == original_combined
    converted = DecayList(
        readparticles(converted_particle_path),
        readdecays(converted_decay_path),
    )
    @assert catalog isa ParticleCatalog
    @assert network isa DecayNetwork
    @assert catalog.path == particle_path
    @assert network.path == decay_path
    @assert !isempty(catalog.sha)
    @assert !isempty(network.sha)
    @assert network.decaying_ids == [1]
    unresolved_network = readdecays(unresolved_decay_path)
    @assert haskey(unresolved_network.byparent, 1)
    threw = false
    try
        DecayList(catalog, unresolved_network)
    catch error
        threw = error isa ArgumentError &&
            occursin("1 -> 999", sprint(showerror, error))
    end
    @assert threw

    @assert split.parts == converted.parts
    channel_rows(decay_list) = Dict(
        parent_id => [
            (channel.parent, channel.br, Tuple(channel.daughters))
                for channel in channels
        ]
        for (parent_id, channels) in decay_list.byparent
    )
    @assert channel_rows(split) == channel_rows(converted)
    @assert split.stable_ids == converted.stable_ids
    @assert empty_split.path ==
        string(empty_particle_path, " + ", empty_decay_path)
    @assert empty_split.sha == string(
        empty_split.catalog.sha,
        "+",
        empty_split.network.sha,
    )
    unprovenanced = DecayList(
        ParticleCatalog(Dict{Int, ParticleInfo}()),
        DecayNetwork(Dict{Int, Vector{DecayChannel}}()),
    )
    merged_unprovenanced = merge(unprovenanced, unprovenanced)
    @assert isempty(merged_unprovenanced.path)
    @assert isempty(merged_unprovenanced.sha)

    self_terminal_network = DecayNetwork(Dict(
        1 => [DecayChannel(1, 1.0, [1])],
    ))
    threw = false
    try
        DecayNetwork(
            self_terminal_network.byparent;
            decaying_ids = [1],
        )
    catch error
        threw = error isa ArgumentError &&
            occursin("both decaying and terminal", sprint(showerror, error))
    end
    @assert threw

    filtered_network = DecayNetwork(
        Dict{Int, Vector{DecayChannel}}();
        decaying_ids = [1],
    )
    for merged in (
            merge(filtered_network, self_terminal_network),
            merge(self_terminal_network, filtered_network),
        )
        @assert 1 in merged.decaying_ids
        @assert !haskey(merged.byparent, 1)
    end
    for merged in (
            merge(network, self_terminal_network),
            merge(self_terminal_network, network),
        )
        @assert merged.byparent[1] == network.byparent[1]
        @assert 1 in merged.decaying_ids
    end
    @assert split.stable_ids == converted.stable_ids == [2, 3]

    tiny_grid = range(0.0, 1.5, length = 9)
    tiny_table = CascadeTable(
        converted;
        grid = tiny_grid,
        modes = (0,),
        orders = (nq = 4, nθ = 4, nµ = 4, nσ = 4, nma = 4),
        width_cut = Inf,
    )
    combined_plan = CascadePlan(tiny_table, converted)
    split_plan = CascadePlan(tiny_table, split)
    @assert split_plan.ids == combined_plan.ids
    @assert split_plan.particles == combined_plan.particles
    @assert split_plan.stable_indices == combined_plan.stable_indices
    @assert [(step.parent, step.groups) for step in split_plan.steps] ==
        [(step.parent, step.groups) for step in combined_plan.steps]
    @assert [(group.matrices, group.targets) for group in split_plan.groups] ==
        [(group.matrices, group.targets) for group in combined_plan.groups]
    @assert [(target.daughter, target.weight) for target in split_plan.targets] ==
        [(target.daughter, target.weight) for target in combined_plan.targets]
    tiny_primary(particle_info, _mode, grid) =
        [(1 + particle_info.mass) * exp(-momentum) for momentum in grid]
    combined_result = runcascade(combined_plan; primary = tiny_primary)
    split_result = runcascade(split_plan; primary = tiny_primary)
    @assert split_result.F == combined_result.F
    @assert split_result.Fprim == combined_result.Fprim

    @assert particle(catalog, 1).name == "P"
    @assert particle(catalog, "P").id == 1
    @assert particle(catalog, "parent-alias").id == 1
    @assert particleids(catalog, "parent-alias") == [1]
    @assert particle(split, 1).name == "P"
    @assert particle(split, "P").id == 1
    @assert particleids(split, "P") == [1]
    @assert spin(split.parts[1]) == 1
    @assert split.parts[1].bottomness == 0
    split_stats = validate(split; verbose = false)
    @assert split_stats.selfstable == 2
    @assert split_stats.selfchannels == 0
    filtered_split = filter(nbody(2), split)
    @assert filtered_split.path == split.path
    @assert filtered_split.sha == split.sha
    fully_filtered_stats = validate(filter(nbody(3), split); verbose = false)
    @assert fully_filtered_stats.channels == 0
    @assert fully_filtered_stats.br_norm_outliers == 1
    @assert fully_filtered_stats.br_maxdev == 1.0
    bottom_catalog = ParticleCatalog(Dict(
        1 => ParticleInfo(1, "B", 1.0, 0.0, 1.0, 0, 0, 0, 0, -1, 0),
        2 => split.parts[2],
        3 => split.parts[3],
    ))
    bottom_model = DecayList(bottom_catalog, network)
    @assert validate(bottom_model; verbose = false).bottomness_changing == 1

    alias_catalog = ParticleCatalog(Dict(
        1 => ParticleInfo(1, "Parent", 1.0, 0.1, 3.0, 0, 0),
    ))
    merged_catalog = merge(catalog, alias_catalog)
    @assert merged_catalog.byid[1].name == "P"
    @assert particleids(merged_catalog, "P") == [1]
    @assert particleids(merged_catalog, "Parent") == [1]
    alias_split = DecayList(merged_catalog, network)
    @assert isempty(filter(fromparent("Parent"), alias_split).byparent)
    @assert haskey(
        filter(fromid(particleids(alias_split, "Parent")...), alias_split).byparent,
        1,
    )

    ambiguous_catalog = merge(
        merged_catalog,
        ParticleCatalog(Dict(
            4 => ParticleInfo(4, "P", 0.2, 0.0, 1.0, 0, 0),
        )),
    )
    @assert particleids(ambiguous_catalog, "P") == [1, 4]
    ambiguous = false
    try
        particle(ambiguous_catalog, "P")
    catch error
        ambiguous = error isa ArgumentError &&
            occursin("ambiguous", sprint(showerror, error))
    end
    @assert ambiguous

    unresolved = DecayNetwork(Dict(
        1 => [DecayChannel(1, 1.0, [2, 99])],
    ))
    threw = false
    try
        DecayList(catalog, unresolved)
    catch error
        threw = error isa ArgumentError &&
            occursin("1 -> 99", sprint(showerror, error))
    end
    @assert threw

    bypassed = false
    try
        DecayList(catalog, unresolved, [99], "bad", "bad")
        bypassed = true
    catch error
        @assert error isa MethodError
    end
    @assert !bypassed
    for invalid_constructor in (
            () -> ParticleCatalog(catalog.byid, Dict{String, Vector{Int}}(), "", ""),
            () -> DecayNetwork(network.byparent, Int[], "", ""),
        )
        bypassed = false
        try
            invalid_constructor()
            bypassed = true
        catch error
            @assert error isa MethodError
        end
        @assert !bypassed
    end

    invalid_one_body = DecayNetwork(Dict(
        1 => [DecayChannel(1, 1.0, [2])],
    ))
    threw = false
    try
        DecayList(catalog, invalid_one_body)
    catch error
        threw = error isa ArgumentError &&
            occursin("not a terminal self-channel", sprint(showerror, error))
    end
    @assert threw

    valid_self = DecayList(
        catalog,
        DecayNetwork(Dict(
            2 => [DecayChannel(2, 1.0, [2])],
        )),
    )
    @assert 2 in valid_self.stable_ids
    @assert 2 in filter(nbody(2), valid_self).stable_ids

    unresolved_self = DecayNetwork(Dict(
        99 => [DecayChannel(99, 1.0, [99])],
    ))
    threw = false
    try
        DecayList(catalog, unresolved_self)
    catch error
        threw = error isa ArgumentError &&
            occursin("99 -> 99", sprint(showerror, error))
    end
    @assert threw

    compatibility_model = DecayList(
        catalog.byid,
        network.byparent,
        [3, 2, 2],
        "compatibility",
        "compatibility",
    )
    @assert compatibility_model.stable_ids == [2, 3]

    for requested_stable_ids in ([99], [1, 2, 3])
        threw = false
        try
            DecayList(
                catalog.byid,
                network.byparent,
                requested_stable_ids,
                "compatibility",
                "compatibility",
            )
        catch error
            threw = error isa ArgumentError
        end
        @assert threw
    end

    invalid_self_networks = (
        DecayNetwork(Dict(
            2 => [DecayChannel(2, 0.9, [2])],
        )),
        DecayNetwork(Dict(
            2 => [
                DecayChannel(2, 1.0, [2]),
                DecayChannel(2, 1.0, [2]),
            ],
        )),
        DecayNetwork(Dict(
            1 => [
                DecayChannel(1, 1.0, [1]),
                DecayChannel(1, 1.0, [2, 3]),
            ],
        )),
    )
    for invalid_self in invalid_self_networks
        threw = false
        try
            DecayList(catalog, invalid_self)
        catch error
            threw = error isa ArgumentError &&
                occursin("terminal self-channel", sprint(showerror, error))
        end
        @assert threw
    end

    # Network composition must not erase a malformed self row merely because
    # another layer supplies a physical decaying block for the same parent.
    malformed_terminal = DecayNetwork(Dict(
        2 => [DecayChannel(2, 0.9, [2])],
    ))
    physical_decay = DecayNetwork(Dict(
        2 => [DecayChannel(2, 1.0, [3, 3])],
    ))
    for (left, right) in (
            (malformed_terminal, physical_decay),
            (physical_decay, malformed_terminal),
        )
        threw = false
        try
            merge(left, right)
        catch error
            threw = error isa ArgumentError &&
                occursin(
                    "conflicting channel definitions",
                    sprint(showerror, error),
                )
        end
        @assert threw
    end

    for unsupported_daughters in (
            Int[],
            [2, 3, 2, 3, 2],
            [2, 3, 2, 3, 2, 3],
        )
        unsupported = DecayNetwork(Dict(
            1 => [DecayChannel(1, 1.0, unsupported_daughters)],
        ))
        threw = false
        try
            DecayList(catalog, unsupported)
        catch error
            threw = error isa ArgumentError &&
                occursin("unsupported multiplicity", sprint(showerror, error))
        end
        @assert threw
    end

    nonfinite = DecayNetwork(Dict(
        1 => [DecayChannel(1, NaN, [2, 3])],
    ))
    threw = false
    try
        DecayList(catalog, nonfinite)
    catch error
        threw = error isa ArgumentError &&
            occursin("nonfinite branching ratio", sprint(showerror, error))
    end
    @assert threw

    threw = false
    try
        filter(nbody(2), network)
    catch error
        threw = error isa ArgumentError &&
            occursin("ParticleCatalog", sprint(showerror, error))
    end
    @assert threw

    parent_mismatch = tempname()
    duplicate_header = tempname()
    extra_channel = tempname()
    write(
        parent_mismatch,
        "1 P 1.0 0.1 3 0 0 0 0 0 0 1\n" *
            "9 2 1.0 2 3 0 0 0\n",
    )
    threw = false
    try
        convert_azyhydro(
            [parent_mismatch],
            tempname(),
            tempname(),
        )
    catch error
        threw = error isa ArgumentError &&
            occursin("does not match block parent", sprint(showerror, error))
    end
    @assert threw

    write(
        duplicate_header,
        "1 P 1.0 0.1 3 0 0 0 0 0 0 0\n" *
            "1 P-again 1.0 0.1 3 0 0 0 0 0 0 0\n",
    )
    threw = false
    try
        convert_azyhydro(
            [duplicate_header],
            tempname(),
            tempname(),
        )
    catch error
        threw = error isa ArgumentError &&
            occursin("duplicate particle ID", sprint(showerror, error))
    end
    @assert threw

    write(
        extra_channel,
        "1 P 1.0 0.1 3 0 0 0 0 0 0 0\n" *
            "1 2 1.0 2 3 0 0 0\n",
    )
    threw = false
    try
        convert_azyhydro(
            [extra_channel],
            tempname(),
            tempname(),
        )
    catch error
        threw = error isa ArgumentError &&
            occursin("particle columns", sprint(showerror, error))
    end
    @assert threw

    rm.(
        (
            combined_path,
            converted_particle_path,
            converted_decay_path,
            particle_path,
            decay_path,
            empty_particle_path,
            empty_decay_path,
            unresolved_decay_path,
            parent_mismatch,
            duplicate_header,
            extra_channel,
        );
        force = true,
    )
end

# THERMINATOR uses exact names in decay rows and derives the common particle
# metadata from spin, isospin, and quark content. Duplicate MC codes remain
# discoverable as aliases but never overwrite one another.
let
    particle_path = tempname()
    reordered_particle_path = tempname()
    decay_path = tempname()
    particle_output = tempname()
    decay_output = tempname()
    reordered_particle_output = tempname()
    reordered_decay_output = tempname()
    bad_decay_path = tempname()
    flagged_decay_path = tempname()
    cgc_path = tempname()
    cgc_particle_output = tempname()
    cgc_decay_output = tempname()
    bad_cgc_path = tempname()
    particle_rows = [
        "rho 0.775 0.149 1 1 0 1 0 1 0 0 0 113",
        "pip 0.139 0 0 1 1 1 0 1 0 0 0 211",
        "pim 0.139 0 0 1 -1 1 0 1 0 0 0 -211",
        "omega 0.783 0.008 1 0 0 1 0 1 0 0 0 223",
        "gam 0 0 1 0 0 0 0 0 0 0 0 22",
        "Kplu 0.494 0 0 0.5 0.5 1 0 0 1 0 0 321",
        "Dzero 1.865 0 0 0.5 -0.5 0 0 1 0 1 0 421",
        "Rplus 1.0 0.1 0 1 1 1 0 1 0 0 0 9001",
        "singlet 0.1 0 0 0 0 1 0 1 0 0 0 9002",
        "dupA 1.1 0 0 0 0 1 0 1 0 0 0 999",
        "dupB 1.2 0 0 0 0 1 0 1 0 0 0 999",
    ]
    write(particle_path, join(particle_rows, "\n"))
    write(reordered_particle_path, join(reverse(particle_rows), "\n"))
    write(
        decay_path,
        "# Parent Child1 Child2 [Child3] BR CG\r\n" *
            "rho pip pim 1.0 0\r\n" *
            "omega pip pim gam 0.5 0\r\n",
    )

    convert_therminator(
        particle_path,
        decay_path,
        particle_output,
        decay_output,
    )
    convert_therminator(
        reordered_particle_path,
        decay_path,
        reordered_particle_output,
        reordered_decay_output,
    )
    therminator = DecayList(
        readparticles(particle_output),
        readdecays(decay_output),
    )
    reordered = DecayList(
        readparticles(reordered_particle_output),
        readdecays(reordered_decay_output),
    )
    @assert particle(therminator, "rho").id == 113
    @assert particle(therminator, "rho").degeneracy == 3
    @assert spin(particle(therminator, "rho")) == 1
    @assert particle(therminator, "gam").degeneracy == 2
    @assert spin(particle(therminator, "gam")) == 1
    @assert particle(therminator, "Kplu").strangeness == 1
    @assert particle(therminator, "Kplu").charge == 1
    @assert particle(therminator, "Dzero").charm == 1
    @assert particle(therminator, "Dzero").charge == 0
    duplicate_ids = particleids(therminator, "MC:999")
    @assert length(duplicate_ids) == 2
    @assert 999 ∉ duplicate_ids
    @assert particle(therminator, "dupA").id != particle(therminator, "dupB").id
    @assert particle(reordered, "dupA").id == particle(therminator, "dupA").id
    @assert particle(reordered, "dupB").id == particle(therminator, "dupB").id
    @assert therminator.byparent[113][1].daughters == [211, -211]
    @assert length(therminator.byparent[223][1].daughters) == 3
    @assert therminator.byparent[223][1].br == 0.5

    write(bad_decay_path, "rho pip missing 1.0 0\n")
    threw = false
    try
        convert_therminator(
            particle_path,
            bad_decay_path,
            tempname(),
            tempname(),
        )
    catch error
        threw = error isa ArgumentError &&
            occursin("unknown daughter name", sprint(showerror, error)) &&
            occursin("missing", sprint(showerror, error))
    end
    @assert threw

    write(
        flagged_decay_path,
        "rho pip pim 0.5 1\n" *
            "Rplus singlet pip 0.5 1\n",
    )
    threw = false
    try
        convert_therminator(
            particle_path,
            flagged_decay_path,
            tempname(),
            tempname(),
        )
    catch error
        threw = error isa ArgumentError &&
            occursin("supply --cgc", sprint(showerror, error))
    end
    @assert threw
    write(cgc_path, "# j1 m1 j2 m2 J M CGC\n1 1 1 -1 1 0 0.25\n")
    convert_therminator(
        particle_path,
        flagged_decay_path,
        cgc_particle_output,
        cgc_decay_output;
        cgc_path = cgc_path,
    )
    cgc_model = DecayList(
        readparticles(cgc_particle_output),
        readdecays(cgc_decay_output),
    )
    @assert cgc_model.byparent[113][1].br == 0.25
    # FastReso treats a coefficient absent from CGC.dat as zero.
    @assert cgc_model.byparent[9001][1].br == 0.0
    enrolled_parents = Set{Int}()
    CascadeTables.eachchannelkey(cgc_model) do parent_id, _channel, _daughter, _key
        push!(enrolled_parents, parent_id)
    end
    @assert 9001 ∉ enrolled_parents
    @assert occursin(cgc_path, read(cgc_decay_output, String))
    write(bad_cgc_path, "1 1 1 -1 1 0 NaN\n")
    threw = false
    try
        convert_therminator(
            particle_path,
            flagged_decay_path,
            tempname(),
            tempname();
            cgc_path = bad_cgc_path,
        )
    catch error
        threw = error isa ArgumentError &&
            occursin("coefficient must be finite", sprint(showerror, error))
    end
    @assert threw

    rm.(
        (
            particle_path,
            reordered_particle_path,
            decay_path,
            particle_output,
            decay_output,
            reordered_particle_output,
            reordered_decay_output,
            bad_decay_path,
            flagged_decay_path,
            cgc_path,
            cgc_particle_output,
            cgc_decay_output,
            bad_cgc_path,
        );
        force = true,
    )
end

# synthetic decay list: ρ → ππ (BR 0.6) and ω → ππϖ (BR 1.0)
mρ, mπ, mω, mϖ = 0.7752, 0.1396, 0.7827, 0.135
parts7 = Dict(
    1 => ParticleInfo(1, "rho", mρ, 0.0, 1.0, 0, 0),   # width 0 ⇒ sharp cascade
    2 => ParticleInfo(2, "pi", mπ, 0.0, 1.0, 0, 1),
    3 => ParticleInfo(3, "om", mω, 0.0, 1.0, 0, 0),
    4 => ParticleInfo(4, "pi0", mϖ, 0.0, 1.0, 0, 0)
)
byp7 = Dict(
    1 => [DecayChannel(1, 0.6, [2, 2])],
    3 => [DecayChannel(3, 1.0, [2, 2, 4])]
)
dl7 = DecayList(parts7, byp7, "synthetic", "deadbeef")

g7 = range(0.0, 3.0, length = 61)
ord7 = (nq = 32, nθ = 8, nµ = 8)
tabfile = tempname()
tab7 = CascadeTable(dl7; grid = g7, orders = ord7, modes = [0, 2], cache = tabfile)
@assert length(tab7.W) == 3            # ρ→π|π, ω→π|(πϖ), ω→ϖ|(ππ)
@assert tab7.grid === g7 && tab7.grid isa AbstractRange{Float64}
@assert tab7.modes === (0, 2)
@assert tab7.orders == (nq = 32, nθ = 8, nµ = 8, nσ = 8, nma = 16)
@assert tab7.width_cut == 1.0e-3
@assert tab7.width_window == 4.0
@assert !haskey(tab7.meta, :width_cut)
@assert !haskey(tab7.meta, :width_window)
table_matrix_type7 = typeof(first(first(values(tab7.W))))
@assert table_matrix_type7 ==
    CascadeTables.RowBandMatrix{Float32, UInt8, UInt16}
@assert valtype(typeof(tab7.W)) === NTuple{2, table_matrix_type7}
@assert all(
    typeof(matrix) == table_matrix_type7
        for matrices in values(tab7.W)
        for matrix in matrices
)

# Empty mode sets retain the table's matrix type without running matrix
# assembly, and survive plan execution and cache restoration.
zero_mode_tabfile7 = tempname()
zero_mode_table7 = CascadeTable(
    dl7;
    grid = g7,
    orders = ord7,
    modes = Int[],
    cache = zero_mode_tabfile7,
)
@assert zero_mode_table7.modes === ()
@assert all(==(()), values(zero_mode_table7.W))
zero_mode_plan7 = CascadePlan(zero_mode_table7, dl7)
zero_mode_primary_calls7 = Ref(0)
zero_mode_result7 = runcascade(
    zero_mode_plan7;
    primary = (_particle, _mode, grid) -> begin
        zero_mode_primary_calls7[] += 1
        zeros(length(grid))
    end,
)
@assert zero_mode_primary_calls7[] == 0
@assert all(isempty, values(zero_mode_result7.F))
loaded_zero_mode_table7 = loadtables(zero_mode_tabfile7)
@assert typeof(loaded_zero_mode_table7) === typeof(zero_mode_table7)
@assert loaded_zero_mode_table7.W == zero_mode_table7.W
rm(zero_mode_tabfile7; force = true)

prim7(p, n, grid) = (n == 0 ? 1.0 : 0.05) * [exp(-hypot(p.mass, q) / T) for q in grid]
primary_calls7 = Ref(0)
counted_prim7(p, n, grid) = begin
    primary_calls7[] += 1
    return prim7(p, n, grid)
end
res7 = runcascade(tab7, dl7; primary = counted_prim7)
@assert primary_calls7[] == length(parts7) * length(tab7.modes)
@assert dl7.stable_ids == [2, 4]
@assert [particle.id for particle in allparticles(dl7)] == [1, 2, 3, 4]
@assert [particle.id for particle in stableparticles(dl7)] == [2, 4]
@assert Set(keys(res7.F)) == Set([2, 4])
@assert Set(keys(res7.Fprim)) == Set([2, 4])
@assert Set(keys(res7.F)) ==
    Set(particle.id for particle in stableparticles(dl7))

# Species that cannot reach an output are not asked for a primary: a parent whose
# channels were all filtered away, and a parent feeding only such dead ends.
parts7x = merge(parts7, Dict(5 => ParticleInfo(5, "X", 2mω + 0.1, 0.0, 1.0, 0, 0)))
byp7x = merge(byp7, Dict(5 => [DecayChannel(5, 1.0, [3, 3])]))
dl7x = DecayList(parts7x, byp7x, "synthetic", "deadbeef")
tab7x = CascadeTable(dl7x; grid = g7, orders = ord7, modes = [0, 2], cache = tempname())
asked7x = Set{Int}()
asking_prim7(p, n, grid) = (push!(asked7x, p.id); prim7(p, n, grid))
runcascade(CascadePlan(tab7x, dl7x); primary = asking_prim7)
@assert asked7x == Set([1, 2, 3, 4, 5])
empty!(asked7x)
pruned7x = CascadePlan(tab7x, filter(!fromparent("om"), dl7x))   # ω no longer decays: ω and X → ωω are dead ends
pruned_result7x = runcascade(pruned7x; primary = asking_prim7)
@assert asked7x == Set([1, 2, 4])
@assert pruned_result7x.F == runcascade(
    pruned7x;
    primary = (p, n, grid) -> p.id in (3, 5) ? zeros(length(grid)) : prim7(p, n, grid),
).F
empty!(asked7x)
runcascade(pruned7x; primary = asking_prim7, output = :all)      # every species is an output
@assert asked7x == Set([1, 2, 3, 4, 5])

res7allparticles =
    runcascade(tab7, dl7; primary = prim7, output = :all)
@assert Set(keys(res7allparticles.F)) == Set(keys(parts7))
@assert Set(keys(res7allparticles.Fprim)) == Set(keys(parts7))
@assert Set(keys(res7allparticles.F)) ==
    Set(particle.id for particle in allparticles(dl7))
for particle_id in dl7.stable_ids
    @assert res7.F[particle_id] == res7allparticles.F[particle_id]
    @assert res7.Fprim[particle_id] ==
        res7allparticles.Fprim[particle_id]
end

# Direct reference through the same package-owned DecaySpectra module.
gv = collect(g7)
Fρ = prim7(parts7[1], 0, gv); Fω = prim7(parts7[3], 0, gv)
for (m, n) in ((1, 0), (2, 2))
    local Wρ = DecayMatrix(TwoBodyDecay(mρ, mπ, mπ; nq = 32, nθ = 8), n, gv, gv).W
    local Wω = DecayMatrix(ThreeBodyDecay(mω, mπ, mπ, mϖ; nµ = 8, nq = 32, nθ = 8), n, gv, gv).W
    # mode-n parent inputs are prim7(·, n, ·) = (n==0 ? 1 : 0.05)·Boltzmann;
    # ρ→ππ feeds both slots (×2 on BR 0.6), ω→ππϖ feeds the two π slots (×2)
    local s = n == 0 ? 1.0 : 0.05
    local expect = prim7(parts7[2], n, gv) .+ 1.2 .* (Wρ * (s .* Fρ)) .+ 2.0 .* (Wω * (s .* Fω))
    local dev = maximum(abs.(res7.F[2][m] .- expect) ./ (abs.(expect) .+ 1.0e-300))
    @printf("   mode n=%d: cascade vs direct max rel dev = %.1e\n", n, dev)
    @assert dev < 1.0e-5                 # Float32 matrix storage rounding
end

# round trip: loading must reproduce exactly
tab7native = loadtables(tabfile)
@assert typeof(tab7native.grid) === typeof(tab7.grid)
@assert typeof(tab7native.W) === typeof(tab7.W)
@assert tab7native.modes === (0, 2)
@assert tab7native.width_cut == tab7.width_cut
@assert tab7native.width_window == tab7.width_window
@assert tab7native.grid == tab7.grid && tab7native.W == tab7.W

# Older caches are rejected from their prefix, before their potentially
# multi-GiB payload is deserialized. Packed metadata is revalidated after
# deserialization for the current format.
legacy_tabfile = tempname()
legacy_W7 = Dict{ChannelKey, Vector{table_matrix_type7}}(
    key => collect(matrices) for (key, matrices) in tab7.W
)
open(legacy_tabfile, "w") do io
    write(io, CascadeTables.CACHE_MAGIC)
    write(io, UInt8(8))
    serialize(
        io,
        (
            tab7.grid,
            tab7.orders,
            collect(tab7.modes),
            tab7.meta,
            legacy_W7,
        ),
    )
end
@assert loadtables(legacy_tabfile) === nothing

legacy_tuple_tabfile = tempname()
legacy_width_metadata = merge(
    copy(tab7.meta),
    Dict{Symbol, Any}(
        :width_cut => tab7.width_cut,
        :width_window => tab7.width_window,
    ),
)
open(legacy_tuple_tabfile, "w") do io
    write(io, CascadeTables.CACHE_MAGIC)
    write(io, UInt8(9))
    serialize(
        io,
        (
            tab7.grid,
            tab7.orders,
            tab7.modes,
            legacy_width_metadata,
            tab7.W,
        ),
    )
end
@assert loadtables(legacy_tuple_tabfile) === nothing

invalid_mode_tabfile = tempname()
open(invalid_mode_tabfile, "w") do io
    write(io, CascadeTables.CACHE_MAGIC)
    write(io, UInt8(CascadeTables.FORMAT))
    serialize(
        io,
        (
            tab7.grid,
            tab7.orders,
            (big(typemax(Int)) + 1,),
            tab7.width_cut,
            tab7.width_window,
            tab7.meta,
            tab7.W,
        ),
    )
end
@assert loadtables(invalid_mode_tabfile) === nothing

invalid_W7 = typeof(tab7.W)(
    key => copy.(matrices) for (key, matrices) in tab7.W
)
invalid_matrix7 = first(first(values(invalid_W7)))
invalid_matrix7.value_offsets[end] -= one(eltype(invalid_matrix7.value_offsets))
invalid_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    tab7.modes,
    tab7.width_cut,
    tab7.width_window,
    tab7.spectral,
    copy(tab7.meta),
    invalid_W7,
)
invalid_tabfile = tempname()
savetables(invalid_table7, invalid_tabfile)
@assert loadtables(invalid_tabfile) === nothing
rm.(
    (
        legacy_tabfile,
        legacy_tuple_tabfile,
        invalid_mode_tabfile,
        invalid_tabfile,
    );
    force = true,
)

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
@assert all(
    typeof(matrix) == table_matrix_type7
        for matrices in values(tab7c.W)
        for matrix in matrices
)
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
    @assert parentids(daughtermass(0.139, 0.14)) == Set([1])
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

# A species' spectral threshold belongs to the resolved model, not to the
# selected topology. Otherwise filtering a broad parent can change ChannelKey
# and make a filtered plan fail to reuse a full table.
let
    parts = Dict(
        10 => ParticleInfo(10, "X", 1.842, 0.083, 1.0, 0, 0),
        11 => ParticleInfo(11, "heavy", 1.680, 0.0, 1.0, 0, 0),
        12 => ParticleInfo(12, "gamma", 0.0, 0.0, 2.0, 0, 0),
        13 => ParticleInfo(13, "pi", 0.140, 0.0, 1.0, 0, 0),
    )
    channels = Dict(10 => [
        DecayChannel(10, 0.2, [11, 12]),
        DecayChannel(10, 0.8, [13, 13, 13, 13]),
    ])
    complete = DecayList(parts, channels, "threshold", "threshold")
    two_body = filter(nbody(2), complete)
    @assert complete.thresholds[10] == 4 * parts[13].mass
    @assert two_body.thresholds == complete.thresholds
    @assert merge(two_body).thresholds == complete.thresholds

    small_orders = (nq = 8, nθ = 4, nµ = 4, nσ = 4, nma = 4)
    full_table = CascadeTable(
        complete;
        grid = range(0.0, 2.0, length = 17),
        orders = small_orders,
        modes = [0],
    )
    @assert CascadePlan(full_table, two_body) isa CascadePlan
end
println("   PASS")

# ---------------------------------------------------------------- test 7c
println("== 7c. Cascade with a broad parent (finite width, Part B wiring) ==")
let g = range(0, 3, length = 61), ord = (nq = 32, nθ = 8, nµ = 8, nσ = 8, nma = 8)
    prim(p, n, gr) = (n == 0 ? 1.0 : 0.0) .* [exp(-hypot(p.mass, q) / T) for q in gr]
    πi = ParticleInfo(2, "pi", 0.1396, 0.0, 1.0, 0, 1)
    byp = Dict(1 => [DecayChannel(1, 1.0, [2, 2])])                         # R → ππ
    dl0 = DecayList(Dict(1 => ParticleInfo(1, "R", 0.9, 0.0, 1.0, 0, 0), 2 => πi), byp, "s", "s0")
    dlb = DecayList(Dict(1 => ParticleInfo(1, "R", 0.9, 0.12, 1.0, 0, 0), 2 => πi), byp, "b", "b0")
    width_cache = tempname()
    table0 = CascadeTable(dl0; grid = g, orders = ord, modes = [0])
    tableb = CascadeTable(
        dlb;
        grid = g,
        orders = ord,
        modes = [0],
        cache = width_cache,
    )
    tablesharp = CascadeTable(
        dlb;
        grid = g,
        orders = ord,
        modes = [0],
        width_cut = Inf,
        cache = width_cache,
    )
    @assert tableb.width_cut == 1.0e-3
    @assert tablesharp.width_cut == Inf
    @assert tableb.width_window == 4.0
    @assert tablesharp.width_window == 4.0
    @assert any(key -> key.Γ > 0, keys(tableb.W))
    @assert all(key -> key.Γ == 0, keys(tablesharp.W))
    @assert loadtables(width_cache).width_cut == Inf

    # The integration window is a table option too. Changing it on the same
    # cache path rebuilds the broad-parent matrices.
    tablewindow = CascadeTable(
        dlb;
        grid = g,
        orders = ord,
        modes = [0],
        width_window = 2.0,
        cache = width_cache,
    )
    @assert tablewindow.width_window == 2.0
    @assert loadtables(width_cache).width_window == 2.0
    window_key = only(keys(tablewindow.W))
    @assert tablewindow.W[window_key][1].values !=
        tableb.W[window_key][1].values

    # The spectral shape is a table option and a cache setting like the width
    # options: same keys, different matrices, and switching it on one cache
    # path rebuilds. The broad key carries the species threshold 2 m_pi.
    @assert tableb.spectral === :massdep
    @assert only(keys(tableb.W)).mth == 2 * 0.1396
    tableconst = CascadeTable(
        dlb;
        grid = g,
        orders = ord,
        modes = [0],
        spectral = :constant,
        cache = width_cache,
    )
    @assert tableconst.spectral === :constant
    @assert loadtables(width_cache).spectral === :constant
    const_key = only(keys(tableconst.W))
    @assert const_key == only(keys(tableb.W))          # keys don't depend on the shape
    @assert tableconst.W[const_key][1].values != tableb.W[const_key][1].values
    caught_spectral = try
        CascadeTable(dlb; grid = g, spectral = :ebw)
        false
    catch e
        e isa ArgumentError
    end
    @assert caught_spectral

    r0 = runcascade(table0, dl0; primary = prim)
    rb = runcascade(tableb, dlb; primary = prim)
    rsharp = runcascade(tablesharp, dlb; primary = prim)
    direct_sharp = runcascade(
        dlb;
        primary = prim,
        grid = g,
        orders = ord,
        modes = [0],
        width_cut = Inf,
    )
    @assert rsharp.F[2][1] == r0.F[2][1]
    @assert rsharp.Fprim[2][1] == r0.Fprim[2][1]
    direct_error = maximum(
        abs.(direct_sharp.F[2][1] .- rsharp.F[2][1]),
    ) / max(maximum(abs.(rsharp.F[2][1])), 1.0)
    @assert direct_error ≤ 1.0e-12

    # The zero cut enrolls every positive width, including widths below the
    # historical 1 MeV default.
    dlnarrow = DecayList(
        Dict(
            1 => ParticleInfo(1, "R", 0.9, 1.0e-4, 1.0, 0, 0),
            2 => πi,
        ),
        byp,
        "n",
        "n0",
    )
    default_keys, _ = CascadeTables.neededkeys(dlnarrow)
    all_width_keys, _ =
        CascadeTables.neededkeys(dlnarrow; width_cut = 0.0)
    @assert all(key -> key.Γ == 0, default_keys)
    @assert any(key -> key.Γ == 1.0e-4, all_width_keys)

    dlboundary = DecayList(
        Dict(
            1 => ParticleInfo(1, "R", 0.9, 1.0e-3, 1.0, 0, 0),
            2 => πi,
        ),
        byp,
        "boundary",
        "boundary0",
    )
    boundary_keys, _ = CascadeTables.neededkeys(dlboundary)
    zero_width_keys, _ =
        CascadeTables.neededkeys(dl0; width_cut = 0.0)
    @assert all(key -> key.Γ == 0, boundary_keys)
    @assert all(key -> key.Γ == 0, zero_width_keys)

    # Switching the option on one cache path rebuilds rather than mixing sharp
    # and broad keys.
    tableb_again = CascadeTable(
        dlb;
        grid = g,
        orders = ord,
        modes = [0],
        cache = width_cache,
    )
    @assert tableb_again.width_cut == 1.0e-3
    @assert all(key -> key.Γ == 0.12, keys(tableb_again.W))
    @assert loadtables(width_cache).width_cut == 1.0e-3

    rm(width_cache; force = true)

    # compare the feed-down (F − Fprim); the total F is diluted by the primary pion
    f0 = r0.F[2][1] .- r0.Fprim[2][1]
    fb = rb.F[2][1] .- rb.Fprim[2][1]
    dif = maximum(abs.(fb .- f0)) / maximum(abs.(f0))
    @assert dif > 1.0e-2        # the Γ=0.12 parent takes the BroadParent path and smears the pion
    @printf("   broad-parent pion feed-down differs from sharp by %.1e\n", dif)
    println("   PASS")
end

# ---------------------------------------------------------------- test 7d
println("== 7d. Reopened channel feeding a heavier daughter (sweep ordering) ==")
# Broad P (pole 1.0, Γ=0.1 → tail to 1.4) → D(1.1) x(0.15): closed at the pole
# (Σm=1.25), reopened by the tail. D(1.1) is HEAVIER than P, so a pole-mass sweep
# processes D first and drops the P→D→(a,b) feed-down. Plan compilation must
# place P before D (feeder first).
let g = range(0, 3, length = 61), ord = (nq = 32, nθ = 8, nµ = 8, nσ = 8, nma = 8)
    parts = Dict(
        1 => ParticleInfo(1, "P", 1.0, 0.1, 1.0, 0, 0),
        2 => ParticleInfo(2, "D", 1.1, 0.0, 1.0, 0, 0),
        3 => ParticleInfo(3, "x", 0.15, 0.0, 1.0, 0, 0),
        4 => ParticleInfo(4, "a", 0.5, 0.0, 1.0, 0, 0),
        5 => ParticleInfo(5, "b", 0.5, 0.0, 1.0, 0, 0)
    )
    byp = Dict(1 => [DecayChannel(1, 1.0, [2, 3])], 2 => [DecayChannel(2, 1.0, [4, 5])])
    dl = DecayList(parts, byp, "syn", "h0")
    @assert CascadeTables._cascadeorder(dl, [(1, 2)]) == [1, 2]
    @assert CascadeTables._cascadeorder(dl, Tuple{Int, Int}[]) == [2, 1]
    sharp_keys, sharp_enrollment =
        CascadeTables.neededkeys(dl; width_cut = Inf)
    @assert all(key -> key.Γ == 0, sharp_keys)
    @assert sharp_enrollment[:reopened] == 0
    @assert sharp_enrollment[:closed] == 1
    short_window_keys, short_window_enrollment =
        CascadeTables.neededkeys(dl; width_window = 2.0)
    @assert all(key -> key.ma != 1.0, short_window_keys)
    @assert short_window_enrollment[:reopened] == 0
    @assert short_window_enrollment[:closed] == 1

    tab = CascadeTable(dl; grid = g, orders = ord, modes = [0], cache = nothing)
    sharp_tab = CascadeTable(
        dl;
        grid = g,
        orders = ord,
        modes = [0],
        width_cut = Inf,
        cache = nothing,
    )
    short_window_tab = CascadeTable(
        dl;
        grid = g,
        orders = ord,
        modes = [0],
        width_window = 2.0,
        cache = nothing,
    )
    @assert short_window_tab.width_window == 2.0
    plan = CascadePlan(tab, dl)
    sharp_plan = CascadePlan(sharp_tab, dl)
    short_window_plan = CascadePlan(short_window_tab, dl)
    @assert plan.ids[[step.parent for step in plan.steps]] == [1, 2]
    @assert sharp_plan.ids[[step.parent for step in sharp_plan.steps]] == [2]
    @assert short_window_plan.ids[
        [step.parent for step in short_window_plan.steps]
    ] == [2]
    prim(p, n, gr) = (n == 0 && p.id == 1) ? [exp(-hypot(p.mass, q) / 0.15) for q in gr] : zeros(length(gr))
    res = runcascade(plan; primary = prim, output = :all)
    sharp_res = runcascade(sharp_plan; primary = prim, output = :all)
    short_window_res =
        runcascade(short_window_plan; primary = prim, output = :all)
    @assert short_window_res.F == sharp_res.F
    @assert short_window_res.Fprim == sharp_res.Fprim
    fda = sum(res.F[4][1] .- res.Fprim[4][1])                 # P→D→a downstream feed-down
    @assert fda > 0                                           # was exactly 0.0 under mass-order
    @assert sharp_res.F[4][1] == sharp_res.Fprim[4][1]
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
            @assert error ≤ 1.0e-12
            @assert candidate.Fprim[particle_id][mode_index] ==
                reference.Fprim[particle_id][mode_index]
        end
    end
    return nothing
end

# The packed matrix kernel derives its accumulator from matrix and source
# arithmetic. Exercise generic numeric types plus the production
# Float32-matrix/Float64-spectrum path before checking complete executors.
let
    matrix = Float32[
        0.25 -0.5 0.75 1.0
        -1.25 0.125 0.5 -0.75
        0.375 1.5 -0.25 0.625
        2.0 -0.875 0.0625 0.25
        -0.125 0.5 1.25 -1.5
    ]
    source = [0.75, -1.25, 0.5, 1.75]
    initial = [-0.2, 0.4, 1.1, -0.7, 0.3]

    banded_matrix = copy(matrix)
    banded_matrix[1, 1] = 0
    banded_matrix[1, 3] = 0
    banded_matrix[2, 4] = 0
    banded_matrix[3, 1:2] .= 0
    banded_matrix[4, :] .= 0
    banded_matrix[5, 4] = 0
    packed_matrix = CascadeTables.RowBandMatrix(banded_matrix)

    for (α, β) in (
            (1.0, 0.0),
            (0.37, 1.0),
            (-0.4, 0.25),
            (0.0, 1.0),
            (0.0, 0.0),
        )
        expected = copy(initial)
        mul!(expected, banded_matrix, source, α, β)
        actual = copy(initial)
        returned = mul!(
            actual,
            packed_matrix,
            source,
            α,
            β,
        )
        @assert returned === actual
        @assert isapprox(actual, expected; rtol = 2.0e-14, atol = 2.0e-14)
    end

    packed_expected = similar(initial)
    mul!(packed_expected, banded_matrix, source)
    packed_actual = similar(initial)
    @assert mul!(
        packed_actual,
        packed_matrix,
        source,
    ) === packed_actual
    @assert isapprox(
        packed_actual,
        packed_expected;
        rtol = 2.0e-14,
        atol = 2.0e-14,
    )
    @assert isapprox(
        packed_matrix * source,
        packed_expected;
        rtol = 2.0e-14,
        atol = 2.0e-14,
    )

    generic_matrix = [
        1 0 -2
        0 3 0
        4 0 5
    ]
    generic_source = [2, -1, 3]
    generic_initial = [5, -2, 1]
    for (matrix_type, source_type, destination_type) in (
            (Float32, Float32, Float32),
            (Float64, Float32, Float64),
            (Float32, Float64, Float64),
            (Float32, ComplexF64, ComplexF64),
            (ComplexF64, Float64, ComplexF64),
            (Int, Int, Int),
            (BigInt, BigInt, BigInt),
            (Rational{Int}, Rational{Int}, Rational{Int}),
        )
        dense = matrix_type.(generic_matrix)
        packed = CascadeTables.RowBandMatrix(dense)
        typed_source = source_type.(generic_source)
        expected = destination_type.(generic_initial)
        actual = copy(expected)
        mul!(expected, dense, typed_source, 2, -1)
        @assert mul!(actual, packed, typed_source, 2, -1) === actual
        @assert actual ≈ expected
        @assert packed * typed_source ≈ dense * typed_source
    end

    generic_float32_matrix =
        CascadeTables.RowBandMatrix(Float32.(generic_matrix))
    generic_float32_source = Float32.(generic_source)
    generic_float32_destination = zeros(Float32, 3)
    generic_complex_source = ComplexF64.(generic_source)
    generic_complex_destination = zeros(ComplexF64, 3)
    mul!(
        generic_float32_destination,
        generic_float32_matrix,
        generic_float32_source,
    )
    mul!(
        generic_complex_destination,
        generic_float32_matrix,
        generic_complex_source,
    )
    @assert @allocated(
        mul!(
            generic_float32_destination,
            generic_float32_matrix,
            generic_float32_source,
        )
    ) == 0
    @assert @allocated(
        mul!(
            generic_complex_destination,
            generic_float32_matrix,
            generic_complex_source,
        )
    ) == 0

    source_buffer = [0.0; source; 0.0]
    destination_buffer = zeros(length(initial) + 2)
    source_view = view(source_buffer, 2:(length(source) + 1))
    destination_view =
        view(destination_buffer, 2:(length(initial) + 1))
    @assert mul!(
        destination_view,
        packed_matrix,
        source_view,
    ) === destination_view
    @assert isapprox(
        destination_view,
        packed_expected;
        rtol = 2.0e-14,
        atol = 2.0e-14,
    )

    scatter_source = [0.5, -0.75, 1.25, -1.5, 0.125]
    expected_scatter = copy(initial)
    axpy!(-0.35, scatter_source, expected_scatter)
    actual_scatter = copy(initial)
    @assert CascadeTables._cascade_axpy!(
        -0.35,
        scatter_source,
        actual_scatter,
    ) === actual_scatter
    @assert isapprox(
        actual_scatter,
        expected_scatter;
        rtol = 2.0e-15,
        atol = 2.0e-15,
    )

    empty_matrix = zeros(Float32, length(initial), 0)
    empty_destination = fill(NaN, length(initial))
    packed_empty_matrix = CascadeTables.RowBandMatrix(empty_matrix)
    mul!(
        empty_destination,
        packed_empty_matrix,
        Float64[],
    )
    @assert all(iszero, empty_destination)

    for (α, β, expected) in (
            (Inf, 0.0, zeros(length(initial))),
            (Inf, 1.0, initial),
            (Inf, 0.25, 0.25 .* initial),
        )
        actual = copy(initial)
        mul!(
            actual,
            packed_empty_matrix,
            Float64[],
            α,
            β,
        )
        @assert actual == expected
    end

    no_rows_matrix =
        CascadeTables.RowBandMatrix(zeros(Float32, 0, length(source)))
    no_rows_destination = Float64[]
    @assert mul!(
        no_rows_destination,
        no_rows_matrix,
        source,
        0.37,
        0.25,
    ) === no_rows_destination

    for invalid_call in (
            () -> mul!(
                zeros(length(initial) - 1),
                packed_matrix,
                source,
            ),
            () -> mul!(
                zeros(length(initial)),
                packed_matrix,
                source[1:(end - 1)],
            ),
            () -> CascadeTables._cascade_axpy!(
                1.0,
                scatter_source,
                zeros(length(initial) - 1),
            ),
        )
        threw = false
        try
            invalid_call()
        catch error
            threw = error isa DimensionMismatch
        end
        @assert threw
    end

    # Warm the exact production methods before checking the steady-state path.
    mul!(
        packed_actual,
        packed_matrix,
        source,
        0.37,
        1.0,
    )
    mul!(packed_actual, packed_matrix, source)
    mul!(destination_view, packed_matrix, source_view)
    CascadeTables._cascade_axpy!(-0.35, scatter_source, actual_scatter)
    @assert @allocated(
        mul!(
            packed_actual,
            packed_matrix,
            source,
            0.37,
            1.0,
        )
    ) == 0
    @assert @allocated(
        mul!(
            packed_actual,
            packed_matrix,
            source,
        )
    ) == 0
    @assert @allocated(
        mul!(
            destination_view,
            packed_matrix,
            source_view,
        )
    ) == 0
    @assert @allocated(
        CascadeTables._cascade_axpy!(-0.35, scatter_source, actual_scatter)
    ) == 0
end

# The ordinary two-mode topology covers repeated daughter slots (rho → pi pi)
# and verifies that the compiled arrays form gap-free flat ranges.
plan7 = CascadePlan(tab7, dl7)
planned7 = runcascade(plan7; primary = prim7)
assert_plan_matches(planned7, res7)

# The high-level constructor owns its table and forwards all table options.
convenience_cache7 = tempname()
convenience_plan7 = CascadePlan(
    dl7;
    grid = g7,
    orders = ord7,
    modes = [0, 2],
    cache = convenience_cache7,
)
@assert isfile(convenience_cache7)
@assert convenience_plan7.table.grid === g7
@assert convenience_plan7.table.modes === (0, 2)
@assert convenience_plan7.table.orders == tab7.orders
assert_plan_matches(
    runcascade(convenience_plan7; primary = prim7),
    planned7,
)

# The one-shot wrapper separates table options from execution options.
direct7 = runcascade(
    dl7;
    primary = prim7,
    grid = g7,
    orders = ord7,
    modes = [0, 2],
    cache = convenience_cache7,
    parallel = :modes,
    output = :all,
)
assert_plan_matches(direct7, res7allparticles)

# Invalid execution options fail before an expensive table is constructed.
invalid_direct_cache7 = tempname()
let threw = false
    try
        runcascade(
            dl7;
            primary = prim7,
            grid = g7,
            cache = invalid_direct_cache7,
            parallel = :invalid,
        )
    catch error
        threw =
            error isa ArgumentError &&
            occursin("parallel", error.msg)
    end
    @assert threw
end
@assert !isfile(invalid_direct_cache7)
rm(convenience_cache7; force = true)

@assert plan7.ids == sort!(collect(keys(dl7.parts)))
@assert plan7.ids[plan7.stable_indices] == dl7.stable_ids
plan_particles7 = allparticles(plan7)
@assert [particle.id for particle in plan_particles7] == plan7.ids
pop!(plan_particles7)
@assert length(plan7.particles) == length(plan7.ids)
@assert [particle.id for particle in stableparticles(plan7)] ==
    dl7.stable_ids
assert_plan_matches(
    runcascade(plan7; primary = prim7, output = :all),
    res7allparticles,
)
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
assert_plan_matches(
    runcascade(
        plan7;
        primary = prim7,
        parallel = :modes,
        output = :all,
    ),
    res7allparticles,
)

# Reuse with a different callback must not retain or mutate run state.
prim7alt(p, n, grid) =
    (1.0 + p.id / 10) * (n == 0 ? 0.7 : -0.03) *
    [exp(-hypot(p.mass, q) / (T * 1.1)) for q in grid]
planned7_snapshot = deepcopy(planned7.F)
alternate7 = runcascade(plan7; primary = prim7alt)
@assert alternate7.F != planned7.F
@assert alternate7.Fprim != planned7.Fprim
@assert planned7.F == planned7_snapshot

# The plan itself is immutable during execution and may serve concurrent calls.
task7a = Threads.@spawn runcascade(plan7; primary = prim7)
task7b = Threads.@spawn runcascade(plan7; primary = prim7alt)
concurrent7a, concurrent7b = fetch(task7a), fetch(task7b)
@assert concurrent7a.F == planned7.F
@assert concurrent7a.Fprim == planned7.Fprim
@assert concurrent7b.F == alternate7.F
@assert concurrent7b.Fprim == alternate7.Fprim

# Complex primaries propagate both components through the real decay kernels,
# so a complex run equals the real and imaginary runs combined on both the
# serial and the mode-parallel paths, and the output element type follows
# `spectrum_type`.
prim7complex(p, n, grid) = complex.(prim7(p, n, grid), prim7alt(p, n, grid))
complex7 = runcascade(
    plan7;
    primary = prim7complex,
    spectrum_type = ComplexF64,
)
@assert keys(complex7.F) == keys(planned7.F)
for particle_id in keys(complex7.F), mode_index in 1:length(plan7.table.modes)
    @assert complex7.F[particle_id][mode_index] isa Vector{ComplexF64}
    @assert complex7.F[particle_id][mode_index] ≈ complex.(
        planned7.F[particle_id][mode_index],
        alternate7.F[particle_id][mode_index],
    )
    @assert complex7.Fprim[particle_id][mode_index] == complex.(
        planned7.Fprim[particle_id][mode_index],
        alternate7.Fprim[particle_id][mode_index],
    )
end
@assert runcascade(
    plan7;
    primary = prim7complex,
    spectrum_type = ComplexF64,
    parallel = :modes,
).F == complex7.F
# A real primary is promoted when a complex run is requested, and a complex
# primary is rejected by the default real run instead of being truncated.
promoted7 = runcascade(plan7; primary = prim7, spectrum_type = ComplexF64)
for particle_id in keys(planned7.F), mode_index in 1:length(plan7.table.modes)
    @assert promoted7.F[particle_id][mode_index] isa Vector{ComplexF64}
    @assert promoted7.F[particle_id][mode_index] ≈
        planned7.F[particle_id][mode_index]
end
let threw = false
    try
        runcascade(plan7; primary = prim7complex)
    catch error
        threw = error isa InexactError
    end
    @assert threw
end

# A filtered topology compiles against the already-built full table.
two_body7 = filter(nbody(2), dl7)
@assert two_body7.stable_ids == dl7.stable_ids
two_body_plan7 = CascadePlan(tab7, two_body7)
two_body_result7 = runcascade(two_body_plan7; primary = prim7)
@assert two_body_result7.Fprim == planned7.Fprim
@assert all(two_body_result7.F[2][1] .≤ planned7.F[2][1])
@assert any(two_body_result7.F[2][1] .< planned7.F[2][1])
empty_decay_list7 =
    filter(channelfilter((_parent, _channel, _daughters) -> false), dl7)
empty_plan7 = CascadePlan(tab7, empty_decay_list7)
@assert isempty(empty_plan7.steps)
@assert isempty(empty_plan7.groups)
@assert isempty(empty_plan7.targets)
empty_result7 = runcascade(empty_plan7; primary = prim7)
@assert empty_result7.F == empty_result7.Fprim
empty_all_result7 =
    runcascade(empty_plan7; primary = prim7, output = :all)
@assert empty_all_result7.F == empty_all_result7.Fprim
@assert Set(keys(empty_all_result7.F)) == Set(keys(dl7.parts))

# Execution no longer consults the keyed table after plan construction.
snapshot_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    tab7.modes,
    tab7.width_cut,
    tab7.width_window,
    tab7.spectral,
    copy(tab7.meta),
    copy(tab7.W),
)
snapshot_plan7 = CascadePlan(snapshot_table7, dl7)
snapshot_result7 = runcascade(snapshot_plan7; primary = prim7)
empty!(snapshot_table7.W)
@assert runcascade(snapshot_plan7; primary = prim7).F == snapshot_result7.F

# Mode position, rather than the numerical mode value, selects a matrix.
reversed_W7 = typeof(tab7.W)(
    key => reverse(matrices) for (key, matrices) in tab7.W
)
reversed_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    reverse(tab7.modes),
    tab7.width_cut,
    tab7.width_window,
    tab7.spectral,
    copy(tab7.meta),
    reversed_W7,
)
reversed_plan7 = CascadePlan(reversed_table7, dl7)
reversed_serial7 = runcascade(reversed_plan7; primary = prim7)
for particle_id in keys(planned7.F)
    @assert reversed_serial7.F[particle_id] ==
        reverse(planned7.F[particle_id])
    @assert reversed_serial7.Fprim[particle_id] ==
        reverse(planned7.Fprim[particle_id])
end
@assert runcascade(
    reversed_plan7;
    primary = prim7,
    parallel = :modes,
).F == reversed_serial7.F

one_mode_W7 = Dict{ChannelKey, NTuple{1, table_matrix_type7}}(
    key => (matrices[1],) for (key, matrices) in tab7.W
)
one_mode_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    (tab7.modes[1],),
    tab7.width_cut,
    tab7.width_window,
    tab7.spectral,
    copy(tab7.meta),
    one_mode_W7,
)
one_mode_plan7 = CascadePlan(one_mode_table7, dl7)
one_mode_serial7 = runcascade(one_mode_plan7; primary = prim7)
for particle_id in keys(planned7.F)
    @assert only(one_mode_serial7.F[particle_id]) ==
        planned7.F[particle_id][1]
    @assert only(one_mode_serial7.Fprim[particle_id]) ==
        planned7.Fprim[particle_id][1]
end
@assert runcascade(
    one_mode_plan7;
    primary = prim7,
    parallel = :modes,
).F == one_mode_serial7.F

# Invalid execution options fail before the callback is invoked.
never_primary7(_args...) = error("primary must not be called")
for invalid_parallel in (:threads, "modes", 2)
    let threw = false
        try
            runcascade(
                plan7;
                primary = never_primary7,
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

for invalid_output in (:leaves, "stable", 2)
    for run in (
            () -> runcascade(
                plan7;
                primary = never_primary7,
                output = invalid_output,
            ),
            () -> runcascade(
                tab7,
                dl7;
                primary = never_primary7,
                output = invalid_output,
            ),
        )
        let threw = false
            try
                run()
            catch error
                threw =
                    error isa ArgumentError &&
                    occursin(":stable or :all", sprint(showerror, error))
            end
            @assert threw
        end
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

# Missing table payloads are rejected while compiling the plan.
missing_W7 = copy(tab7.W)
missing_key7 = first(keys(missing_W7))
delete!(missing_W7, missing_key7)
missing_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    tab7.modes,
    tab7.width_cut,
    tab7.width_window,
    tab7.spectral,
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

# Matrix count is encoded in the dictionary value type. A same-format cache
# whose tuple arity does not match its modes is rejected during restoration.
malformed_W7 = Dict{ChannelKey, NTuple{1, table_matrix_type7}}(
    key => (matrices[1],) for (key, matrices) in tab7.W
)
malformed_tabfile7 = tempname()
open(malformed_tabfile7, "w") do io
    write(io, CascadeTables.CACHE_MAGIC)
    write(io, UInt8(CascadeTables.FORMAT))
    serialize(
        io,
        (
            tab7.grid,
            tab7.orders,
            tab7.modes,
            tab7.width_cut,
            tab7.width_window,
            tab7.meta,
            malformed_W7,
        ),
    )
end
@assert loadtables(malformed_tabfile7) === nothing
rm(malformed_tabfile7; force = true)

# Restore-time shape validation remains even though tuple arity needs no scan.
bad_shape_W7 = copy(tab7.W)
bad_shape_key7 = first(keys(bad_shape_W7))
bad_shape_matrices7 = bad_shape_W7[bad_shape_key7]
reference_matrix7 = bad_shape_matrices7[1]
bad_shape_matrix7 = CascadeTables.RowBandMatrix(
    zeros(Float32, length(tab7.grid) - 1, length(tab7.grid)),
    eltype(reference_matrix7.column_starts),
    eltype(reference_matrix7.value_offsets),
)
bad_shape_W7[bad_shape_key7] =
    (bad_shape_matrix7, bad_shape_matrices7[2])
bad_shape_table7 = CascadeTable(
    tab7.grid,
    tab7.orders,
    tab7.modes,
    tab7.width_cut,
    tab7.width_window,
    tab7.spectral,
    copy(tab7.meta),
    bad_shape_W7,
)
bad_shape_tabfile7 = tempname()
savetables(bad_shape_table7, bad_shape_tabfile7)
@assert loadtables(bad_shape_tabfile7) === nothing
rm(bad_shape_tabfile7; force = true)

# Direct raw tables trust matrix shapes; public multiplication still reports
# the mismatch before entering its @inbounds kernel.
bad_shape_plan7 = CascadePlan(bad_shape_table7, dl7)
let threw = false
    try
        runcascade(bad_shape_plan7; primary = prim7)
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
    @assert isapprox(shared_weights[20], 0.3; atol = 1.0e-15)
    @assert isapprox(shared_weights[21], 0.3; atol = 1.0e-15)

    primary(p, n, gr) =
        (n == 0 ? 1.0 : 0.04) *
        [exp(-hypot(p.mass, q) / T) for q in gr]
    result = runcascade(plan; primary, output = :all)
    light_key = ChannelKey(1.5, 0.2, [0.3])
    partner_key = ChannelKey(1.5, 0.3, [0.2])
    for (mode_index, mode) in pairs(table.modes)
        parent_p = primary(parts[10], mode, grid)
        parent_q = primary(parts[11], mode, grid)
        light_from_p = table.W[light_key][mode_index] * parent_p
        light_from_q = table.W[light_key][mode_index] * parent_q
        partner_from_p = table.W[partner_key][mode_index] * parent_p
        partner_from_q = table.W[partner_key][mode_index] * parent_q
        expected = Dict(
            20 => primary(parts[20], mode, grid) .+
                0.3 .* light_from_p,
            21 => primary(parts[21], mode, grid) .+
                0.3 .* light_from_p .+
                0.4 .* light_from_q,
            30 => primary(parts[30], mode, grid) .+
                0.6 .* partner_from_p .+
                0.4 .* partner_from_q,
        )
        for particle_id in keys(expected)
            @assert isapprox(
                result.F[particle_id][mode_index],
                expected[particle_id];
                rtol = 1.0e-12,
                atol = 1.0e-12,
            )
        end
    end
end

# A cycle uses the deterministic mass-descending approximate sweep.
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
    rejected = false
    try
        feeddown_order(decay_list)
    catch error
        rejected = error isa ArgumentError &&
            occursin("cyclic topology", sprint(showerror, error))
    end
    @assert rejected
    matrix_type = CascadeTables._tablematrixtype(length(grid))
    matrix_entry = 0.002f0
    matrices = Dict{ChannelKey, NTuple{1, matrix_type}}()
    CascadeTables.eachchannelkey(decay_list) do _parent, _channel, _daughter, key
        get!(matrices, key) do
            (
                CascadeTables._packtablematrix(
                    matrix_type,
                    fill(matrix_entry, length(grid), length(grid)),
                ),
            )
        end
    end
    table = CascadeTable(
        grid,
        tab7.orders,
        (0,),
        1.0e-3,
        4.0,
        :massdep,
        Dict{Symbol, Any}(),
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
    result = runcascade(plan; primary, output = :all)

    # The cycle is approximated by one deterministic P-then-Q sweep. Check its
    # numerical consequence directly rather than comparing two plan entry points.
    p_transfer = length(grid) * Float64(matrix_entry)
    expected_q = 0.1 + 0.2 * p_transfer
    q_transfer = length(grid) * Float64(matrix_entry) * expected_q
    expected_p = 1.0 + 0.3 * q_transfer
    expected_x = 0.1 + 0.2 * p_transfer + 0.3 * q_transfer
    @assert all(isapprox.(result.F[101][1], expected_p; rtol = 1.0e-14))
    @assert all(isapprox.(result.F[102][1], expected_q; rtol = 1.0e-14))
    @assert all(isapprox.(result.F[103][1], expected_x; rtol = 1.0e-14))

    # A heavy particle downstream of a cycle still runs after the complete
    # cyclic component; only the irreducible edges inside that component are
    # approximated.
    ordering_parts = Dict(
        1 => ParticleInfo(1, "A", 3.0, 0.0, 1.0, 0, 0),
        2 => ParticleInfo(2, "B", 2.0, 0.0, 1.0, 0, 0),
        3 => ParticleInfo(3, "C", 10.0, 0.0, 1.0, 0, 0),
        4 => ParticleInfo(4, "stable", 0.1, 0.0, 1.0, 0, 0),
    )
    ordering_channels = Dict(
        1 => [DecayChannel(1, 1.0, [4, 4])],
        2 => [DecayChannel(2, 1.0, [4, 4])],
        3 => [DecayChannel(3, 1.0, [4, 4])],
    )
    ordering_list = DecayList(
        ordering_parts,
        ordering_channels,
        "cycle ordering",
        "cycle ordering",
    )
    dependencies = Set([(1, 2), (2, 1), (2, 3)])
    repeated_log = IOBuffer()
    orders = with_logger(SimpleLogger(repeated_log, Logging.Warn)) do
        [
            CascadeTables._cascadeorder(ordering_list, dependencies),
            CascadeTables._cascadeorder(ordering_list, dependencies),
        ]
    end
    @assert orders == [[1, 2, 3], [1, 2, 3]]
    @assert count("cycle components", String(take!(repeated_log))) == 2
end

println("   PASS")

# ---------------------------------------------------------------- test 8
println("== 8. Public API contract: shapes, types, dispatch, invariants ==")
# Pins the boost-invariant public interface so refactors (and the planned
# rapidity-tensor extension) cannot silently drift it. The math is covered
# above; this asserts the *contract* every extension must preserve.
@assert all(
    name -> name in names(CascadeTables),
    (:readparticles, :readdecays),
)
@assert FastReso2.ParticleInfo === CascadeTables.ParticleInfo
@assert FastReso2.ParticleInfo === FastReso2.ParticleList.ParticleInfo
@assert FastReso2.DecayList === CascadeTables.DecayList
@assert FastReso2.DecayList === FastReso2.DecayModel.DecayList
@assert FastReso2.MassDependentBW === CascadeTables.MassDependentBW
@assert FastReso2.MassDependentBW === ThermalModel.MassDependentBW
@assert all(
    name -> name ∉ names(CascadeTables),
    (
        :readgenericparticles,
        :readgenericdecays,
        :readgeneric,
        :readazyhydro,
        :readtherminator,
    ),
)
@assert !isdefined(FastReso2.ParticleList, :read_decaylist)
@assert !hasmethod(DecayList, Tuple{String})
@assert !isdefined(CascadeTables, :ParticleSchema)
@assert !isdefined(CascadeTables, :DecaySchema)
let g = range(0.0, 4.0, length = 40), pT8 = collect(0.1:0.1:1.5)
    rules8 = Rules(; nq = 8, nθ = 4, nµ = 4, nσ = 4, nma = 4)
    @assert fieldtype(typeof(rules8), :µ) === typeof(rules8.µ)
    @assert fieldtype(typeof(rules8), :σ) === typeof(rules8.σ)

    ρ8 = TwoBodyDecay(0.7752, 0.1396, 0.1396)
    ω8 = ThreeBodyDecay(0.7827, 0.1396, 0.1396, 0.135)
    d48 = FourBodyDecay(1.3, 0.1396, 0.1396, 0.1396, 0.135)
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
    @assert decaymatrices(ρ8, Int8[0, 2], pT8, g) == W
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
    for bad in (
            () -> BreitWigner(1.0, 0.0),                        # Γ = 0: ψ map divides by 0
            () -> BreitWigner(1.0, -0.1),                        # Γ < 0: inverts the window
            () -> BreitWigner(-1.0, 0.1),                        # pole outside the support
            () -> BreitWigner(NaN, 0.1),                        # non-finite
            () -> BreitWigner(Inf, 0.1),
            () -> BreitWigner(1.0, 0.1; mmin = 1.5, mmax = 0.5), # reversed
            () -> BreitWigner(1.0, 0.1; mmin = 1.0, mmax = 1.0), # empty
            () -> Rules(; nq = 0), () -> Rules(; nµ = -1),
        )       # orders
        threw = false
        try
            bad()
        catch e
            threw = e isa ArgumentError
        end
        @assert threw
    end

    # constructors reject non-open decays: at exact threshold p_b = 0 makes
    # coef = Inf (the functor then returns NaN) — must fail loud instead
    for bad in (
            () -> TwoBodyDecay(0.14 + 0.14, 0.14, 0.14),
            () -> ThreeBodyDecay(0.14 + 0.14 + 0.14, 0.14, 0.14, 0.14),
            () -> FourBodyDecay(4 * 0.14, 0.14, 0.14, 0.14, 0.14),
        )
        threw = false
        try
            bad()
        catch
            threw = true
        end
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
        dcl = BroadParent(0.6, 0.6, BreitWigner(1.0, 0.05; mmin = 0.85, mmax = 1.15))
        Wz = (fill(NaN, np, nq),)
        decaymatrices!(Wz, dcl, (0,), pT8, g)
        @assert all(iszero, Wz[1])
        # Shape is checked, not assumed — of the output and of serial scratch.
        for bad in (
                () -> decaymatrices!((fill(NaN, np, nq),), ρ8, (0, 2), pT8, g),
                () -> decaymatrices!(
                    (fill(NaN, np + 1, nq), fill(NaN, np + 1, nq)),
                    ρ8, (0, 2), pT8, g
                ),
                () -> decaymatrices!(
                    Wz, ρ8, (0,), pT8, g;             # buffers too short
                    threaded = false,
                    scratch = rowscratch((0,), nq - 1)
                ),
                () -> decaymatrices!(
                    Wz, ρ8, (0,), pT8, g;             # wrong buffer count
                    threaded = false,
                    scratch = rowscratch((0, 2), nq)
                ),
            )
            threw = false
            try
                bad()
            catch e
                threw = e isa DimensionMismatch
            end
            @assert threw
        end

        # Threaded assembly owns one scratch set per task. Caller scratch is
        # intentionally ignored, so even a wrong shape cannot affect output.
        threaded_output = (fill(NaN, np, nq),)
        decaymatrices!(
            threaded_output,
            ρ8,
            (0,),
            pT8,
            g;
            threaded = true,
            scratch = (),
        )
        @assert threaded_output[1] ==
            decaymatrices(ρ8, [0], pT8, g; threaded = true)[1]

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

# ---------------------------------------------------------------- test 9
println("== 9. Heavy-flavour supplement: parser and merged-list invariants ==")
let
    data_directory = joinpath(REPO, "data")
    base_particle_path = joinpath(data_directory, "particles_QM2016Plus.dat")
    base_decay_path = joinpath(data_directory, "decays_QM2016Plus.dat")
    supplement_particle_path =
        joinpath(data_directory, "particles_charm_bottom.dat")
    supplement_decay_path =
        joinpath(data_directory, "decays_charm_bottom.dat")
    base = DecayList(
        readparticles(base_particle_path),
        readdecays(base_decay_path),
    )
    supplement = (
        catalog = readparticles(supplement_particle_path),
        network = readdecays(supplement_decay_path),
    )

    # The checked-in canonical production pair remains an exact conversion of
    # the external-format source used by the FastReso cross-check.
    converted_particles_path = tempname()
    converted_decays_path = tempname()
    convert_azyhydro(
        [joinpath(REPO, "data/azyhydro/decays_QM2016Plus_massorder.dat")],
        converted_particles_path,
        converted_decays_path,
    )
    converted_base = DecayList(
        readparticles(converted_particles_path),
        readdecays(converted_decays_path),
    )
    @assert converted_base.parts == base.parts
    @assert converted_base.stable_ids == base.stable_ids
    signatures(decay_list) = Dict(
        parent_id => [
            (channel.parent, channel.br, Tuple(channel.daughters))
                for channel in channels
        ] for (parent_id, channels) in decay_list.byparent
    )
    @assert signatures(converted_base) == signatures(base)
    @assert CascadeTables.neededkeys(converted_base) ==
        CascadeTables.neededkeys(base)
    rm.((converted_particles_path, converted_decays_path); force = true)

    @assert length(supplement.catalog.byid) == 150
    @assert isempty(intersect(keys(base.parts), keys(supplement.catalog.byid)))
    @assert sum(length, values(supplement.network.byparent)) == 460

    unresolved_supplement = nothing
    try
        DecayList(supplement.catalog, supplement.network)
    catch error
        unresolved_supplement = error
    end
    @assert unresolved_supplement isa ArgumentError
    @assert occursin(
        "unresolved particle references",
        sprint(showerror, unresolved_supplement),
    )

    combined = DecayList(
        merge(base.catalog, supplement.catalog),
        merge(base.network, supplement.network),
    )
    @assert combined.parts !== base.parts
    @assert combined.parts !== supplement.catalog.byid
    @assert combined.byparent !== base.byparent
    @assert combined.byparent !== supplement.network.byparent
    @assert combined.path == join(
        (
            base.catalog.path,
            supplement.catalog.path,
            base.network.path,
            supplement.network.path,
        ),
        " + ",
    )
    @assert combined.sha == join(
        (
            base.catalog.sha,
            supplement.catalog.sha,
            base.network.sha,
            supplement.network.sha,
        ),
        "+",
    )
    supplement_stable_ids = collect(setdiff(
        keys(supplement.catalog.byid),
        supplement.network.decaying_ids,
    ))
    @assert combined.stable_ids ==
        sort!(vcat(base.stable_ids, supplement_stable_ids))

    copied_base = merge(base)
    @assert copied_base.parts == base.parts
    @assert copied_base.parts !== base.parts
    @assert copied_base.byparent == base.byparent
    @assert copied_base.byparent !== base.byparent
    @assert copied_base.stable_ids == base.stable_ids
    @assert copied_base.stable_ids !== base.stable_ids
    @assert copied_base.path == base.path && copied_base.sha == base.sha

    third_id = 999_999_998
    third = DecayList(
        Dict(
            third_id =>
                ParticleInfo(third_id, "third", 0.1, 0.0, 1.0, 0, 0),
        ),
        Dict{Int, Vector{DecayChannel}}(),
        "third source",
        "third hash",
    )
    direct_merge = DecayList(
        merge(base.catalog, supplement.catalog, third.catalog),
        merge(base.network, supplement.network, third.network),
    )
    nested_merge = DecayList(
        merge(merge(base.catalog, supplement.catalog), third.catalog),
        merge(merge(base.network, supplement.network), third.network),
    )
    @assert direct_merge.parts == nested_merge.parts
    @assert direct_merge.byparent == nested_merge.byparent
    @assert direct_merge.stable_ids == nested_merge.stable_ids
    @assert direct_merge.path == nested_merge.path
    @assert direct_merge.sha == nested_merge.sha

    filtered_base = filter(
        channelfilter((_parent, _channel, _daughters) -> false),
        base,
    )
    filtered_combined = DecayList(
        merge(filtered_base.catalog, supplement.catalog),
        merge(filtered_base.network, supplement.network),
    )
    @assert filtered_combined.stable_ids == combined.stable_ids

    duplicated_base = merge(base, base)
    @assert duplicated_base.parts == base.parts
    @assert duplicated_base.byparent == base.byparent
    @assert duplicated_base.stable_ids == base.stable_ids

    overlapping_parent = first(
        parent_id for parent_id in keys(base.byparent)
            if parent_id ∉ base.stable_ids
    )
    header_only = DecayList(
        Dict(overlapping_parent => base.parts[overlapping_parent]),
        Dict{Int, Vector{DecayChannel}}(),
        "header only",
        "header only",
    )
    merged_header = merge(base, header_only)
    @assert merged_header.parts == base.parts
    @assert merged_header.byparent == base.byparent
    @assert overlapping_parent ∉ merged_header.stable_ids

    reordered_channels = reverse([
        DecayChannel(channel.parent, channel.br, reverse(channel.daughters))
            for channel in base.byparent[overlapping_parent]
    ])
    reordered_catalog = ParticleCatalog(
        Dict(overlapping_parent => base.parts[overlapping_parent]);
        path = "reordered block",
        sha = "reordered block",
    )
    reordered_network = DecayNetwork(
        Dict(overlapping_parent => reordered_channels);
        path = "reordered block",
        sha = "reordered block",
    )
    merged_reordered = DecayList(
        merge(base.catalog, reordered_catalog),
        merge(base.network, reordered_network),
    )
    @assert merged_reordered.byparent[overlapping_parent] ===
        base.byparent[overlapping_parent]

    particle = base.parts[overlapping_parent]
    alias_particle = ParticleInfo(
        particle.id,
        particle.name * "_alias",
        particle.mass,
        particle.width,
        particle.degeneracy,
        particle.baryon,
        particle.charge,
        particle.strangeness,
        particle.charm,
        particle.bottomness,
        particle.twiceisospin,
    )
    alias_header = DecayList(
        Dict(overlapping_parent => alias_particle),
        Dict{Int, Vector{DecayChannel}}(),
        "alias header",
        "alias header",
    )
    merged_alias = merge(base, alias_header)
    @assert merged_alias.parts[overlapping_parent].name == particle.name
    @assert particleids(merged_alias, particle.name * "_alias") ==
        [overlapping_parent]

    conflicting_particle = ParticleInfo(
        particle.id,
        particle.name,
        nextfloat(particle.mass),
        particle.width,
        particle.degeneracy,
        particle.baryon,
        particle.charge,
        particle.strangeness,
        particle.charm,
        particle.bottomness,
        particle.twiceisospin,
    )
    conflicting_header = DecayList(
        Dict(overlapping_parent => conflicting_particle),
        Dict{Int, Vector{DecayChannel}}(),
        "conflicting header",
        "conflicting header",
    )
    particle_conflict = nothing
    try
        merge(base, conflicting_header)
    catch error
        particle_conflict = error
    end
    @assert particle_conflict isa ArgumentError
    @assert occursin("mass", sprint(showerror, particle_conflict))

    conflicting_channels = copy(base.byparent[overlapping_parent])
    channel = conflicting_channels[1]
    conflicting_channels[1] = DecayChannel(
        channel.parent,
        nextfloat(channel.br),
        copy(channel.daughters),
    )
    conflicting_catalog = ParticleCatalog(
        Dict(
            overlapping_parent => base.parts[overlapping_parent],
        );
        path = "conflicting block",
        sha = "conflicting block",
    )
    conflicting_network = DecayNetwork(
        Dict(overlapping_parent => conflicting_channels);
        path = "conflicting block",
        sha = "conflicting block",
    )
    channel_conflict = nothing
    try
        DecayList(
            merge(base.catalog, conflicting_catalog),
            merge(base.network, conflicting_network),
        )
    catch error
        channel_conflict = error
    end
    @assert channel_conflict isa ArgumentError

    stats = validate(combined; verbose = false)
    @assert stats.unresolved == 0
    @assert stats.br_out_of_range == 0
    @assert stats.charge_violations == 0
    @assert stats.baryon_violations == 0
    @assert stats.subthreshold == 0
    @assert stats.bottomness_changing == 0
    @assert stats.duplicate_channels == 2

    # Absolute retained PDG probabilities intentionally leave some parent sums
    # below one. Their normalization outliers are documented data coverage, not
    # malformed individual probabilities.
    @assert stats.br_norm_outliers == 30

    required_keys, enrollment = CascadeTables.neededkeys(combined)
    # 18841 before broad keys carried the species threshold: 447 keys were
    # shared by parents with equal (masses, width) but different lightest
    # channels elsewhere, and those are now — correctly — distinct decays.
    @assert length(required_keys) == 19288
    @assert enrollment == Dict(
        :self => 122,
        :closed => 0,
        :reopened => 0,
        :γdaughter => 1325,
    )

    dependencies = Set{Tuple{Int, Int}}()
    CascadeTables.eachchannelkey(combined) do parent_id, _channel, daughter_id, _key
        if daughter_id != parent_id && haskey(combined.byparent, daughter_id)
            push!(dependencies, (parent_id, daughter_id))
        end
    end
    cycle_log = IOBuffer()
    parent_order = with_logger(SimpleLogger(cycle_log, Logging.Warn)) do
        CascadeTables._cascadeorder(combined, dependencies)
    end
    @assert isempty(String(take!(cycle_log)))
    @assert length(parent_order) == length(combined.byparent)
    @assert Set(parent_order) == Set(keys(combined.byparent))
end
println("   PASS")


# ---------------------------------------------------------------- test 10
println("== 10. Thermal model: densities, statistics, widths, feed-down ==")
let
    data_directory = joinpath(REPO, "data")
    catalog = readparticles(joinpath(data_directory, "particles_light_pdg.dat"))
    pion = catalog.byid[211]
    proton = catalog.byid[2212]
    temperature = 0.1565

    # (a) Boltzmann limit. Quantum and classical statistics converge when
    # m/T is large, and separate for the pion -- where Bose sits *above*
    # Boltzmann because the occupancy 1/(e^x - 1) exceeds e^{-x}.
    heavy_quantum = thermaldensity(proton, temperature; quantum = true, width = false)
    heavy_classical = thermaldensity(proton, temperature; quantum = false, width = false)
    @assert abs(heavy_quantum - heavy_classical) / heavy_classical < 1.0e-3
    pion_quantum = thermaldensity(pion, temperature; quantum = true, width = false)
    pion_classical = thermaldensity(pion, temperature; quantum = false, width = false)
    @assert pion_quantum > pion_classical
    @assert (pion_quantum - pion_classical) / pion_classical > 0.02
    @printf("   pion Bose/Boltzmann = %.4f; proton agrees to %.2e\n",
            pion_quantum / pion_classical,
            abs(heavy_quantum - heavy_classical) / heavy_classical)

    # (b) Zero-width limit: averaging over a vanishing Breit-Wigner has to
    # reproduce the pole mass.
    rho = catalog.byid[213]
    narrow = typeof(rho)(
        rho.id, rho.name, rho.mass, 1.0e-9, rho.degeneracy, rho.baryon,
        rho.charge, rho.strangeness, rho.charm, rho.bottomness, rho.twiceisospin,
    )
    broadened = thermaldensity(narrow, temperature; width = true)
    sharp = thermaldensity(narrow, temperature; width = false)
    @assert abs(broadened - sharp) / sharp < 1.0e-6

    # A real width moves the density, but *which way* is not universal, and at
    # l = 0 neither is the ordering between the two shapes. Truncating at the
    # threshold removes low-mass weight while the tail adds high-mass weight;
    # which wins depends on where the threshold sits inside the window.
    # Measured here at T = 156.5 MeV, density relative to the pole mass:
    #
    #             massdep  constant
    #     rho      1.044    1.043
    #     sigma    0.551    0.457     threshold cuts deep into the window
    #     Delta    0.902    0.952
    #     K*       1.004    1.017
    #
    # So the assertions are per state. The sigma is the one that matters: the
    # two shapes differ by 21% there, and it is the state the whole threshold
    # treatment was built for.
    two_pion = 2 * 0.13957
    rho_sharp = thermaldensity(rho, temperature; width = false)
    rho_massdep = thermaldensity(rho, temperature; threshold = two_pion)
    rho_constant = thermaldensity(rho, temperature;
                                  shape = :constant, threshold = two_pion)
    @assert rho_massdep > rho_sharp && rho_constant > rho_sharp
    sigma = (mass = 0.475, width = 0.55, degeneracy = 1.0, baryon = 0,
             charge = 0, strangeness = 0, charm = 0, bottomness = 0)
    sigma_sharp = thermaldensity(sigma, temperature; width = false)
    sigma_massdep = thermaldensity(sigma, temperature; threshold = two_pion)
    sigma_constant = thermaldensity(sigma, temperature;
                                    shape = :constant, threshold = two_pion)
    @assert sigma_massdep < sigma_sharp && sigma_constant < sigma_sharp
    @assert (sigma_massdep - sigma_constant) / sigma_constant > 0.15
    @printf("   density/pole: rho %.4f, sigma %.4f (massdep) vs %.4f (constant)\n",
            rho_massdep / rho_sharp, sigma_massdep / sigma_sharp,
            sigma_constant / sigma_sharp)

    # ThermalModel builds DecaySpectra's shape types rather than transcribing
    # their formulas, so "the cascade and the yields average over one shape" is
    # true by construction. What still needs checking is that the arguments are
    # wired into the right slots -- a swapped mass/threshold would construct a
    # perfectly valid, wrong shape. Compared against the independently loaded
    # package-owned DecaySpectra instance, pointwise and exact.
    for (mR, G, mth) in ((0.7752, 0.1491, two_pion), (0.475, 0.55, two_pion),
                         (1.232, 0.117, 1.0777))
        lower, upper = max(mR - 4G, mth), mR + 4G
        for (choice, reference) in (
                (:massdep, DecaySpectra.MassDependentBW(
                    mR, G, mth; mmin = lower, mmax = upper)),
                (:constant, DecaySpectra.BreitWigner(
                    mR, G; mmin = lower, mmax = upper)),
            )
            built = ThermalModel._spectralshape(choice, mR, G, mth, lower, upper)
            for m in range(lower + 1.0e-9, upper; length = 41)
                @assert built(m) == reference(m)
            end
        end
    end
    # A pole at or below its own threshold has no q(mR): that species falls back
    # to the constant shape instead of throwing, matching decayfor. Checked by
    # exact shared type -- the package includes DecaySpectra only once.
    @assert typeof(ThermalModel._spectralshape(
        :massdep, 1.0, 0.2, 1.25, 1.25, 1.8,
    )) === BreitWigner
    @assert typeof(ThermalModel._spectralshape(
        :massdep, 1.0, 0.2, 0.5, 0.5, 1.8,
    )) === MassDependentBW

    # And the two quadratures built on that one shape must integrate it the
    # same way: DecaySpectra's psi-mapped parentmassnodes average of a smooth f
    # against ThermalPlan's Chebyshev rule.
    let mR = 0.475, G = 0.55, mth = two_pion, upper = mR + 4G
        ds_shape = DecaySpectra.MassDependentBW(
            mR, G, mth; mmin = mth, mmax = upper)
        masses, weights, normalization = DecaySpectra.parentmassnodes(
            ds_shape, DecaySpectra.Rules(nma = 48).ma, mth)
        smooth(m) = m^2 * exp(-m / temperature)
        ds_average = sum(weights .* smooth.(masses)) / normalization
        spectral = ThermalModel._spectralshape(:massdep, mR, G, mth, mth, upper)
        nodes, rule_weights = ThermalModel._spectralrule(spectral, 24)
        tm_average = sum(rule_weights .* smooth.(nodes))
        @assert abs(tm_average - ds_average) / ds_average < 1.0e-5
        @printf("   one shape, two quadratures: averages agree to %.1e\n",
                abs(tm_average - ds_average) / ds_average)
    end

    # A broad parent may reopen a channel to a daughter heavier than its pole.
    # Feed-down order therefore follows graph dependencies, not pole mass.
    let
        uphill_particles = Dict(
            1 => ParticleInfo(1, "A", 1.0, 0.5, 1.0, 0, 0),
            2 => ParticleInfo(2, "B", 2.0, 0.0, 1.0, 0, 0),
            3 => ParticleInfo(3, "C", 0.2, 0.0, 1.0, 0, 0),
            4 => ParticleInfo(4, "D", 0.1, 0.0, 1.0, 0, 0),
        )
        uphill_channels = Dict(
            1 => [DecayChannel(1, 1.0, [2, 4])],
            2 => [DecayChannel(2, 1.0, [3, 3])],
        )
        uphill = DecayList(
            uphill_particles, uphill_channels, "uphill", "uphill",
        )
        direct = thermalyields(
            uphill, 0.155; V = 1.0, quantum = false,
        )
        compiled = thermalyields(
            ThermalPlan(uphill), 0.155; V = 1.0, quantum = false,
        )
        expected_b = direct.primary[2] + direct.primary[1]
        expected_c = direct.primary[3] + 2 * expected_b
        compiled_expected_b = compiled.primary[2] + compiled.primary[1]
        compiled_expected_c = compiled.primary[3] + 2 * compiled_expected_b
        @assert isapprox(direct.total[2], expected_b; rtol = 1.0e-13)
        @assert isapprox(direct.total[3], expected_c; rtol = 1.0e-13)
        @assert isapprox(compiled.total[2], compiled_expected_b; rtol = 1.0e-13)
        @assert isapprox(compiled.total[3], compiled_expected_c; rtol = 1.0e-13)
    end

    # (c) Conservation under feed-down. Strong and electromagnetic decays
    # conserve B, Q and S, and every weak decayer in this list is terminal, so
    # the totals cannot move. This is checked on light_pdg: the quark-model
    # layer contains 103 neutral self-conjugate parents whose channel *sets*
    # are not charge-conjugation symmetric (e.g. rho(1921)3(0) gives BR 0.0110
    # to rho+ pi- but 0.0076 to rho- pi+, and its rho- pi+ pi0 channel has no
    # mirror at all). Each channel conserves charge, so `validate` accepts
    # them, but the yields then drift.
    decay_list = DecayList(
        catalog,
        readdecays(joinpath(data_directory, "decays_light_pdg.dat")),
    )
    yields = thermalyields(decay_list, temperature; muB = 0.0, V = 5300.0)
    two_body_decay_list = filter(nbody(2), decay_list)
    two_body_yields = thermalyields(
        two_body_decay_list, temperature; muB = 0.0, V = 5300.0,
    )
    @assert two_body_yields.primary == yields.primary
    records = decay_list.catalog.byid
    conserved(quantity, book) =
        sum(quantity(records[id]) * book[id] for id in keys(book))
    for (label, quantity) in (
            ("baryon", particle -> particle.baryon),
            ("charge", particle -> particle.charge),
            ("strangeness", particle -> particle.strangeness),
        )
        before = conserved(quantity, yields.primary)
        after = conserved(quantity, yields.total)
        scale = max(abs(before), sum(abs(quantity(records[id])) * yields.total[id]
                                     for id in keys(yields.total)), 1.0)
        @assert abs(after - before) / scale < 1.0e-12 "$label drifted: $before -> $after"
    end

    # The quark-model layer only conserves once charge conjugation is repaired.
    # tools/symmetrize_conjugation.jl averages each channel of a self-conjugate
    # parent with its mirror, which is why the yields below are unchanged while
    # the drift disappears.
    for (layer, tolerance) in (("light_qm", 1.0e-4), ("light_qm_csym", 1.0e-12))
        stacked = DecayList(
            merge(catalog, readparticles(joinpath(data_directory, "particles_light_qm.dat"))),
            merge(readdecays(joinpath(data_directory, "decays_light_pdg.dat")),
                  readdecays(joinpath(data_directory, "decays_$layer.dat"))),
        )
        stacked_yields = thermalyields(stacked, temperature; muB = 0.0, V = 5300.0)
        stacked_records = stacked.catalog.byid
        charge(book) = sum(stacked_records[id].charge * book[id] for id in keys(book))
        drift = abs(charge(stacked_yields.total) - charge(stacked_yields.primary))
        scale = sum(abs(stacked_records[id].charge) * stacked_yields.total[id]
                    for id in keys(stacked_yields.total))
        if layer == "light_qm_csym"
            @assert drift / scale < tolerance "repaired layer still drifts: $drift"
        else
            @assert drift / scale > 1.0e-6 "unrepaired layer unexpectedly conserves"
        end
    end
    @printf("   charge drift removed by symmetrize_conjugation\n")

    # (e) A compiled plan reproduces the decay-list path, but not bit-for-bit.
    # The plan averages each resonance on a fixed 32-node rule that carries its
    # spectral shape as the weight function; the direct path drives an adaptive
    # integral through the peak at the caller's 1e-6. The bound below is the
    # model's tolerance, not the quadrature's -- both shapes measure ~1e-7.
    plan = ThermalPlan(decay_list)
    two_body_plan = ThermalPlan(two_body_decay_list)
    @assert plan.shape === :massdep
    @assert two_body_plan.mass_range == plan.mass_range
    @assert two_body_plan.mass_nodes == plan.mass_nodes
    @assert two_body_plan.mass_weights == plan.mass_weights
    planned = thermalyields(plan, temperature; muB = 0.0, V = 5300.0)
    @assert Set(keys(planned.total)) == Set(keys(yields.total))
    for plan_shape in (:massdep, :constant)
        shaped_plan = plan_shape === :massdep ?
            plan : ThermalPlan(decay_list; shape = plan_shape)
        direct_yields = plan_shape === :massdep ? yields :
            thermalyields(decay_list, temperature;
                          muB = 0.0, V = 5300.0, shape = plan_shape)
        compiled_yields =
            thermalyields(shaped_plan, temperature; muB = 0.0, V = 5300.0)
        for book in (:primary, :total)
            direct = getfield(direct_yields, book)
            compiled = getfield(compiled_yields, book)
            deviation = maximum(
                abs(compiled[id] - direct[id]) / max(direct[id], 1.0e-300)
                for id in keys(direct)
            )
            @assert deviation < 1.0e-6 "plan ($plan_shape) differs from the direct path: $deviation"
        end
    end
    # and it must still track a changed temperature, not cache the first answer
    hotter = thermalyields(plan, 0.17; muB = 0.0, V = 5300.0)
    @assert hotter.total[211] > planned.total[211]
    @printf("   ThermalPlan agrees with the direct path (both shapes) to 1e-6\n")

    # Feed-down dominates the pion yield, as it must at freeze-out.
    @assert yields.total[211] > 2 * yields.primary[211]
    @printf("   feed-down conserves B, Q, S; pion total/primary = %.2f\n",
            yields.total[211] / yields.primary[211])

    # (d) Monotonicity: density rises with temperature, and baryons with muB.
    @assert thermaldensity(proton, 0.17) > thermaldensity(proton, 0.14)
    @assert thermaldensity(proton, temperature; muB = 0.05) >
            thermaldensity(proton, temperature; muB = 0.0)
    @assert thermaldensity(pion, temperature; muB = 0.05) ≈
            thermaldensity(pion, temperature; muB = 0.0)
end
println("   PASS")

println("\nAll tests passed.")
