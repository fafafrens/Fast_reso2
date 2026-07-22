"""
    DecaySpectra

Two- and three-body isotropic decay feed-down from `Maximum_Entropy_notes.pdf`
(equation numbers refer to the notes). Functor style: construct a decay once
(masses, branching ratio, Gauss–Legendre nodes), then call it like a function
on a parent spectrum:

    ρ = TwoBodyDecay(0.7752, 0.1396, 0.1396)          # a → b + c
    ω = ThreeBodyDecay(0.7827, 0.1396, 0.1396, 0.1350) # a → b + c + d
    Fb = ρ(Fa, pT, ηp, ϕp)                             # Eq. (32)
    Fb = ω(Fa, pT, ηp, ϕp)                             # Eq. (59)

`Fa(qT, ηq, ϕq)` is the invariant parent spectrum E dN_a/d³q (Eq. 14) and the
result is the invariant daughter spectrum F_b(pT, ηp, ϕp) of particle b.

Both decays share the kernel `decaykernel` — the qT × Δη Gauss–Legendre sum
of the square-root representation, common to Eqs. (32)/(41)/(59)/(61) — and
differ only in the rest-frame energy E_b of the daughter: fixed by the masses
for two-body (Eq. 5), averaged over the effective mass µ of the c+d pair
(weight p_b(µ)·p_c̃(µ)/N, Eqs. 43–48) for three-body.

Fourier modes (Eqs. 41 and 61, convention F = Σₙ Fₙ e^{inϕ}, Eq. 62) need no
separate code: the n-th daughter mode is obtained by calling the same functor
with `Fa = (qT, η, ϕ) -> Fan(qT, η) * cos(n*ϕ)` and `ϕp = 0`, which
reproduces the 2 Tₙ(Z) bracket identically.

Conventions: natural units (GeV), mostly-positive metric (Eq. 3), momenta
parametrized as in Eqs. (8)–(10). Requires pT > 0 and mb > 0.
"""
module DecaySpectra

using FastGaussQuadrature: gausslegendre

include("LagrangeWeights.jl")
using .LagrangeWeights

export kallen, TwoBodyDecay, ThreeBodyDecay, FourBodyDecay, DecayMatrix, decaymatrices,
       BreitWigner, BroadParent

"Källén triangle function λ(x, y, z), Eq. (7)."
kallen(x, y, z) = x^2 + y^2 + z^2 - 2x * y - 2x * z - 2y * z

# Cosine-mapped Gauss–Legendre rule for ∫ₐᵇ f dx with x = mid − half·cosθ:
# returns (cosθᵢ, (π/2)·wᵢ·sinθᵢ) so that ∫ₐᵇ f ≈ half·Σᵢ vᵢ f(mid − half·cᵢ).
# The sinθ Jacobian (inside vᵢ) absorbs 1/√ endpoint singularities of f.
# Memoized per order and shared across all decay objects — the vectors are
# never mutated after creation, so sharing is safe (also under the threaded
# assemblies); the lock covers concurrent first-time construction.
const _COSNODES = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const _COSNODES_LOCK = ReentrantLock()
function cosnodes(n::Int)
    lock(_COSNODES_LOCK) do
        get!(_COSNODES, n) do
            x, w = gausslegendre(n)
            θ = @. π * (x + 1) / 2
            (cos.(θ), @. π / 2 * w * sin(θ))
        end
    end
end

# ---------------------------------------------------------------------------
# Common decay kernel
# ---------------------------------------------------------------------------

# K(Eb) = ∫ qT dqT ∫ dηq Θ(Q²−D²)/√(Q²−D²) [Fa(ϕp+α) + Fa(ϕp−α)]
# for a daughter b emitted with rest-frame energy Eb (momentum p_b = √(Eb²−mb²))
# from parent a; M = maT·mbT, Q = qT·pT, C = ma·Eb, D = M cosh(Δη) − C
# (Eq. 29), Z = D/Q (Eq. 30), α = arccos Z (Eq. 33). This is the double
# integral common to Eqs. (32), (41), (59), (61), evaluated as one
# tensor-product sum on the precomputed cosine-mapped nodes (cq, vq) × (cθ, vθ).
# The qT integrand has an integrable ln|qT − qT*| singularity where the Δη
# domain splits; the interval is divided there, leaving a mild θ·lnθ endpoint
# behavior that converges at ~nq⁻⁴ (nq = 128 gives ≲1e-7).
function decaykernel(Fa::F, pT, ηp, ϕp, ma, mb, Eb, cq, vq, cθ, vθ) where {F}
    pb = sqrt(max(Eb^2 - mb^2, 0.0))
    C = ma * Eb
    mbT = hypot(mb, pT)

    # qT support of the Θ function (maT·mbT ≤ C + qT·pT): [qT−, qT+] with
    # qT± = ma (Eb pT ± pb mbT)/mb²; the log singularity sits at qT* = −qT−.
    qlo = max(ma * (Eb * pT - pb * mbT) / mb^2, 0.0)
    qhi = ma * (Eb * pT + pb * mbT) / mb^2
    qstar = ma * (pb * mbT - Eb * pT) / mb^2
    segments = qlo < qstar < qhi ? ((qlo, qstar), (qstar, qhi)) :
                                   ((qlo, qhi), (qhi, qhi))      # 2nd is empty

    K = 0.0
    for (a, b) in segments
        a < b || continue
        hq, mq = (b - a) / 2, (a + b) / 2
        n = length(cq)
        nt = min(Threads.nthreads(), n)
        if nt == 1
            K += qnodes(Fa, pT, ηp, ϕp, ma, C, mbT, hq, mq, cq, vq, cθ, vθ, 1:n)
        else   # qT nodes are independent: strided chunks on nt tasks
            tasks = [Threads.@spawn qnodes(Fa, pT, ηp, ϕp, ma, C, mbT,
                                           hq, mq, cq, vq, cθ, vθ, k:nt:n)
                     for k in 1:nt]
            for t in tasks
                K += fetch(t)::Float64
            end
        end
    end
    return K
end

# Partial qT sum over the nodes `is` of one segment (all of them, or a
# strided chunk of a threaded split). Body of the tensor-product sum;
# @inline matters: it lets the serial call site specialize on the unit-stride
# range (~1.4× on the single-thread path).
@inline function qnodes(Fa::F, pT, ηp, ϕp, ma, C, mbT, hq, mq, cq, vq, cθ, vθ,
                        is::AbstractRange{Int}) where {F}
    K = 0.0
    @inbounds for i in is
        qT = mq - hq * cq[i]
        M = hypot(ma, qT) * mbT
        Q = qT * pT
        # Δη support: cosh(Δη) ∈ [max(1, (C−Q)/M), (C+Q)/M] (Eq. 21)
        chmax = (C + Q) / M
        chmax > 1 || continue
        ηhi = acosh(chmax)
        chmin = (C - Q) / M
        ηlo = chmin > 1 ? acosh(chmin) : 0.0
        hη, mη = (ηhi - ηlo) / 2, (ηlo + ηhi) / 2
        s = 0.0
        @inbounds for j in eachindex(cθ)
            Δη = mη - hη * cθ[j]
            D = M * cosh(Δη) - C                     # Eq. (29)
            S2 = (Q - D) * (Q + D)                   # = Q²(1−Z²), Eq. (31)
            S2 > 0 || continue
            α = acos(clamp(D / Q, -1, 1))            # Eqs. (30), (33)
            # both Δη signs (Fa need not be symmetric in rapidity)
            s += vθ[j] / sqrt(S2) *
                 (Fa(qT, ηp + Δη, ϕp + α) + Fa(qT, ηp + Δη, ϕp - α) +
                  Fa(qT, ηp - Δη, ϕp + α) + Fa(qT, ηp - Δη, ϕp - α))
        end
        K += vq[i] * hq * qT * hη * s
    end
    return K
end

# ---------------------------------------------------------------------------
# Two-body decay a → b + c, Eq. (32)
# ---------------------------------------------------------------------------

"""
    TwoBodyDecay(ma, mb, mc; B = 1.0, nq = 128, nθ = 32)

Callable decay map of the isotropic two-body decay a → b + c with branching
ratio `B`; `(d::TwoBodyDecay)(Fa, pT, ηp, ϕp)` returns the daughter spectrum
of b, Eq. (32). `nq`, `nθ` are the Gauss–Legendre orders in qT and Δη
(defaults converged to ≲1e-7 for thermal-like sources).
"""
struct TwoBodyDecay
    ma::Float64
    mb::Float64
    mc::Float64
    B::Float64
    Eb::Float64                  # E^a_{b|c}, Eq. (5)
    pb::Float64                  # p^a_{b|c}, Eq. (6)
    cq::Vector{Float64}          # cosine-mapped qT nodes/weights (cosnodes)
    vq::Vector{Float64}
    cθ::Vector{Float64}          # cosine-mapped Δη nodes/weights (cosnodes)
    vθ::Vector{Float64}
end

function TwoBodyDecay(ma, mb, mc; B = 1.0, nq = 128, nθ = 32)
    ma ≥ mb + mc || throw(ArgumentError("decay closed: ma < mb + mc"))
    mb > 0 || throw(ArgumentError("massless daughter b not supported"))
    Eb = (ma^2 + mb^2 - mc^2) / (2ma)
    pb = sqrt(max(kallen(ma^2, mb^2, mc^2), 0.0)) / (2ma)
    TwoBodyDecay(ma, mb, mc, B, Eb, pb, cosnodes(nq)..., cosnodes(nθ)...)
end

(d::TwoBodyDecay)(Fa::F, pT, ηp, ϕp) where {F} =
    d.B * d.ma / (4π * d.pb) *                           # Eq. (32)
    decaykernel(Fa, pT, ηp, ϕp, d.ma, d.mb, d.Eb, d.cq, d.vq, d.cθ, d.vθ)

# ---------------------------------------------------------------------------
# Effective-mass (µ-averaged) decays: three- and four-body
# ---------------------------------------------------------------------------
# Both cluster the unobserved daughters into a single effective particle of
# mass µ and average the two-body kernel over µ.  They share the functor and
# the matrix assembly (which use only the fields ma, mb, B, Eb, coef, cq…vθ)
# and differ solely in how the per-µ weight `coef` is built: the cluster
# invariant-mass density ρ_{n-1}(µ) — the pair density q(µ;mc,md) for
# three-body, the nested ρ₃(µ) for four-body (matrix_notes §3.1).

abstract type MuAveraged end

# Shared functor: B·ma/(4π) · Σᵢ coefᵢ K(E_b(µᵢ)), Eq. (59) and its 4-body analog.
(d::MuAveraged)(Fa::F, pT, ηp, ϕp) where {F} =
    d.B * d.ma / (4π) *
    sum(d.coef[i] * decaykernel(Fa, pT, ηp, ϕp, d.ma, d.mb, d.Eb[i],
                                d.cq, d.vq, d.cθ, d.vθ) for i in eachindex(d.coef))

# Shared µ-average build (Eqs. 45–48): node energies E_b(µᵢ) and normalized
# weights coefᵢ = vᵢ ρ(µᵢ)/N, N = Σᵢ vᵢ p_b(µᵢ) ρ(µᵢ), on nµ cosine-mapped
# nodes µ ∈ [lo, ma−mb]. `density(µ)` is the cluster invariant-mass density —
# the pair density q(µ;mc,md) for three-body, ρ₃(µ) for four-body — the only
# thing that differs between them. `where {D}`: specialize on the density
# closure so its call inlines (construction runs ~13k times per table build).
function muweights(ma, mb, lo, density::D, nµ) where {D}
    cµ, vµ = cosnodes(nµ)                # cosine map absorbs the √ endpoint zeros
    hi = ma - mb
    h, m = (hi - lo) / 2, (hi + lo) / 2
    Eb = Vector{Float64}(undef, nµ)
    coef = Vector{Float64}(undef, nµ)
    norm = 0.0
    for i in 1:nµ
        µ = m - h * cµ[i]
        Eb[i] = (ma^2 + mb^2 - µ^2) / (2ma)                    # Eq. (45)
        pb = sqrt(max(kallen(ma^2, mb^2, µ^2), 0.0)) / (2ma)   # Eq. (46)
        ρ = density(µ)                                         # Eq. (47) / ρ₃
        coef[i] = vµ[i] * ρ
        norm += vµ[i] * pb * ρ                                 # → N, Eq. (48)
    end
    coef ./= norm
    return Eb, coef
end

# ---------------------------------------------------------------------------
# Three-body decay a → b + c + d, Eq. (59)
# ---------------------------------------------------------------------------

"""
    ThreeBodyDecay(ma, mb, mc, md; B = 1.0, nµ = 32, nq = 128, nθ = 32)

Callable decay map of the isotropic three-body decay a → b + c + d with
branching ratio `B`; `(d::ThreeBodyDecay)(Fa, pT, ηp, ϕp)` returns the
daughter spectrum of b, Eq. (59). The pair c+d acts as an effective particle
of mass µ ∈ [mc+md, ma−mb] (Eqs. 43–44): the same kernel as two-body is
averaged over `nµ` cosine-mapped Gauss–Legendre nodes µᵢ with rest-frame
energies E_b(µᵢ) (Eq. 45) and weights ∝ p_c̃(µᵢ) (Eq. 47), normalized by
N = ∫ dµ p_b p_c̃ (Eq. 48).
"""
struct ThreeBodyDecay <: MuAveraged
    ma::Float64
    mb::Float64
    mc::Float64
    md::Float64
    B::Float64
    Eb::Vector{Float64}          # E^a_b(µᵢ), Eq. (45)
    coef::Vector{Float64}        # vᵢ p_c̃(µᵢ) / N
    cq::Vector{Float64}
    vq::Vector{Float64}
    cθ::Vector{Float64}
    vθ::Vector{Float64}
end

function ThreeBodyDecay(ma, mb, mc, md; B = 1.0, nµ = 32, nq = 128, nθ = 32)
    ma ≥ mb + mc + md || throw(ArgumentError("decay closed: ma < mb + mc + md"))
    mb > 0 || throw(ArgumentError("massless daughter b not supported"))
    pair(µ) = sqrt(max(kallen(µ^2, mc^2, md^2), 0.0)) / (2µ)   # ρ₂ = p_c̃(µ), Eq. (47)
    Eb, coef = muweights(ma, mb, mc + md, pair, nµ)
    ThreeBodyDecay(ma, mb, mc, md, B, Eb, coef, cosnodes(nq)..., cosnodes(nθ)...)
end

# ---------------------------------------------------------------------------
# Four-body decay a → b + c + d + e (matrix_notes §3.1)
# ---------------------------------------------------------------------------

# Cluster invariant-mass density ρ₃(µ) = ∫_{md+me}^{µ−mc} dσ q(µ;mc,σ) q(σ;md,me),
# the three-particle (cde) phase-space volume (Dalitz area) at cluster mass µ.
# One cosine-mapped Gauss sweep over the sub-mass σ; the map absorbs the √
# endpoint zeros of q at σ = md+me and σ = µ−mc. Kernel-free, hence cheap.
@inline function clusterdensity3(µ, mc, md, me, cσ, vσ)
    lo, hi = md + me, µ - mc
    hi > lo || return 0.0
    h, m = (hi - lo) / 2, (hi + lo) / 2
    s = 0.0
    @inbounds for k in eachindex(cσ)
        σ = m - h * cσ[k]
        qµ = sqrt(max(kallen(µ^2, mc^2, σ^2), 0.0)) / (2µ)   # q(µ; mc, σ)
        qσ = sqrt(max(kallen(σ^2, md^2, me^2), 0.0)) / (2σ)  # q(σ; md, me)
        s += vσ[k] * qµ * qσ
    end
    return h * s
end

"""
    FourBodyDecay(ma, mb, mc, md, me; B = 1.0, nµ = 32, nσ = 16, nq = 128, nθ = 32)

Callable decay map of the isotropic four-body decay a → b + c + d + e. Identical
in structure to `ThreeBodyDecay` — the same kernel averaged over the cluster
mass µ ∈ [mc+md+me, ma−mb] — but the per-µ weight uses the three-particle
cluster density ρ₃(µ) (an inner `nσ`-point σ-integral) in place of the pair
density. Normalized by N = ∫ dµ p_b ρ₃ (matrix_notes §3.1).
"""
struct FourBodyDecay <: MuAveraged
    ma::Float64
    mb::Float64
    mc::Float64
    md::Float64
    me::Float64
    B::Float64
    Eb::Vector{Float64}          # E^a_b(µᵢ)
    coef::Vector{Float64}        # vᵢ ρ₃(µᵢ) / N
    cq::Vector{Float64}
    vq::Vector{Float64}
    cθ::Vector{Float64}
    vθ::Vector{Float64}
end

function FourBodyDecay(ma, mb, mc, md, me; B = 1.0, nµ = 32, nσ = 16, nq = 128, nθ = 32)
    ma ≥ mb + mc + md + me || throw(ArgumentError("decay closed: ma < mb + mc + md + me"))
    mb > 0 || throw(ArgumentError("massless daughter b not supported"))
    cσ, vσ = cosnodes(nσ)
    ρ3(µ) = clusterdensity3(µ, mc, md, me, cσ, vσ)            # ρ₃(µ), inner σ-integral
    Eb, coef = muweights(ma, mb, mc + md + me, ρ3, nµ)
    FourBodyDecay(ma, mb, mc, md, me, B, Eb, coef, cosnodes(nq)..., cosnodes(nθ)...)
end

# ---------------------------------------------------------------------------
# Finite parent width: average the decay matrix over the parent spectral function
# ---------------------------------------------------------------------------

"""
    BreitWigner(mR, Γ; k = 4.0, mmin = mR - k*Γ, mmax = mR + k*Γ)

Relativistic (constant-width) Breit–Wigner spectral function of a resonance,
`ϱ(m) ∝ m²Γ / ((m²−mR²)² + mR²Γ²)`, truncated to `[mmin, mmax]` and zero
outside. Unnormalized: callers that average over it (`widthaverage`) divide by
the integrated weight, so only the shape matters.
"""
struct BreitWigner
    mR::Float64
    Γ::Float64
    mmin::Float64
    mmax::Float64
end
BreitWigner(mR, Γ; k = 4.0, mmin = mR - k * Γ, mmax = mR + k * Γ) =
    BreitWigner(mR, Γ, mmin, mmax)
@inline (bw::BreitWigner)(m) =
    bw.mmin ≤ m ≤ bw.mmax ?
    m^2 * bw.Γ / ((m^2 - bw.mR^2)^2 + bw.mR^2 * bw.Γ^2) : 0.0

# Parent-mass nodes for the width average. The tan-substitution
# s = mR² + mR·Γ·tanψ flattens the Breit–Wigner Lorentzian (∫ϱ(m)·f dm ∝
# ∫√s·f dψ, smooth in ψ), so plain Gauss–Legendre in ψ resolves the central
# peak for any Γ — a cosine map would starve it (nodes cluster at the endpoints,
# not the peak). Node weight ∝ √s = m_a; `norm` sums over the full support
# (incl. closed nodes) so a channel closed at the pole keeps only its open-tail
# fraction. Returns all nodes, weights, and the full-support norm.
function parentmassnodes(bw::BreitWigner, nma)
    x, u = gausslegendre(nma)
    ψlo = atan((bw.mmin^2 - bw.mR^2) / (bw.mR * bw.Γ))
    ψhi = atan((bw.mmax^2 - bw.mR^2) / (bw.mR * bw.Γ))
    hψ, mψ = (ψhi - ψlo) / 2, (ψhi + ψlo) / 2
    mas, ws, norm = Float64[], Float64[], 0.0
    for i in 1:nma
        s = bw.mR^2 + bw.mR * bw.Γ * tan(mψ + hψ * x[i])
        s > 0 || continue
        ma, w = sqrt(s), u[i] * sqrt(s)
        push!(mas, ma); push!(ws, w); norm += w
    end
    return mas, ws, norm
end

"""
    BroadParent(daughter masses..., bw::BreitWigner; B = 1.0, nma = 16, kw...)

A decay whose parent has finite width: it wraps the underlying sharp decay
(`TwoBodyDecay`/`ThreeBodyDecay`/`FourBodyDecay`, chosen by the number of
daughter masses) and averages it over the parent spectral function `bw`
(pole `bw.mR`, width `bw.Γ`). Usable exactly like a sharp decay — as a functor
`(d)(Fa, pT, ηp, ϕp)` or via `DecayMatrix`/`decaymatrices` — the average being
`Σᵢ wᵢ ϱ(m_aⁱ) [·] / N` over the parent mass. Nodes below the daughter
threshold are dropped, so a channel closed at the pole keeps only its open-tail
fraction (channel reopening). `kw...` pass through to the underlying decay
(e.g. `nµ, nσ, nq, nθ`).
"""
struct BroadParent{D}
    subs::Vector{D}       # sharp decays at the open parent-mass nodes
    ws::Vector{Float64}   # their ϱ weights
    norm::Float64         # full-support normalization (incl. closed nodes)
end

function BroadParent(build, bw::BreitWigner, opensum; nma = 16)
    mas, wall, norm = parentmassnodes(bw, nma)
    keep = findall(>(opensum), mas)               # drop closed nodes
    BroadParent([build(mas[i]) for i in keep], wall[keep], norm)
end

BroadParent(mb, mc, bw::BreitWigner; B = 1.0, nma = 16, kw...) =
    BroadParent(ma -> TwoBodyDecay(ma, mb, mc; B, kw...), bw, mb + mc; nma)
BroadParent(mb, mc, md, bw::BreitWigner; B = 1.0, nma = 16, kw...) =
    BroadParent(ma -> ThreeBodyDecay(ma, mb, mc, md; B, kw...), bw, mb + mc + md; nma)
BroadParent(mb, mc, md, me, bw::BreitWigner; B = 1.0, nma = 16, kw...) =
    BroadParent(ma -> FourBodyDecay(ma, mb, mc, md, me; B, kw...), bw, mb + mc + md + me; nma)

# functor: ϱ-average of the wrapped decay's functor
(d::BroadParent)(Fa::F, pT, ηp, ϕp) where {F} =
    sum(d.ws[i] * d.subs[i](Fa, pT, ηp, ϕp) for i in eachindex(d.subs); init = 0.0) / d.norm

# matrix: ϱ-average of the wrapped decay's matrices (one per mode). Two typed
# dispatches (Vector / Tuple modes) resolve the ambiguity with the generic
# modes-vector entry; both share this body.
function _bpmatrices(d::BroadParent, modes, pTs, qgrid, P)
    Ws = [zeros(length(pTs), length(qgrid)) for _ in modes]
    for i in eachindex(d.subs)
        for (k, Wi) in enumerate(decaymatrices(d.subs[i], modes, pTs, qgrid; P))
            Ws[k] .+= d.ws[i] .* Wi
        end
    end
    d.norm > 0 && foreach(W -> W ./= d.norm, Ws)
    return Ws
end
decaymatrices(d::BroadParent, modes::AbstractVector{<:Integer}, pTs, qgrid; P = Val(4)) =
    _bpmatrices(d, modes, pTs, qgrid, P)
decaymatrices(d::BroadParent, modes::Tuple, pTs, qgrid; P = Val(4)) =
    _bpmatrices(d, modes, pTs, qgrid, P)

# ---------------------------------------------------------------------------
# Matrix representation on a fixed parent grid (η-independent sources)
# ---------------------------------------------------------------------------

"""
    DecayMatrix(decay, n, pTs, qgrid; P = Val(4))

Precompute the decay map of Fourier mode `n` (Eq. 41 / Eq. 61) as a matrix W
over a fixed parent-qT grid, exploiting linearity in the parent spectrum:

    M = DecayMatrix(ρ, 0, pTs, range(0, 4.2, length = 121))
    Fbn = M(Fan)                  # Fan(qT) callable, or a vector on the grid
    Fbn = M.W * Fan.(M.q)         # same thing, explicit GEMV

Each row runs the same quadrature as the functor call, but instead of
evaluating the source it scatters the quadrature factor × 2Tₙ(Z) onto the
Lagrange stencil (`LagrangeWeights`, order `P`, default cubic) of each qT
node — so `M(Fan)` reproduces `decay(...)` applied to the interpolant of
`Fan` through the grid values, and the only new error is O(h^P) interpolation.

Restrictions: the parent spectrum must be η-independent (boost-invariant) —
the η dependence was folded into W. Quadrature nodes falling outside `qgrid`
are treated as F_a = 0, i.e. the source must be negligible beyond the grid.
`qgrid` may be an `AbstractRange` (uniform) or a sorted vector (non-uniform).
"""
struct DecayMatrix
    n::Int
    pT::Vector{Float64}
    q::Vector{Float64}
    W::Matrix{Float64}
end

(M::DecayMatrix)(F::AbstractVector) = M.W * F
(M::DecayMatrix)(Fan) = M.W * map(Fan, M.q)

_stencil(x, g::AbstractRange, P) = lagrangeweights(x, first(g), step(g), length(g), P)
_stencil(x, g::AbstractVector, P) = lagrangeweights(x, g, P)

# ---------------------------------------------------------------------------
# Fused multi-mode assembly: every Fourier mode in one quadrature sweep. The
# qT/Z geometry, the R(Z) factors and the Lagrange stencil are mode-independent
# — only the Chebyshev factor Tₙ(Z) differs — so building the modes together
# shares that work. Modes are carried as a Tuple, so the per-mode accumulators
# live on the stack (no heap, no per-thread scratch); one recurrence supplies
# the whole Chebyshev string for the requested modes at each node.
# Rows are independent (each thread writes only its own row of every matrix),
# so the assembly threads over daughter pT and is bitwise identical to serial.
# ---------------------------------------------------------------------------

# Core (mt a Tuple of modes) → one Float64 matrix per mode.
function decaymatrices(d::TwoBodyDecay, mt::Tuple, pTs, qgrid; P = Val(4))
    pTv = collect(Float64, pTs)
    Ws = map(_ -> zeros(length(pTv), length(qgrid)), mt)
    nθ, pref = length(d.cθ), d.B * d.ma / (4π * d.pb)
    Threads.@threads for k in eachindex(pTv)
        scatterrows!(map(W -> view(W, k, :), Ws), mt, d.ma, d.mb, d.Eb, pTv[k],
                     d.cq, d.vq, nθ, qgrid, P, pref)
    end
    return collect(Ws)
end

# Three- and four-body: the µ-average is a coef-weighted sum of the same scatter.
function decaymatrices(d::MuAveraged, mt::Tuple, pTs, qgrid; P = Val(4))
    pTv = collect(Float64, pTs)
    Ws = map(_ -> zeros(length(pTv), length(qgrid)), mt)
    nθ, base = length(d.cθ), d.B * d.ma / (4π)
    Threads.@threads for k in eachindex(pTv)
        rows = map(W -> view(W, k, :), Ws)
        for i in eachindex(d.coef)                          # µ-average
            scatterrows!(rows, mt, d.ma, d.mb, d.Eb[i], pTv[k],
                         d.cq, d.vq, nθ, qgrid, P, base * d.coef[i])
        end
    end
    return collect(Ws)
end

# Entry from a modes vector, with a fallback for many modes (tuple ops stop
# unrolling past ~32, so cap the fused path and loop single-mode beyond that).
function decaymatrices(d, modes::AbstractVector{<:Integer}, pTs, qgrid; P = Val(4))
    length(modes) ≤ 16 && return decaymatrices(d, Tuple(modes), pTs, qgrid; P)
    return [DecayMatrix(d, n, pTs, qgrid; P).W for n in modes]
end

# Single-mode public API: one DecayMatrix struct, via the fused core with (n,).
DecayMatrix(d::Union{TwoBodyDecay,MuAveraged}, n::Integer, pTs, qgrid; P = Val(4)) =
    DecayMatrix(n, collect(Float64, pTs), collect(Float64, qgrid),
                decaymatrices(d, (n,), pTs, qgrid; P)[1])

# broad-parent wrapper (defined here, after the DecayMatrix struct exists)
DecayMatrix(d::BroadParent, n::Integer, pTs, qgrid; P = Val(4)) =
    DecayMatrix(n, collect(Float64, pTs), collect(Float64, qgrid),
                decaymatrices(d, (n,), pTs, qgrid; P)[1])

# Chebyshev–Gauss nodes Z_k = cos((2k−1)π/(2n)), memoized like cosnodes.
const _CHEBNODES = Dict{Int,Vector{Float64}}()
chebnodes(n::Int) = lock(_COSNODES_LOCK) do
    get!(() -> [cos((2k - 1) * π / (2n)) for k in 1:n], _CHEBNODES, n)
end

# Tₙ(Z) = cos(n·acos Z) by the Chebyshev recurrence (Eq. 40): the assembly
# needs only this polynomial, never the angle itself, so no acos/cos here.
@inline function chebT(n::Int, Z::Float64)
    n == 0 && return 1.0
    Tm1, Tm = 1.0, Z
    for _ in 2:n
        Tm1, Tm = Tm, 2 * Z * Tm - Tm1
    end
    return Tm
end

# Tₘ(Z) for every requested mode m, as a tuple. `map` over the mode tuple
# unrolls; chebT is cheap, so recomputing per mode is fine — the shared cost
# is the √, the geometry and the stencil, all hoisted out of the mode loop.
@inline chebvals(modes::Tuple, Z) = map(n -> chebT(n, Z), modes)

# Multi-mode scatter: mirrors decaykernel's qT structure but scatters the
# quadrature factor onto the parent-grid stencil instead of evaluating Fa. The
# inner integral is done in Z = D/Q rather than Δη (an η-independent source never
# needs the rapidity), so everything is algebraic (matrix_notes §8):
#     2 ∫dΔη Θ/√(Q²−D²) · 2Tₙ(Z)  =  (4/M) ∫_{Zlo}^{1} Tₙ(Z) dZ / √((1−Z²)(c²−1)),
# c = (C+QZ)/M, Zlo = (M−C)/Q. Both windows use a Chebyshev–Gauss rule with the
# inverse-√ weight removed analytically (R = √(c²−1) full, √((Q/M)(1+Z)(c+1))
# partial), so no positivity guard. All of that is mode-independent; only Tₙ(Z)
# varies. So K modes accumulate together in the stack tuple `s` and scatter onto
# their K rows through the one shared stencil — each row bit-identical to a
# single-mode build.
function scatterrows!(rows, modes::NTuple{K,Int}, ma, mb, Eb, pT, cq, vq, nθ,
                      qgrid, P, pref) where {K}
    pb = sqrt(max(Eb^2 - mb^2, 0.0))
    C = ma * Eb
    mbT = hypot(mb, pT)
    qlo = max(ma * (Eb * pT - pb * mbT) / mb^2, 0.0)
    qhi = ma * (Eb * pT + pb * mbT) / mb^2
    qstar = ma * (pb * mbT - Eb * pT) / mb^2
    segments = qlo < qstar < qhi ? ((qlo, qstar), (qstar, qhi)) :
                                   ((qlo, qhi), (qhi, qhi))
    qmin, qmax = first(qgrid), last(qgrid)
    zc = chebnodes(nθ)
    zero_s = ntuple(_ -> 0.0, Val(K))
    for (a, b) in segments
        a < b || continue
        hq, mq = (b - a) / 2, (a + b) / 2
        @inbounds for i in eachindex(cq)
            qT = mq - hq * cq[i]
            qmin ≤ qT ≤ qmax || continue           # F_a ≡ 0 off the grid
            M = hypot(ma, qT) * mbT
            Q = qT * pT
            M < C + Q || continue                  # Θ support (Z window nonempty)
            Zlo = (M - C) / Q
            s = zero_s
            if Zlo ≤ -1                            # full window: Chebyshev–Gauss
                z = Zlo > -1.5 ? chebnodes(4 * nθ) : zc   # densify near the transition
                for Z in z
                    c = (C + Q * Z) / M
                    s = s .+ chebvals(modes, Z) ./ sqrt((c - 1) * (c + 1))
                end
                s = s .* (π / length(z))
            else                                   # partial window: shifted Chebyshev–Gauss
                # analytic cancellation (matrix_notes §8.2): endpoint weight removed,
                # regular R² = (Q/M)(1+Z)(c+1) > 0, no guard; densify near Zlo→−1.
                zp = Zlo < -0.5 ? chebnodes(4 * nθ) : zc
                hz, mz = (1 - Zlo) / 2, (1 + Zlo) / 2
                for ζ in zp
                    Z = mz + hz * ζ
                    c = (C + Q * Z) / M
                    s = s .+ chebvals(modes, Z) ./ sqrt((Q / M) * (1 + Z) * (c + 1))
                end
                s = s .* (π / length(zp))
            end
            fac0 = pref * vq[i] * hq * qT * (4 / M)
            j0, w = _stencil(qT, qgrid, P)
            for kk in 1:K
                fac = fac0 * s[kk]
                for (m, wm) in pairs(w)
                    rows[kk][j0 + m - 1] += fac * wm
                end
            end
        end
    end
    return rows
end

end # module
