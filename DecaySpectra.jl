"""
    DecaySpectra

Isotropic decay feed-down — two-, three- and four-body, with optional finite
parent width — derived in `matrix_notes.tex`; equation numbers below refer to
`Maximum_Entropy_notes.pdf`, the original derivation. Functor style: construct a
decay once (masses, branching ratio, Gauss–Legendre nodes), then call it like a
function on a parent spectrum:

    ρ = TwoBodyDecay(0.7752, 0.1396, 0.1396)           # a → b + c
    ω = ThreeBodyDecay(0.7827, 0.1396, 0.1396, 0.1350) # a → b + c + d
    ρw = BroadParent(0.1396, 0.1396, BreitWigner(0.7752, 0.149))   # broad a
    Fb = ρ(Fa, pT, ηp, ϕp)                             # Eq. (32)
    Fb = ω(Fa, pT, ηp, ϕp)                             # Eq. (59)

`Fa(qT, ηq, ϕq)` is the invariant parent spectrum E dN_a/d³q (Eq. 14) and the
result is the invariant daughter spectrum F_b(pT, ηp, ϕp) of particle b.

Every decay is the same type, [`MassAverage`]: a coef-weighted sum of the shared
two-body kernel `decaykernel` (the qT × Δη Gauss–Legendre sum of the square-root
representation, common to Eqs. 32/41/59/61) over nodes that each carry their own
parent mass and daughter rest-frame energy E_b. Only those nodes differ between
cases — one node for two-body (E_b fixed by the masses, Eq. 5); nµ nodes over the
c+d effective mass for three-body (weight p_b(µ)·p_c̃(µ)/N, Eqs. 43–48) and its
four-body analog; and, for a broad parent, those same nodes spread over the
parent spectral function (matrix_notes §4).

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

export kallen, MassAverage, TwoBodyDecay, ThreeBodyDecay, FourBodyDecay,
       DecayMatrix, decaymatrices, BreitWigner, BroadParent

"Källén triangle function λ(x, y, z), Eq. (7)."
kallen(x, y, z) = x^2 + y^2 + z^2 - 2x * y - 2x * z - 2y * z

# Cosine-mapped Gauss–Legendre rule for ∫ₐᵇ f dx with x = mid − half·cosθ:
# returns (cosθᵢ, (π/2)·wᵢ·sinθᵢ) so that ∫ₐᵇ f ≈ half·Σᵢ vᵢ f(mid − half·cᵢ).
# The sinθ Jacobian (inside vᵢ) absorbs 1/√ endpoint singularities of f.
#
# Pure — computed fresh per call, stored in the decay object that asked. This
# replaced a memoized global cache whose lock-free reads required all sizes to
# be pre-warmed before any threaded build (a latent data race when a size was
# missed, e.g. nma ∉ {nq, nθ, nµ, nσ}). Measured at 8 threads on the full table
# build, fresh computation costs ~2% (gausslegendre is 0.5–7 µs and runs once
# per construction, never in the assembly's hot loops) — not worth any
# concurrency machinery.
function cosnodes(n::Int)
    x, w = gausslegendre(n)
    θ = @. π * (x + 1) / 2
    (cos.(θ), @. π / 2 * w * sin(θ))
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
# One decay map: a coef-weighted sum of two-body kernels at per-node (ma, Eb)
# ---------------------------------------------------------------------------
# Every isotropic decay is the two-body kernel of an observed daughter b (mass
# mb) recoiling against an effective particle, summed over nodes:
#     F_b  =  Σᵢ coefᵢ · K(Fa; ma_i, mb, Eb_i).
# What varies per node is the parent mass ma_i and the recoil energy Eb_i;
# coefᵢ carries B, the 1/p_b Jacobian, and any averaging weight. This one type
# covers every case (constructors below build the (ma, Eb, coef) arrays):
#   • two-body  a→bc        — one node (Eq. 32);
#   • 3-/4-body a→b(c…)     — nµ nodes, ma fixed, Eb over the cluster mass µ,
#                              coef ∝ ρ_{n-1}(µ) (Eq. 59 and its 4-body analog);
#   • broad parent          — the above with ma_i varying over the parent
#                              spectral function ϱ_a (matrix_notes §4).
# Nothing downstream needs to know which decay it was: the functor and the
# matrix assembly read only (ma, mb, Eb, coef, cq…vθ).

struct MassAverage
    ma::Vector{Float64}          # per-node parent mass
    mb::Float64                  # observed daughter (fixed)
    Eb::Vector{Float64}          # per-node E_b(node)
    coef::Vector{Float64}        # per-node weight: B·ma·[1/p_b or ρ/N]/(4π)
    cq::Vector{Float64}          # cosine-mapped qT nodes/weights (cosnodes)
    vq::Vector{Float64}
    cθ::Vector{Float64}          # Δη nodes/weights (cosnodes; length sets nθ)
    vθ::Vector{Float64}
end

# functor:  Σᵢ coefᵢ · K(Fa; ma_i, mb, Eb_i)
(d::MassAverage)(Fa::F, pT, ηp, ϕp) where {F} =
    sum(d.coef[i] * decaykernel(Fa, pT, ηp, ϕp, d.ma[i], d.mb, d.Eb[i],
                                d.cq, d.vq, d.cθ, d.vθ) for i in eachindex(d.coef))

# --- constructors: each builds the (ma, Eb, coef) arrays -------------------

"""
    TwoBodyDecay(ma, mb, mc; B = 1.0, nq = 128, nθ = 32)

Isotropic two-body decay a → b + c (Eq. 32), as a one-node [`MassAverage`].
Callable: `d(Fa, pT, ηp, ϕp)` is the daughter spectrum of b.
"""
function TwoBodyDecay(ma, mb, mc; B = 1.0, nq = 128, nθ = 32)
    ma > mb + mc || throw(ArgumentError("decay not open: ma ≤ mb + mc (equality ⇒ p_b = 0, coef = Inf)"))
    mb > 0 || throw(ArgumentError("massless daughter b not supported"))
    Eb = (ma^2 + mb^2 - mc^2) / (2ma)                          # Eq. (5)
    pb = sqrt(max(kallen(ma^2, mb^2, mc^2), 0.0)) / (2ma)      # Eq. (6)
    MassAverage([float(ma)], mb, [Eb], [B * ma / (4π * pb)], cosnodes(nq)..., cosnodes(nθ)...)
end

# Shared µ-average build (Eqs. 45–48): node energies E_b(µᵢ) and normalized
# weights coefᵢ = vᵢ ρ(µᵢ)/N, N = Σᵢ vᵢ p_b(µᵢ) ρ(µᵢ), on nµ cosine-mapped
# nodes µ ∈ [lo, ma−mb]. `density(µ)` is the cluster invariant-mass density —
# the pair density q(µ;mc,md) for three-body, ρ₃(µ) for four-body — the only
# thing that differs. `where {D}`: specialize on the density closure so its
# call inlines (construction runs ~13k times per table build).
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

# a µ-averaged decay: nµ nodes at fixed parent mass ma, Eb over the cluster µ,
# coef = B·ma/(4π)·(ρ/N). Shared by three- and four-body.
function muaverage(ma, mb, lo, density; B, nµ, nq, nθ)
    Eb, coef = muweights(ma, mb, lo, density, nµ)
    MassAverage(fill(float(ma), length(Eb)), mb, Eb, (B * ma / (4π)) .* coef,
                cosnodes(nq)..., cosnodes(nθ)...)
end

"""
    ThreeBodyDecay(ma, mb, mc, md; B = 1.0, nµ = 32, nq = 128, nθ = 32)

Isotropic three-body decay a → b + c + d (Eq. 59): the two-body kernel averaged
over the (cd) effective mass µ ∈ [mc+md, ma−mb] with weight ∝ q(µ;mc,md).
Returns a [`MassAverage`].
"""
function ThreeBodyDecay(ma, mb, mc, md; B = 1.0, nµ = 32, nq = 128, nθ = 32)
    ma > mb + mc + md || throw(ArgumentError("decay not open: ma ≤ mb + mc + md"))
    mb > 0 || throw(ArgumentError("massless daughter b not supported"))
    pair(µ) = sqrt(max(kallen(µ^2, mc^2, md^2), 0.0)) / (2µ)   # ρ₂ = p_c̃(µ), Eq. (47)
    muaverage(ma, mb, mc + md, pair; B, nµ, nq, nθ)
end

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

Isotropic four-body decay a → b + c + d + e (matrix_notes §3.1): the two-body
kernel averaged over the (cde) cluster mass µ with weight ∝ ρ₃(µ), the nested
Dalitz-area density (an inner `nσ`-point σ-integral). Returns a [`MassAverage`].
"""
function FourBodyDecay(ma, mb, mc, md, me; B = 1.0, nµ = 32, nσ = 16, nq = 128, nθ = 32)
    ma > mb + mc + md + me || throw(ArgumentError("decay not open: ma ≤ mb + mc + md + me"))
    mb > 0 || throw(ArgumentError("massless daughter b not supported"))
    cσ, vσ = cosnodes(nσ)
    ρ3(µ) = clusterdensity3(µ, mc, md, me, cσ, vσ)            # ρ₃(µ), inner σ-integral
    muaverage(ma, mb, mc + md + me, ρ3; B, nµ, nq, nθ)
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
# mmin is clamped to m ≥ 0: for very broad states (14 in the QM2016 list, e.g.
# a1(1260), Γ = 0.425) the default mR − kΓ goes negative, and a negative mmin
# *squares* inside the ψ map — the window silently inverts into the far upper
# tail (an f0(500)-like state put every node ABOVE 1.7 GeV). m = 0 is exact:
# ψof(0) = atan(−mR/Γ) is the correct lower edge of the physical support.
BreitWigner(mR, Γ; k = 4.0, mmin = mR - k * Γ, mmax = mR + k * Γ) =
    BreitWigner(mR, Γ, max(mmin, 0.0), mmax)
@inline (bw::BreitWigner)(m) =
    bw.mmin ≤ m ≤ bw.mmax ?
    m^2 * bw.Γ / ((m^2 - bw.mR^2)^2 + bw.mR^2 * bw.Γ^2) : 0.0

# Parent-mass nodes for the width average. The tan-substitution
# s = mR² + mR·Γ·tanψ flattens the Breit–Wigner Lorentzian (∫ϱ(m)·f dm ∝
# ∫√s·f dψ, smooth in ψ), so plain Gauss–Legendre in ψ resolves the central
# peak for any Γ — a cosine map would starve it (nodes cluster at the endpoints,
# not the peak). Node weight ∝ √s = m_a.
#
# The quadrature is *split at the channel threshold* `mlo`: nodes are placed only
# where the channel is open, [max(mmin, mlo), mmax], while `norm` still spans the
# full window [mmin, mmax] — the sub-threshold fraction of the spectral function
# carries no decay, and suppressing the result by exactly that fraction is what
# the normalization must do. Sampling the full window and discarding the closed
# nodes instead leaves a narrowly-reopened channel under-sampled, or with no node
# at all (a silent zero matrix): with nma = 8, a threshold at 1.3 GeV against
# support to 1.4 GeV caught zero nodes.
#
# When the threshold does cut in, the lower endpoint is a real threshold where
# coef ∝ 1/p_b has an integrable 1/√ singularity, so the open part uses the
# cosine map that absorbs it (as the qT and µ integrals do). Uncut, that endpoint
# is regular and plain Gauss is kept — bit-identical to before.
function parentmassnodes(bw::BreitWigner, nma, mlo = bw.mmin)
    x, u = gausslegendre(nma)
    ψof(m) = atan((m^2 - bw.mR^2) / (bw.mR * bw.Γ))
    sof(ψ) = bw.mR^2 + bw.mR * bw.Γ * tan(ψ)
    ψlo, ψhi = ψof(bw.mmin), ψof(bw.mmax)
    hψ, mψ = (ψhi - ψlo) / 2, (ψhi + ψlo) / 2

    norm = 0.0                                   # full window: kernel-free, regular
    for i in 1:nma
        s = sof(mψ + hψ * x[i])
        s > 0 && (norm += u[i] * sqrt(s))
    end

    mas, ws = Float64[], Float64[]
    mo = max(bw.mmin, mlo)
    mo < bw.mmax || return mas, ws, norm         # closed across the whole window
    # Integrate the open part [ψ(mo), ψhi]. Only the rule differs: a threshold
    # that cuts in puts coef ∝ 1/p_b ~ 1/√ at the lower endpoint, which the
    # cosine map absorbs (nodes mid − h·c); an uncut endpoint is regular, so
    # plain Gauss is kept (mid + h·x) and the result is bit-identical to the
    # pre-split code. `scale` restates both on the shared ψ measure.
    cut = mo > bw.mmin
    z, w, sgn = cut ? (cosnodes(nma)..., -1.0) : (x, u, 1.0)
    h, mid = cut ? ((ψhi - ψof(mo)) / 2, (ψhi + ψof(mo)) / 2) : (hψ, mψ)
    scale = h / hψ
    for i in 1:nma
        s = sof(mid + sgn * h * z[i])
        s > 0 || continue
        push!(mas, sqrt(s)); push!(ws, scale * w[i] * sqrt(s))
    end
    return mas, ws, norm
end

"""
    BroadParent(daughter masses..., bw::BreitWigner; B = 1.0, nma = 16, kw...)

A decay whose parent has finite width: the underlying sharp decay (two-, three-,
or four-body, chosen by the number of daughter masses) averaged over the parent
spectral function `bw` (pole `bw.mR`, width `bw.Γ`). Returns a [`MassAverage`] —
the parent-mass nodes of every sub-decay are *flattened* into one node list, so
a broad-parent decay is just another `MassAverage`, handled exactly like a sharp
one. Nodes below the daughter threshold are dropped and enter only the
normalization, so a channel closed at the pole keeps its open-tail fraction
(channel reopening). `kw...` pass through to the underlying decay (nµ, nσ, nq, nθ).
"""
function BroadParent(build, bw::BreitWigner, opensum; nma = 16)
    # nodes come back already restricted to where the channel is open; `norm`
    # still spans the full spectral window, so a channel closed at the pole keeps
    # exactly its open-tail fraction (channel reopening).
    mas, wall, norm = parentmassnodes(bw, nma, opensum)
    ma_all, Eb_all, coef_all = Float64[], Float64[], Float64[]
    local sub                                     # a sample sub-decay (shared cq…vθ, mb)
    got = false
    for i in eachindex(mas)
        mas[i] > opensum || continue              # roundoff pinned a node to the threshold
        s = build(mas[i])::MassAverage
        all(isfinite, s.coef) || continue         # near-threshold: p_b rounds to 0
        append!(ma_all, s.ma)
        append!(Eb_all, s.Eb)
        append!(coef_all, (wall[i] / norm) .* s.coef)   # fold ϱ weight into coef
        sub, got = s, true
    end
    got || (sub = build(opensum + 0.05))          # degenerate: fully closed ⇒ zero matrix
    MassAverage(ma_all, sub.mb, Eb_all, coef_all, sub.cq, sub.vq, sub.cθ, sub.vθ)
end

BroadParent(mb, mc, bw::BreitWigner; B = 1.0, nma = 16, kw...) =
    BroadParent(ma -> TwoBodyDecay(ma, mb, mc; B, kw...), bw, mb + mc; nma)
BroadParent(mb, mc, md, bw::BreitWigner; B = 1.0, nma = 16, kw...) =
    BroadParent(ma -> ThreeBodyDecay(ma, mb, mc, md; B, kw...), bw, mb + mc + md; nma)
BroadParent(mb, mc, md, me, bw::BreitWigner; B = 1.0, nma = 16, kw...) =
    BroadParent(ma -> FourBodyDecay(ma, mb, mc, md, me; B, kw...), bw, mb + mc + md + me; nma)

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

# Fill rows `ks` of every mode matrix — the node sum for each daughter pT.
#
# A row accumulates in the contiguous Float64 buffers `buf` (one per mode)
# rather than straight into a strided matrix row: a P-point stencil then hits
# consecutive memory (same cache line) instead of P rows npT apart. The finished
# Float64 row is rounded once into `Ws`, whose eltype may be Float32 for storage
# — folding the buildkeys! downcast in here, so no throwaway Float64 matrix.
#
# `buf` is allocated here, so every caller — the serial sweep or one spawned
# chunk — owns its own set for the call's whole lifetime.
function fillrows!(Ws, ks, d::MassAverage, mt, pTv, zc, zc4, qgrid, P)
    buf = map(_ -> zeros(length(qgrid)), mt)
    for k in ks
        for r in buf; fill!(r, 0.0); end
        for i in eachindex(d.coef)
            scatterrows!(buf, mt, d.ma[i], d.mb, d.Eb[i], pTv[k],
                         d.cq, d.vq, zc, zc4, qgrid, P, d.coef[i])
        end
        for (W, r) in zip(Ws, buf)
            @inbounds for j in eachindex(r)
                W[k, j] = eltype(W)(r[j])      # exact when Float64; the one Float32 round
            end
        end
    end
end

# Core (mt a Tuple of modes) → one Float64 matrix per mode. The node sum covers
# every case at once: one node (two-body), the µ-average (three-/four-body), or
# the flattened parent-mass × cluster nodes (broad parent) — each carrying its
# own (ma_i, Eb_i, coef_i). `threaded` parallelizes over daughter pT rows; set
# it false when already inside a threaded loop (e.g. the cascade key build), so
# the inner and outer `@threads` don't oversubscribe the pool.
function decaymatrices(d::MassAverage, mt::Tuple, pTs, qgrid;
                       P = Val(4), threaded = true, T::Type = Float64)
    mt = map(Int, mt)                              # the assembly is Int-typed (chebT,
                                                   # scatterrows!); accept any Integer mode
    # used read-only (pTv[k], eachindex) and never copied, so any AbstractVector
    # serves — a Vector is aliased, a range is indexed in place. Same convention
    # `qgrid` already follows; callers must not mutate either during the build.
    pTv = pTs
    nq = length(qgrid)
    Ws = map(_ -> zeros(T, length(pTv), nq), mt)   # output eltype T (Float32 to store)
    nθ = length(d.cθ)
    zc, zc4 = chebnodes(nθ), chebnodes(4nθ)   # compute once; threaded down to the scatter
    if threaded
        # One task per chunk of rows, each owning its buffers for the task's whole
        # lifetime — so nothing is shared and task migration cannot alias them
        # (unlike a threadid()-indexed pool or OncePerThread). Buffers drop from
        # one set per row to one per task. Rows are independent, so this is
        # bitwise identical to the serial sweep however the chunks land.
        nt = min(Threads.nthreads(), length(pTv))
        @sync for ks in Iterators.partition(eachindex(pTv), cld(length(pTv), nt))
            Threads.@spawn fillrows!(Ws, ks, d, mt, pTv, zc, zc4, qgrid, P)
        end
    else
        fillrows!(Ws, eachindex(pTv), d, mt, pTv, zc, zc4, qgrid, P)
    end
    return collect(Ws)
end

# Entry from a modes vector, with a fallback for many modes (tuple ops stop
# unrolling past ~32, so cap the fused path and loop single-mode beyond that).
# `Tuple(modes)` is a runtime tuple, so the dispatch to the core is dynamic (once
# per call, cold); the ::return annotation keeps the boundary type-stable for the
# caller (e.g. the Float32 conversion in buildkeys!) instead of leaking `Any`.
function decaymatrices(d::MassAverage, modes::AbstractVector{<:Integer}, pTs, qgrid;
                       P = Val(4), threaded = true, T::Type = Float64)::Vector{Matrix{T}}
    length(modes) ≤ 16 && return decaymatrices(d, Tuple(modes), pTs, qgrid; P, threaded, T)
    return [Matrix{T}(DecayMatrix(d, n, pTs, qgrid; P).W) for n in modes]
end

# Single-mode public API: one DecayMatrix struct, via the fused core with (n,).
DecayMatrix(d::MassAverage, n::Integer, pTs, qgrid; P = Val(4)) =
    DecayMatrix(n, collect(Float64, pTs), collect(Float64, qgrid),
                decaymatrices(d, (n,), pTs, qgrid; P)[1])

# Chebyshev–Gauss nodes Z_k = cos((2k−1)π/(2n)). Pure like cosnodes — computed
# once per decaymatrices call (~100 ns) and passed down to the row scatter.
chebnodes(n::Int) = [cos((2k - 1) * π / (2n)) for k in 1:n]

# Tₙ(Z) = cos(n·acos Z) by the Chebyshev recurrence (Eq. 40): the assembly
# needs only this polynomial, never the angle itself, so no acos/cos here.
# Even in n (T₋ₙ = Tₙ, since cos is even), so a negative Fourier mode reuses the
# |n| polynomial — without the abs, n < 0 skips the recurrence and returns Z.
@inline function chebT(n::Int, Z::Float64)
    n = abs(n)
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
function scatterrows!(rows, modes::NTuple{K,Int}, ma, mb, Eb, pT, cq, vq, zc, zc4,
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
                z = Zlo > -1.5 ? zc4 : zc   # densify near the transition
                for Z in z
                    c = (C + Q * Z) / M
                    s = s .+ chebvals(modes, Z) ./ sqrt((c - 1) * (c + 1))
                end
                s = s .* (π / length(z))
            else                                   # partial window: shifted Chebyshev–Gauss
                # analytic cancellation (matrix_notes §8.2): endpoint weight removed,
                # regular R² = (Q/M)(1+Z)(c+1) > 0, no guard; densify near Zlo→−1.
                zp = Zlo < -0.5 ? zc4 : zc
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
