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

export kallen, TwoBodyDecay, ThreeBodyDecay, DecayMatrix

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
struct ThreeBodyDecay
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
    cµ, vµ = cosnodes(nµ)        # cosine map absorbs the √ endpoint zeros
    lo, hi = mc + md, ma - mb
    h, m = (hi - lo) / 2, (hi + lo) / 2
    Eb = Vector{Float64}(undef, nµ)
    coef = Vector{Float64}(undef, nµ)
    norm = 0.0
    for i in 1:nµ
        µ = m - h * cµ[i]
        Eb[i] = (ma^2 + mb^2 - µ^2) / (2ma)                    # Eq. (45)
        pb = sqrt(max(kallen(ma^2, mb^2, µ^2), 0.0)) / (2ma)   # Eq. (46)
        pc = sqrt(max(kallen(µ^2, mc^2, md^2), 0.0)) / (2µ)    # Eq. (47)
        coef[i] = vµ[i] * pc
        norm += vµ[i] * pb * pc                                # → N, Eq. (48)
    end
    coef ./= norm
    ThreeBodyDecay(ma, mb, mc, md, B, Eb, coef, cosnodes(nq)..., cosnodes(nθ)...)
end

(d::ThreeBodyDecay)(Fa::F, pT, ηp, ϕp) where {F} =
    d.B * d.ma / (4π) *                                  # Eq. (59)
    sum(d.coef[i] * decaykernel(Fa, pT, ηp, ϕp, d.ma, d.mb, d.Eb[i],
                                d.cq, d.vq, d.cθ, d.vθ) for i in eachindex(d.coef))

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

# Rows are independent (each thread writes only its own row view), so the
# assembly threads over daughter pT; result is bitwise identical to serial.
function DecayMatrix(d::TwoBodyDecay, n::Integer, pTs, qgrid; P = Val(4))
    pTv = collect(Float64, pTs)
    W = zeros(length(pTv), length(qgrid))
    Threads.@threads for k in eachindex(pTv)
        scatterrow!(view(W, k, :), n, d.ma, d.mb, d.Eb, pTv[k],
                    d.cq, d.vq, d.cθ, d.vθ, qgrid, P, d.B * d.ma / (4π * d.pb))
    end
    DecayMatrix(n, pTv, collect(Float64, qgrid), W)
end

function DecayMatrix(d::ThreeBodyDecay, n::Integer, pTs, qgrid; P = Val(4))
    pTv = collect(Float64, pTs)
    W = zeros(length(pTv), length(qgrid))
    Threads.@threads for k in eachindex(pTv)
        for i in eachindex(d.coef)                          # µ-average, Eq. (59)
            scatterrow!(view(W, k, :), n, d.ma, d.mb, d.Eb[i], pTv[k],
                        d.cq, d.vq, d.cθ, d.vθ, qgrid, P, d.B * d.ma / (4π) * d.coef[i])
        end
    end
    DecayMatrix(n, pTv, collect(Float64, qgrid), W)
end

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

# Mirrors decaykernel's qT structure, but instead of evaluating Fa it
# scatters the quadrature factor onto the parent-grid stencil. The inner
# integral is done in Z = D/Q rather than Δη — an η-independent source never
# needs the rapidity itself, and in Z everything is algebraic:
#     2 ∫dΔη Θ/√(Q²−D²) · 2Tₙ(Z)  =  (4/M) ∫_{Zlo}^{1} Tₙ(Z) dZ / √((1−Z²)(c²−1)),
# with c = cosh Δη = (C + QZ)/M and Zlo = (M−C)/Q (where c = 1). No acosh,
# cosh, or acos appears. For Zlo ≤ −1 the 1/√(1−Z²) factor is exactly the
# Chebyshev–Gauss weight (fixed nodes, weight π/n); otherwise the cosine map
# absorbs the 1/√ endpoints at Zlo (from c²−1) and at 1 (from 1−Z²).
function scatterrow!(row, n, ma, mb, Eb, pT, cq, vq, cθ, vθ, qgrid, P, pref)
    pb = sqrt(max(Eb^2 - mb^2, 0.0))
    C = ma * Eb
    mbT = hypot(mb, pT)
    qlo = max(ma * (Eb * pT - pb * mbT) / mb^2, 0.0)
    qhi = ma * (Eb * pT + pb * mbT) / mb^2
    qstar = ma * (pb * mbT - Eb * pT) / mb^2
    segments = qlo < qstar < qhi ? ((qlo, qstar), (qstar, qhi)) :
                                   ((qlo, qhi), (qhi, qhi))
    qmin, qmax = first(qgrid), last(qgrid)
    for (a, b) in segments
        a < b || continue
        hq, mq = (b - a) / 2, (a + b) / 2
        zc = chebnodes(length(cθ))
        @inbounds for i in eachindex(cq)
            qT = mq - hq * cq[i]
            qmin ≤ qT ≤ qmax || continue           # F_a ≡ 0 off the grid
            M = hypot(ma, qT) * mbT
            Q = qT * pT
            M < C + Q || continue                  # Θ support (Z window nonempty)
            Zlo = (M - C) / Q
            s = 0.0
            if Zlo ≤ -1                            # full window: Chebyshev–Gauss
                # for Zlo just below −1 the factor 1/√(c²−1) is steep at the
                # Z = −1 edge (scale |Zlo+1|); densify the rule there
                z = Zlo > -1.5 ? chebnodes(4 * length(cθ)) : zc
                for Z in z
                    c = (C + Q * Z) / M
                    s += chebT(n, Z) / sqrt((c - 1) * (c + 1))
                end
                s *= π / length(z)
            else                                   # partial window [Zlo, 1]
                hz, mz = (1 - Zlo) / 2, (1 + Zlo) / 2
                for j in eachindex(cθ)
                    Z = mz - hz * cθ[j]
                    c = (C + Q * Z) / M
                    u = (1 - Z) * (1 + Z) * (c - 1) * (c + 1)
                    u > 0 || continue
                    s += vθ[j] * chebT(n, Z) / sqrt(u)
                end
                s *= hz
            end
            fac = pref * vq[i] * hq * qT * (4 / M) * s
            j0, w = _stencil(qT, qgrid, P)
            for (m, wm) in pairs(w)
                row[j0 + m - 1] += fac * wm
            end
        end
    end
    return row
end

end # module
