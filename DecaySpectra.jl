"""
    DecaySpectra

Isotropic decay feed-down — two-, three- and four-body, with optional finite
parent width — derived in `matrix_notes.tex`; equation numbers below refer to
`Maximum_Entropy_notes.pdf`, the original derivation. Functor style: construct a
decay once, then call it like a function on a parent spectrum:

    ρ = TwoBodyDecay(0.7752, 0.1396, 0.1396)           # a → b + c
    ω = ThreeBodyDecay(0.7827, 0.1396, 0.1396, 0.1350) # a → b + c + d
    ρw = BroadParent(0.1396, 0.1396, BreitWigner(0.7752, 0.149))   # broad a
    Fb = ρ(Fa, pT, ηp, ϕp)                             # Eq. (32)
    Fb = ω(Fa, pT, ηp, ϕp)                             # Eq. (59)

`Fa(qT, ηq, ϕq)` is the invariant parent spectrum E dN_a/d³q (Eq. 14) and the
result is the invariant daughter spectrum F_b(pT, ηp, ϕp) of particle b.

Every decay is represented by [`MassAverage`], a weighted collection of parent
masses and daughter rest-frame energies evaluated through the shared two-body
kernel. The constructors differ only in how they build those nodes.

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
       DecayMatrix, decaymatrices, decaymatrices!, rowscratch, SpectralShape,
       BreitWigner, MassDependentBW, BroadParent, Rules

"Källén triangle function λ(x, y, z), Eq. (7)."
kallen(x, y, z) = x^2 + y^2 + z^2 - 2x * y - 2x * z - 2y * z

# ---------------------------------------------------------------------------
# Quadrature rules
# ---------------------------------------------------------------------------
# Quadrature rules are built once and treated as read-only by channel builds.

"""
A quadrature rule on [−1, 1]: nodes `x`, ascending, applied as `mid + half*x[i]`.
The weight parameter is the rule's *shape*:

  * `QuadRule{Vector{Float64}}` — per-node weights (Gauss–Legendre, and the
    cosine map built on it);
  * `QuadRule{Float64}` — one weight for every node (Chebyshev–Gauss, where
    removing the inverse-√ weight analytically leaves π/n at each node). Kept
    scalar rather than as n copies: the sum factors it out, so one multiply at
    the end replaces n inside the innermost loop — and factoring it out is also
    what keeps the result bit-for-bit what the unfactored form gave.
"""
struct QuadRule{W}
    x::Vector{Float64}
    w::W
end
QuadRule(nodes_and_weights::Tuple{Vector{Float64},Vector{Float64}}) =
    QuadRule(nodes_and_weights...)
Base.length(q::QuadRule) = length(q.x)
Base.eachindex(q::QuadRule) = eachindex(q.x)

# Chebyshev–Gauss rule of the first kind, order n: Zₖ = cos((2k−1)π/2n), w = π/n.
chebnodes(order::Int) =
    [cos((2 * index - 1) * π / (2 * order)) for index in 1:order]
chebrule(order::Int) = QuadRule(chebnodes(order), π / order)

"""
The two rules the parent-mass average needs, both at order `nma`: plain
Gauss–Legendre for the normalization sweep and the regular lower endpoint, and
the cosine map for when a channel threshold cuts in and puts coef ∝ 1/p_b ~ 1/√
there. Both ascend −1 → 1, so `parentmassnodes` swaps one for the other and
changes nothing else.
"""
struct MassRule
    gl::QuadRule{Vector{Float64}}
    cos::QuadRule{Vector{Float64}}
end
MassRule(order::Int) =
    MassRule(QuadRule(gausslegendre(order)), QuadRule(cosnodes(order)))

"""
The rules applied at *evaluation* — the subset a [`MassAverage`] carries. The
qT rule serves both the functor and the matrix path; the Δη rule only the
functor (`decaykernel`); the two Chebyshev rules only the matrix path
(`scatterrows!`, which picks the dense one near a window transition).
"""
struct KernelRules
    q::QuadRule{Vector{Float64}}   # cosnodes(nq)
    θ::QuadRule{Vector{Float64}}   # cosnodes(nθ)
    zc::QuadRule{Float64}          # Chebyshev, nθ  nodes
    zc4::QuadRule{Float64}         # Chebyshev, 4nθ nodes
end
KernelRules(nq::Int, nθ::Int) =
    KernelRules(QuadRule(cosnodes(nq)), QuadRule(cosnodes(nθ)),
                chebrule(nθ), chebrule(4nθ))

"""
    Rules(; nq = 128, nθ = 32, nµ = 32, nσ = 16, nma = 16)

Every quadrature rule implied by a set of orders, built once. `kern` is applied
at evaluation and is stored in each [`MassAverage`]; `µ`, `σ` and `ma` are
consumed during construction — folded into the node energies and weights — and
are not kept afterwards.

Build one per table and share it across channels.
"""
struct Rules
    kern::KernelRules
    µ::QuadRule{Vector{Float64}} # cosnodes(nµ)   — cluster-mass average
    σ::QuadRule{Vector{Float64}} # cosnodes(nσ)   — inner sub-mass integral (4-body)
    ma::MassRule                 # order nma      — parent spectral average
end
_checkorder(order_name, order) =
    order ≥ 1 ||
    throw(ArgumentError("quadrature order $order_name must be ≥ 1, got $order"))

function Rules(; nq = 128, nθ = 32, nµ = 32, nσ = 16, nma = 16)
    _checkorder(:nq, nq)
    _checkorder(:nθ, nθ)
    _checkorder(:nµ, nµ)
    _checkorder(:nσ, nσ)
    _checkorder(:nma, nma)
    return Rules(KernelRules(nq, nθ), QuadRule(cosnodes(nµ)),
                 QuadRule(cosnodes(nσ)), MassRule(nma))
end

# Cosine-mapped Gauss–Legendre rule for ∫ₐᵇ f dx with x = mid + half·xᵢ:
# returns (−cosθᵢ, (π/2)·wᵢ·sinθᵢ) so that ∫ₐᵇ f ≈ half·Σᵢ vᵢ f(mid + half·xᵢ).
# The sinθ Jacobian (inside vᵢ) absorbs 1/√ endpoint singularities of f.
#
# The leading minus is what makes the nodes ascend −1 → 1, matching
# Gauss–Legendre convention, so all rules use `mid + half*xᵢ` interchangeably.
function cosnodes(order::Int)
    legendre_nodes, legendre_weights = gausslegendre(order)
    angles = @. π * (legendre_nodes + 1) / 2
    return (-cos.(angles), @. π / 2 * legendre_weights * sin(angles))
end

# ---------------------------------------------------------------------------
# Common decay kernel
# ---------------------------------------------------------------------------

# K(Eb) = ∫ qT dqT ∫ dηq Θ(Q²−D²)/√(Q²−D²) [Fa(ϕp+α) + Fa(ϕp−α)]
# for a daughter b emitted with rest-frame energy Eb (momentum p_b = √(Eb²−mb²))
# from parent a; M = maT·mbT, Q = qT·pT, C = ma·Eb, D = M cosh(Δη) − C
# (Eq. 29), Z = D/Q (Eq. 30), α = arccos Z (Eq. 33). This is the double
# integral common to Eqs. (32), (41), (59), (61).
# The qT integrand has an integrable ln|qT − qT*| singularity where the Δη
# domain splits, so the interval is divided there.
function decaykernel(Fa::F, pT, ηp, ϕp, ma, mb, Eb,
                     momentum_rule::QuadRule, rapidity_rule::QuadRule) where {F}
    pb = sqrt(max(Eb^2 - mb^2, 0.0))
    C = ma * Eb
    mbT = hypot(mb, pT)

    # qT support of the Θ function (maT·mbT ≤ C + qT·pT): [qT−, qT+] with
    # qT± = ma (Eb pT ± pb mbT)/mb²; the log singularity sits at qT* = −qT−.
    qT_min = max(ma * (Eb * pT - pb * mbT) / mb^2, 0.0)
    qT_max = ma * (Eb * pT + pb * mbT) / mb^2
    qT_split = ma * (pb * mbT - Eb * pT) / mb^2
    segments = qT_min < qT_split < qT_max ?
               ((qT_min, qT_split), (qT_split, qT_max)) :
               ((qT_min, qT_max), (qT_max, qT_max))

    integral = 0.0
    for (qT_start, qT_stop) in segments
        qT_start < qT_stop || continue
        qT_half_width = (qT_stop - qT_start) / 2
        qT_midpoint = (qT_start + qT_stop) / 2
        node_count = length(momentum_rule)
        task_count = min(Threads.nthreads(), node_count)
        if task_count == 1
            integral += qnodes(Fa, pT, ηp, ϕp, ma, C, mbT,
                               qT_half_width, qT_midpoint,
                               momentum_rule, rapidity_rule, 1:node_count)
        else
            tasks = [Threads.@spawn qnodes(Fa, pT, ηp, ϕp, ma, C, mbT,
                                           qT_half_width, qT_midpoint,
                                           momentum_rule, rapidity_rule,
                                           task_index:task_count:node_count)
                     for task_index in 1:task_count]
            for task in tasks
                integral += fetch(task)::Float64
            end
        end
    end
    return integral
end

# Sum selected qT nodes. Range specialization keeps the serial unit-stride path fast.
@inline function qnodes(Fa::F, pT, ηp, ϕp, ma, C, mbT,
                        qT_half_width, qT_midpoint,
                        momentum_rule::QuadRule, rapidity_rule::QuadRule,
                        node_indices::AbstractRange{Int}) where {F}
    q_nodes, q_weights = momentum_rule.x, momentum_rule.w
    rapidity_nodes, rapidity_weights = rapidity_rule.x, rapidity_rule.w
    integral = 0.0
    @inbounds for q_index in node_indices
        qT = qT_midpoint + qT_half_width * q_nodes[q_index]
        M = hypot(ma, qT) * mbT
        Q = qT * pT
        # Δη support: cosh(Δη) ∈ [max(1, (C−Q)/M), (C+Q)/M] (Eq. 21)
        cosh_eta_max = (C + Q) / M
        cosh_eta_max > 1 || continue
        eta_max = acosh(cosh_eta_max)
        cosh_eta_min = (C - Q) / M
        eta_min = cosh_eta_min > 1 ? acosh(cosh_eta_min) : 0.0
        eta_half_width = (eta_max - eta_min) / 2
        eta_midpoint = (eta_min + eta_max) / 2
        rapidity_sum = 0.0
        @inbounds for rapidity_index in eachindex(rapidity_nodes)
            Δη = eta_midpoint + eta_half_width * rapidity_nodes[rapidity_index]
            D = M * cosh(Δη) - C                     # Eq. (29)
            S2 = (Q - D) * (Q + D)                   # = Q²(1−Z²), Eq. (31)
            S2 > 0 || continue
            α = acos(clamp(D / Q, -1, 1))            # Eqs. (30), (33)
            # both Δη signs (Fa need not be symmetric in rapidity)
            rapidity_sum += rapidity_weights[rapidity_index] / sqrt(S2) *
                            (Fa(qT, ηp + Δη, ϕp + α) +
                             Fa(qT, ηp + Δη, ϕp - α) +
                             Fa(qT, ηp - Δη, ϕp + α) +
                             Fa(qT, ηp - Δη, ϕp - α))
        end
        integral += q_weights[q_index] * qT_half_width * qT *
                    eta_half_width * rapidity_sum
    end
    return integral
end

# ---------------------------------------------------------------------------
# Weighted decay representation
# ---------------------------------------------------------------------------

"""
An isotropic decay represented as
`F_b = Σᵢ coefᵢ K(F_a; maᵢ, mb, Ebᵢ)`.

The constructors below encode two-, three-, four-body, and broad-parent decays
only by changing the `(ma, Eb, coef)` nodes. Evaluation and matrix assembly use
this shared representation.
"""
struct MassAverage
    ma::Vector{Float64}          # per-node parent mass
    mb::Float64                  # observed daughter (fixed)
    Eb::Vector{Float64}          # per-node E_b(node)
    coef::Vector{Float64}        # per-node weight: B·ma·[1/p_b or ρ/N]/(4π)
    kern::KernelRules            # shared evaluation rules, treated as read-only
end

# An empty node list is a channel closed across its full parent-mass window.
(decay::MassAverage)(parent_spectrum::F, pT, ηp, ϕp) where {F} =
    sum((decay.coef[index] *
         decaykernel(parent_spectrum, pT, ηp, ϕp, decay.ma[index], decay.mb,
                     decay.Eb[index], decay.kern.q, decay.kern.θ)
         for index in eachindex(decay.coef));
        init = 0.0)

# --- constructors: each builds the (ma, Eb, coef) arrays -------------------

"""
    TwoBodyDecay(ma, mb, mc; B = 1.0, nq = 128, nθ = 32)
    TwoBodyDecay(ma, mb, mc, rules::Rules; B = 1.0)

Isotropic two-body decay a → b + c (Eq. 32), as a one-node [`MassAverage`].
Callable: `d(Fa, pT, ηp, ϕp)` is the daughter spectrum of b.

The second form takes prebuilt [`Rules`] instead of orders — use it when
constructing many decays, so they share one set of quadrature rules.
"""
function TwoBodyDecay(ma, mb, mc, rules::Rules; B = 1.0)
    ma > mb + mc || throw(ArgumentError("decay not open: ma ≤ mb + mc (equality ⇒ p_b = 0, coef = Inf)"))
    mb > 0 || throw(ArgumentError("massless daughter b not supported"))
    Eb = (ma^2 + mb^2 - mc^2) / (2ma)                          # Eq. (5)
    pb = sqrt(max(kallen(ma^2, mb^2, mc^2), 0.0)) / (2ma)      # Eq. (6)
    MassAverage([float(ma)], mb, [Eb], [B * ma / (4π * pb)], rules.kern)
end

TwoBodyDecay(ma, mb, mc; B = 1.0, nq = 128, nθ = 32) =
    TwoBodyDecay(ma, mb, mc, Rules(; nq, nθ); B)

# Normalized µ-average (Eqs. 45–48). The typed closure keeps `density` inlined.
function muweights(ma, mb, minimum_cluster_mass, density::D,
                   cluster_mass_rule::QuadRule) where {D}
    mu_nodes, mu_weights = cluster_mass_rule.x, cluster_mass_rule.w
    node_count = length(cluster_mass_rule)
    maximum_cluster_mass = ma - mb
    half_width = (maximum_cluster_mass - minimum_cluster_mass) / 2
    midpoint = (maximum_cluster_mass + minimum_cluster_mass) / 2
    Eb = Vector{Float64}(undef, node_count)
    coef = Vector{Float64}(undef, node_count)
    normalization = 0.0
    for node_index in 1:node_count
        µ = midpoint + half_width * mu_nodes[node_index]
        Eb[node_index] = (ma^2 + mb^2 - µ^2) / (2ma)           # Eq. (45)
        pb = sqrt(max(kallen(ma^2, mb^2, µ^2), 0.0)) / (2ma)   # Eq. (46)
        density_value = density(µ)                              # Eq. (47) / ρ₃
        coef[node_index] = mu_weights[node_index] * density_value
        normalization += mu_weights[node_index] * pb * density_value
    end
    coef ./= normalization
    return Eb, coef
end

function muaverage(ma, mb, minimum_cluster_mass, density, rules::Rules; B)
    Eb, coef = muweights(ma, mb, minimum_cluster_mass, density, rules.µ)
    return MassAverage(fill(float(ma), length(Eb)), mb, Eb,
                       (B * ma / (4π)) .* coef, rules.kern)
end

"""
    ThreeBodyDecay(ma, mb, mc, md; B = 1.0, nµ = 32, nq = 128, nθ = 32)
    ThreeBodyDecay(ma, mb, mc, md, rules::Rules; B = 1.0)

Isotropic three-body decay a → b + c + d (Eq. 59): the two-body kernel averaged
over the (cd) effective mass µ ∈ [mc+md, ma−mb] with weight ∝ q(µ;mc,md).
Returns a [`MassAverage`].
"""
function ThreeBodyDecay(ma, mb, mc, md, rules::Rules; B = 1.0)
    ma > mb + mc + md || throw(ArgumentError("decay not open: ma ≤ mb + mc + md"))
    mb > 0 || throw(ArgumentError("massless daughter b not supported"))
    pair_density(µ) =
        sqrt(max(kallen(µ^2, mc^2, md^2), 0.0)) / (2µ)         # Eq. (47)
    muaverage(ma, mb, mc + md, pair_density, rules; B)
end

ThreeBodyDecay(ma, mb, mc, md; B = 1.0, nµ = 32, nq = 128, nθ = 32) =
    ThreeBodyDecay(ma, mb, mc, md, Rules(; nq, nθ, nµ); B)

# Cluster invariant-mass density ρ₃(µ) = ∫_{md+me}^{µ−mc} dσ q(µ;mc,σ) q(σ;md,me),
# the three-particle (cde) phase-space volume (Dalitz area) at cluster mass µ.
# One cosine-mapped Gauss sweep over the sub-mass σ; the map absorbs the √
# endpoint zeros of q at σ = md+me and σ = µ−mc. Kernel-free, hence cheap.
@inline function clusterdensity3(µ, mc, md, me, sigma_rule::QuadRule)
    sigma_nodes, sigma_weights = sigma_rule.x, sigma_rule.w
    minimum_submass, maximum_submass = md + me, µ - mc
    maximum_submass > minimum_submass || return 0.0
    half_width = (maximum_submass - minimum_submass) / 2
    midpoint = (maximum_submass + minimum_submass) / 2
    integral = 0.0
    @inbounds for node_index in eachindex(sigma_nodes)
        σ = midpoint + half_width * sigma_nodes[node_index]
        qµ = sqrt(max(kallen(µ^2, mc^2, σ^2), 0.0)) / (2µ)   # q(µ; mc, σ)
        qσ = sqrt(max(kallen(σ^2, md^2, me^2), 0.0)) / (2σ)  # q(σ; md, me)
        integral += sigma_weights[node_index] * qµ * qσ
    end
    return half_width * integral
end

"""
    FourBodyDecay(ma, mb, mc, md, me; B = 1.0, nµ = 32, nσ = 16, nq = 128, nθ = 32)
    FourBodyDecay(ma, mb, mc, md, me, rules::Rules; B = 1.0)

Isotropic four-body decay a → b + c + d + e (matrix_notes §3.1): the two-body
kernel averaged over the (cde) cluster mass µ with weight ∝ ρ₃(µ), the nested
Dalitz-area density (an inner `nσ`-point σ-integral). Returns a [`MassAverage`].
"""
function FourBodyDecay(ma, mb, mc, md, me, rules::Rules; B = 1.0)
    ma > mb + mc + md + me || throw(ArgumentError("decay not open: ma ≤ mb + mc + md + me"))
    mb > 0 || throw(ArgumentError("massless daughter b not supported"))
    cluster_density(µ) = clusterdensity3(µ, mc, md, me, rules.σ)
    muaverage(ma, mb, mc + md + me, cluster_density, rules; B)
end

FourBodyDecay(ma, mb, mc, md, me; B = 1.0, nµ = 32, nσ = 16, nq = 128, nθ = 32) =
    FourBodyDecay(ma, mb, mc, md, me, Rules(; nq, nθ, nµ, nσ); B)

# ---------------------------------------------------------------------------
# Finite parent width: average the decay over the parent spectral function
# ---------------------------------------------------------------------------

"""
    SpectralShape

A parent-mass distribution: an unnormalized weight `ϱ(m)`, zero off a finite
support. [`BroadParent`] averages a decay over any of these and divides by the
integrated weight, so only the shape matters. The contract is three methods:

    support(ϱ) -> (mmin, mmax)   the finite interval outside which ϱ is zero
    polemap(ϱ) -> (mR, Γ0)       builds the tangent map that flattens the peak
                                 (the map only, never the weight itself)
    (ϱ)(m)    -> Float64         the unnormalized weight at mass m

plus, optionally, `_lowerendpointvanishes(ϱ)` — declare `true` when ϱ vanishes
like a square root at its own lower support edge (a threshold), so
[`parentmassnodes`](@ref) picks the cosine-mapped rule there.
"""
abstract type SpectralShape end

"The finite `(mmin, mmax)` support outside which the shape is zero."
support(shape::SpectralShape) = (shape.mmin, shape.mmax)

"`(mR, Γ0)` for the tangent substitution; the map only, not the weight."
function polemap end

# `true` when ϱ ~ √(m - mmin) at its own lower edge (a decay threshold): the
# cosine rule's sin-Jacobian absorbs that vanishing, plain Gauss does not.
#
# Worth its keep for the *second* consumer, not the first. Here the ψ
# substitution already flattens most of the endpoint, so the cosine rule gains
# only 1-2x. `ThermalModel._spectralrule` integrates ϱ(m)ℓⱼ(m) raw over the
# window with no substitution, and there the same trait is the difference
# between 1.4e-7 and 1.1e-14 on the σ — seven orders. Delete it and the plan's
# quadrature silently loses that accuracy while every test still passes at the
# model's 1e-6 tolerance.
_lowerendpointvanishes(::SpectralShape) = false

"""
    BreitWigner(mR, Γ; k = 4.0, mmin = mR - k*Γ, mmax = mR + k*Γ)

Relativistic (constant-width) Breit–Wigner spectral function of a resonance,
`ϱ(m) ∝ m²Γ / ((m²−mR²)² + mR²Γ²)`, truncated to `[mmin, mmax]` and zero
outside. Unnormalized: [`BroadParent`] divides by the integrated weight over
the truncated support, so only the shape matters.

Requires `mR > 0`, `Γ > 0` and `0 ≤ mmin < mmax`, all finite.
"""
struct BreitWigner <: SpectralShape
    mR::Float64
    Γ::Float64
    mmin::Float64
    mmax::Float64

    # Invalid domains otherwise produce a silently empty parent-mass average.
    function BreitWigner(mR, Γ, mmin, mmax)
        all(isfinite, (mR, Γ, mmin, mmax)) ||
            throw(ArgumentError("BreitWigner: non-finite (mR, Γ, mmin, mmax) = $((mR, Γ, mmin, mmax))"))
        mR > 0 || throw(ArgumentError("BreitWigner: pole mass must be positive, got mR = $mR"))
        Γ > 0 || throw(ArgumentError("BreitWigner: width must be positive, got Γ = $Γ; " *
                                     "a state too narrow to average should stay sharp"))
        0 ≤ mmin < mmax ||
            throw(ArgumentError("BreitWigner: support must satisfy 0 ≤ mmin < mmax, got [$mmin, $mmax]"))
        new(mR, Γ, mmin, mmax)
    end
end
# Clamp the default support to the physical domain m ≥ 0 before the ψ map.
BreitWigner(mR, Γ; k = 4.0, mmin = mR - k * Γ, mmax = mR + k * Γ) =
    BreitWigner(mR, Γ, max(mmin, 0.0), mmax)
@inline (bw::BreitWigner)(m) =
    bw.mmin ≤ m ≤ bw.mmax ?
    m^2 * bw.Γ / ((m^2 - bw.mR^2)^2 + bw.mR^2 * bw.Γ^2) : 0.0
polemap(bw::BreitWigner) = (bw.mR, bw.Γ)

"""
    MassDependentBW(mR, Γ0, mth; k = 4.0, mmin = mR - k*Γ0, mmax = mR + k*Γ0)

Spectral function with a mass-dependent, threshold-vanishing width
(`matrix_notes.tex` §Finite widths, at ℓ = 0):

    q(m)  = √(m² − mth²)/2                  the symmetric-split break-up momentum
    Γ(m)  = Γ0 · (mR/m) · q(m)/q(mR)        vanishes at m = mth; Γ(mR) = Γ0
    ϱ(m)  ∝ m² Γ(m) / ((m² − mR²)² + m² Γ(m)²)

`mth` is the resonance's decay threshold — the summed daughter masses of its
*lightest* channel, so the shape is a property of the species, not of any one
channel. The symmetric split (both effective daughters at `mth/2`) is exact for
the σ → ππ and an approximation of the q-ratio for asymmetric pairs; barrier
factors (ℓ ≥ 1) are deliberately not modeled. Degenerate limits: `mth = 0`
recovers a constant width exactly, and `Γ0 → 0` recovers the sharp pole.

The support's lower edge is clamped to `mth`, where ϱ vanishes like `√(m−mth)`
— [`parentmassnodes`](@ref) is told through `_lowerendpointvanishes` to use the
cosine-mapped rule there. Requires `0 ≤ mth < mR` (a pole below its own
threshold has no q(mR); use [`BreitWigner`](@ref) for such reopened states).
"""
struct MassDependentBW <: SpectralShape
    mR::Float64
    Γ0::Float64
    mth::Float64
    mmin::Float64
    mmax::Float64

    function MassDependentBW(mR, Γ0, mth, mmin, mmax)
        all(isfinite, (mR, Γ0, mth, mmin, mmax)) ||
            throw(ArgumentError("MassDependentBW: non-finite (mR, Γ0, mth, mmin, mmax) = $((mR, Γ0, mth, mmin, mmax))"))
        Γ0 > 0 || throw(ArgumentError("MassDependentBW: width must be positive, got Γ0 = $Γ0"))
        0 ≤ mth < mR || throw(ArgumentError(
            "MassDependentBW: need 0 ≤ mth < mR, got mth = $mth, mR = $mR; " *
            "a pole at or below threshold has no q(mR) — use BreitWigner"))
        mth ≤ mmin < mmax || throw(ArgumentError(
            "MassDependentBW: support must satisfy mth ≤ mmin < mmax, got [$mmin, $mmax]"))
        new(mR, Γ0, mth, mmin, mmax)
    end
end
MassDependentBW(mR, Γ0, mth; k = 4.0, mmin = mR - k * Γ0, mmax = mR + k * Γ0) =
    MassDependentBW(mR, Γ0, mth, max(mmin, mth), mmax)

# ϱ = m·(mΓ(m))/((m²−mR²)² + (mΓ(m))²) with mΓ(m) = mR Γ0 q(m)/q(mR): the
# division by m cancels analytically, so m = 0 needs no guard.
@inline function (md::MassDependentBW)(m)
    md.mmin ≤ m ≤ md.mmax || return 0.0
    q = sqrt(max((m - md.mth) * (m + md.mth), 0.0)) / 2
    qR = sqrt((md.mR - md.mth) * (md.mR + md.mth)) / 2
    scaled_width = md.mR * md.Γ0 * q / qR              # = m·Γ(m)
    return m * scaled_width /
           ((m^2 - md.mR^2)^2 + scaled_width^2)
end
polemap(md::MassDependentBW) = (md.mR, md.Γ0)
# The shape vanishes like √(m − mth) when its support starts at the threshold.
_lowerendpointvanishes(md::MassDependentBW) = md.mmin ≤ md.mth

# The tangent substitution flattens the Breit–Wigner peak. Normalization uses
# the full support, while decay nodes cover only the open part. A threshold-cut
# lower endpoint uses the cosine rule to absorb its integrable 1/√ singularity.
function parentmassnodes(bw::BreitWigner, mass_rule::MassRule,
                         threshold_mass = bw.mmin)
    normalization_nodes, normalization_weights =
        mass_rule.gl.x, mass_rule.gl.w
    node_count = length(mass_rule.gl)
    psi_of_mass(mass) =
        atan((mass^2 - bw.mR^2) / (bw.mR * bw.Γ))
    mass_squared_at(psi) =
        bw.mR^2 + bw.mR * bw.Γ * tan(psi)
    psi_min, psi_max = psi_of_mass(bw.mmin), psi_of_mass(bw.mmax)
    psi_half_width = (psi_max - psi_min) / 2
    psi_midpoint = (psi_max + psi_min) / 2

    normalization = 0.0
    for node_index in 1:node_count
        mass_squared = mass_squared_at(
            psi_midpoint + psi_half_width * normalization_nodes[node_index])
        mass_squared > 0 &&
            (normalization +=
                normalization_weights[node_index] * sqrt(mass_squared))
    end

    parent_masses, mass_weights = Float64[], Float64[]
    open_mass_minimum = max(bw.mmin, threshold_mass)
    open_mass_minimum < bw.mmax ||
        return parent_masses, mass_weights, normalization

    threshold_cuts_support = open_mass_minimum > bw.mmin
    open_nodes, open_weights = threshold_cuts_support ?
                               (mass_rule.cos.x, mass_rule.cos.w) :
                               (normalization_nodes, normalization_weights)
    open_half_width, open_midpoint = threshold_cuts_support ?
        ((psi_max - psi_of_mass(open_mass_minimum)) / 2,
         (psi_max + psi_of_mass(open_mass_minimum)) / 2) :
        (psi_half_width, psi_midpoint)
    open_psi_scale = open_half_width / psi_half_width
    for node_index in 1:node_count
        mass_squared = mass_squared_at(
            open_midpoint + open_half_width * open_nodes[node_index])
        mass_squared > 0 || continue
        push!(parent_masses, sqrt(mass_squared))
        push!(mass_weights, open_psi_scale * open_weights[node_index] *
                            sqrt(mass_squared))
    end
    return parent_masses, mass_weights, normalization
end

# Generic parent-mass average for any SpectralShape: the same ψ tangent map,
# weighted by the true shape times the substitution Jacobian,
#
#     ϱ(m) dm = ϱ(√s) · jacobian(s) dψ,
#     jacobian(s) = mR Γ₀ (1 + ((s − mR²)/(mR Γ₀))²) / (2√s),
#
# so the map flattens the peak while the weight stays exact. For `BreitWigner`
# the product collapses to √s/(2 mR) — a constant multiple of what the
# specialized method above accumulates, and the constant cancels when
# `BroadParent` divides by the normalization. The specialized method remains
# the dispatch target for `BreitWigner`; this one serves every other shape.
function parentmassnodes(shape::SpectralShape, mass_rule::MassRule,
                         threshold_mass = support(shape)[1])
    minimum_mass, maximum_mass = support(shape)
    pole_mass, pole_width = polemap(shape)
    normalization_nodes, normalization_weights =
        mass_rule.gl.x, mass_rule.gl.w
    node_count = length(mass_rule.gl)
    psi_of_mass(mass) =
        atan((mass^2 - pole_mass^2) / (pole_mass * pole_width))
    mass_squared_at(psi) =
        pole_mass^2 + pole_mass * pole_width * tan(psi)
    jacobian(mass_squared) =
        pole_mass * pole_width *
        (1 + ((mass_squared - pole_mass^2) / (pole_mass * pole_width))^2) /
        (2 * sqrt(mass_squared))
    psi_min, psi_max = psi_of_mass(minimum_mass), psi_of_mass(maximum_mass)
    psi_half_width = (psi_max - psi_min) / 2
    psi_midpoint = (psi_max + psi_min) / 2

    fullsweep(nodes, weights) = begin
        total = 0.0
        for node_index in eachindex(nodes)
            mass_squared = mass_squared_at(
                psi_midpoint + psi_half_width * nodes[node_index])
            mass_squared > 0 || continue
            total += weights[node_index] *
                     jacobian(mass_squared) * shape(sqrt(mass_squared))
        end
        total
    end

    parent_masses, mass_weights = Float64[], Float64[]
    open_mass_minimum = max(minimum_mass, threshold_mass)
    open_mass_minimum < maximum_mass ||
        return parent_masses, mass_weights,
               fullsweep(normalization_nodes, normalization_weights)

    # The cosine rule earns its place at a lower endpoint where the integrand
    # is not smooth: a threshold cutting into the support (as in the
    # specialized method), or a shape that itself vanishes like √ there.
    threshold_cuts_support = open_mass_minimum > minimum_mass
    use_cosine_rule = threshold_cuts_support || _lowerendpointvanishes(shape)
    open_nodes, open_weights = use_cosine_rule ?
                               (mass_rule.cos.x, mass_rule.cos.w) :
                               (normalization_nodes, normalization_weights)
    open_half_width, open_midpoint = threshold_cuts_support ?
        ((psi_max - psi_of_mass(open_mass_minimum)) / 2,
         (psi_max + psi_of_mass(open_mass_minimum)) / 2) :
        (psi_half_width, psi_midpoint)
    open_psi_scale = open_half_width / psi_half_width
    for node_index in 1:node_count
        mass_squared = mass_squared_at(
            open_midpoint + open_half_width * open_nodes[node_index])
        mass_squared > 0 || continue
        push!(parent_masses, sqrt(mass_squared))
        push!(mass_weights, open_psi_scale * open_weights[node_index] *
                            jacobian(mass_squared) * shape(sqrt(mass_squared)))
    end

    # `BroadParent` scales each node by `mass_weights[i] / normalization`, so
    # daughters per decay is `sum(mass_weights) / normalization`. That has to be
    # an *identity* when the open part is the whole support, not two quadratures
    # that happen to agree: taking the sum itself makes it one. Computing the
    # normalization separately on the Gauss-Legendre rule while the open nodes
    # used the cosine rule cost the rho 8.9e-5 of its daughters.
    #
    # When the threshold does cut, the two genuinely differ: the closed part
    # carries normalization and emits nothing, which is channel reopening, and
    # the deficit is the physics (2.56% of the sigma under a constant width).
    normalization = threshold_cuts_support ?
        fullsweep(normalization_nodes, normalization_weights) :
        sum(mass_weights)
    return parent_masses, mass_weights, normalization
end

"""
    BroadParent(daughter masses..., bw::BreitWigner;
                B = 1.0, nma = 16, rule_options...)

A decay whose parent has finite width: the underlying sharp decay (two-, three-,
or four-body, chosen by the number of daughter masses) averaged over the parent
spectral function `bw` (pole `bw.mR`, width `bw.Γ`). Returns a [`MassAverage`] —
the parent-mass nodes of every sub-decay are *flattened* into one node list, so
a broad-parent decay is just another `MassAverage`, handled exactly like a sharp
one. Nodes below the daughter threshold are dropped and enter only the
normalization, so a channel closed at the pole keeps its open-tail fraction
(channel reopening). `rule_options...` construct the shared [`Rules`].
"""
function BroadParent(build_decay, bw::SpectralShape, threshold_mass, mb,
                     rules::Rules)
    sampled_masses, mass_weights, normalization =
        parentmassnodes(bw, rules.ma, threshold_mass)
    parent_masses, daughter_energies, coefficients =
        Float64[], Float64[], Float64[]

    for node_index in eachindex(sampled_masses)
        sampled_masses[node_index] > threshold_mass || continue
        subdecay = build_decay(sampled_masses[node_index])::MassAverage
        all(isfinite, subdecay.coef) || continue
        if isempty(coefficients)
            # Every valid sub-decay has the same node count.
            required_capacity =
                length(sampled_masses) * length(subdecay.coef)
            sizehint!(parent_masses, required_capacity)
            sizehint!(daughter_energies, required_capacity)
            sizehint!(coefficients, required_capacity)
        end
        append!(parent_masses, subdecay.ma)
        append!(daughter_energies, subdecay.Eb)
        append!(coefficients,
                (mass_weights[node_index] / normalization) .* subdecay.coef)
    end

    # An empty list represents a channel closed across the full support.
    return MassAverage(parent_masses, float(mb), daughter_energies,
                       coefficients, rules.kern)
end

BroadParent(mb, mc, bw::SpectralShape, rules::Rules; B = 1.0) =
    BroadParent(ma -> TwoBodyDecay(ma, mb, mc, rules; B),
                bw, mb + mc, mb, rules)
BroadParent(mb, mc, md, bw::SpectralShape, rules::Rules; B = 1.0) =
    BroadParent(ma -> ThreeBodyDecay(ma, mb, mc, md, rules; B),
                bw, mb + mc + md, mb, rules)
BroadParent(mb, mc, md, me, bw::SpectralShape, rules::Rules; B = 1.0) =
    BroadParent(ma -> FourBodyDecay(ma, mb, mc, md, me, rules; B),
                bw, mb + mc + md + me, mb, rules)

BroadParent(mb, mc, bw::SpectralShape; B = 1.0, nma = 16, rule_options...) =
    BroadParent(mb, mc, bw, Rules(; nma, rule_options...); B)
BroadParent(mb, mc, md, bw::SpectralShape; B = 1.0, nma = 16, rule_options...) =
    BroadParent(mb, mc, md, bw, Rules(; nma, rule_options...); B)
BroadParent(mb, mc, md, me, bw::SpectralShape; B = 1.0, nma = 16, rule_options...) =
    BroadParent(mb, mc, md, me, bw, Rules(; nma, rule_options...); B)

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
`qgrid` may be a Float64 `AbstractRange` (uniform) or a sorted vector
(non-uniform). Float64 ranges are stored as ranges; other inputs are copied
to `Vector{Float64}`. The same storage rule applies independently to `pTs`.
"""
struct DecayMatrix{PT<:AbstractVector{Float64},QT<:AbstractVector{Float64}}
    n::Int
    pT::PT
    q::QT
    W::Matrix{Float64}
end

(M::DecayMatrix)(F::AbstractVector) = M.W * F
(M::DecayMatrix)(Fan) = M.W * map(Fan, M.q)

_stencil(momentum, grid::AbstractRange, P) =
    lagrangeweights(momentum, first(grid), step(grid), length(grid), P)
_stencil(momentum, grid::AbstractVector, P) =
    lagrangeweights(momentum, grid, P)

# Keep the compact, immutable representation that unlocks the uniform-grid
# stencil. Every other iterable is copied exactly as Float64 values, preserving
# mutation isolation and the actual samples of non-Float64 ranges.
_normalizegrid(grid::AbstractRange{Float64}) = grid
_normalizegrid(grid) = collect(Float64, grid)

# ---------------------------------------------------------------------------
# Fused assembly shares the qT/Z geometry and stencil across Fourier modes.
# Tuple-valued mode sums specialize fully; matrix rows use task-owned buffers.
# Independent rows make threaded and serial assembly bitwise identical.
# ---------------------------------------------------------------------------

# Contiguous Float64 row buffers avoid strided stencil writes and round only
# once when copied to Float32 output. Each caller or task owns its buffers.
function fillrows!(output_matrices, row_indices, decay::MassAverage, modes,
                   daughter_grid, parent_grid, P, row_buffers)
    for row_index in row_indices
        for row_buffer in row_buffers
            fill!(row_buffer, 0.0)
        end
        for node_index in eachindex(decay.coef)
            scatterrows!(row_buffers, modes, decay.ma[node_index], decay.mb,
                         decay.Eb[node_index], daughter_grid[row_index],
                         decay.kern, parent_grid, P, decay.coef[node_index])
        end
        for (output_matrix, row_buffer) in zip(output_matrices, row_buffers)
            @inbounds for column_index in eachindex(row_buffer)
                output_matrix[row_index, column_index] =
                    eltype(output_matrix)(row_buffer[column_index])
            end
        end
    end
end

"""
    rowscratch(modes::Tuple, parent_grid_length::Integer)

Create the row accumulators used by [`decaymatrices!`](@ref): one
`parent_grid_length`-long Float64 buffer per mode. One set serves one task.

Pass it as `scratch` to make the serial path allocate nothing. Build it once and
reuse it across as many calls as you like — each row zeroes its buffer before
use, so what it holds on entry never matters.

The threaded path allocates one set per task to prevent aliasing. Callers
parallelizing at a higher level should pass task-local scratch with
`threaded = false`.
"""
rowscratch(modes::Tuple, parent_grid_length::Integer) =
    map(_ -> zeros(Float64, parent_grid_length), modes)

"""
    decaymatrices!(output_matrices, decay, modes::Tuple, daughter_grid,
                   parent_grid; P = Val(4), threaded = true, scratch = nothing)

Fill caller-owned matrices `output_matrices` — one per entry of `modes`, each
`length(daughter_grid) × length(parent_grid)` — with the decay map, and return
`output_matrices`.

Every entry is written, so the destination need not be zeroed and its contents
on entry are irrelevant; outputs may be reused staging buffers or `Mmap`-backed
matrices — anything supporting `setindex!`.

The axes are trusted, used read-only, and never scanned: `parent_grid` must be
finite, nonnegative, strictly increasing, and contain at least `P` points.

With a `scratch` from [`rowscratch`](@ref), the serial path
(`threaded = false`) allocates nothing at all. `scratch` is ignored when
`threaded`, which allocates one set per task — and a `Task` per chunk besides,
so it can never reach zero.

[`decaymatrices`](@ref) is this with freshly allocated output.
"""
function decaymatrices!(output_matrices, decay::MassAverage, modes::Tuple,
                        daughter_grid, parent_grid;
                        P::Val = Val(4), threaded = true, scratch = nothing)
    daughter_count, parent_count = length(daughter_grid), length(parent_grid)
    length(output_matrices) == length(modes) ||
        throw(DimensionMismatch(
            "got $(length(output_matrices)) output matrices for $(length(modes)) modes",
        ))
    all(matrix -> size(matrix) == (daughter_count, parent_count),
        output_matrices) ||
        throw(DimensionMismatch(
            "each output must be $daughter_count×$parent_count " *
            "(length(daughter_grid) × length(parent_grid))",
        ))
    if !threaded && scratch !== nothing
        length(scratch) == length(modes) &&
            all(buffer -> length(buffer) == parent_count, scratch) ||
            throw(DimensionMismatch(
                "scratch needs $(length(modes)) buffers of length $parent_count",
            ))
    end
    integer_modes = map(Int, modes)
    daughter_count == 0 && return output_matrices

    if threaded
        task_count = min(Threads.nthreads(), daughter_count)
        row_chunks = Iterators.partition(eachindex(daughter_grid),
                                         cld(daughter_count, task_count))
        @sync for row_chunk in row_chunks
            Threads.@spawn fillrows!(output_matrices, row_chunk, decay,
                                     integer_modes, daughter_grid, parent_grid,
                                     P, rowscratch(modes, parent_count))
        end
    else
        row_buffers =
            scratch === nothing ? rowscratch(modes, parent_count) : scratch
        fillrows!(output_matrices, eachindex(daughter_grid), decay,
                  integer_modes, daughter_grid, parent_grid, P, row_buffers)
    end
    return output_matrices
end

function decaymatrices(decay::MassAverage, modes::Tuple, daughter_grid,
                       parent_grid;
                       P::Val = Val(4), threaded = true, T::Type = Float64,
                       scratch = nothing)
    # Zeroed output makes a future partial writer fail safely as zeros.
    output_matrices =
        map(_ -> zeros(T, length(daughter_grid), length(parent_grid)), modes)
    decaymatrices!(output_matrices, decay, modes, daughter_grid, parent_grid;
                   P, threaded, scratch)
    return collect(output_matrices)
end

# Fuse up to 16 modes; beyond that, build one mode at a time to limit compilation.
# `Tuple(modes)` is a runtime tuple, so the dispatch to the core is dynamic (once
# per call, cold); the ::return annotation keeps the boundary type-stable for the
# caller. Beyond 16 modes, modes are built sequentially but each single-mode
# matrix retains its default threaded row build. Caller `threaded` and `scratch`
# remain unused on that path to preserve existing behavior.
function decaymatrices(decay::MassAverage,
                       modes::AbstractVector{<:Integer},
                       daughter_grid, parent_grid;
                       P::Val = Val(4), threaded = true, T::Type = Float64,
                       scratch = nothing)::Vector{Matrix{T}}
    length(modes) ≤ 16 &&
        return decaymatrices(decay, Tuple(modes), daughter_grid, parent_grid;
                             P, threaded, T, scratch)
    return [Matrix{T}(decaymatrices(decay, (Int(mode),), daughter_grid,
                                    parent_grid; P, T)[1])
            for mode in modes]
end

function DecayMatrix(decay::MassAverage, mode::Integer, daughter_grid,
                     parent_grid; P::Val = Val(4))
    normalized_daughter_grid = _normalizegrid(daughter_grid)
    normalized_parent_grid = _normalizegrid(parent_grid)
    matrix = decaymatrices(decay, (Int(mode),), normalized_daughter_grid,
                           normalized_parent_grid; P)[1]
    return DecayMatrix(Int(mode), normalized_daughter_grid,
                       normalized_parent_grid, matrix)
end

# Chebyshev recurrence for Tₙ(Z), even in n, without evaluating acos/cos.
@inline function chebT(n::Int, Z::Float64)
    n = abs(n)
    n == 0 && return 1.0
    Tm1, Tm = 1.0, Z
    for _ in 2:n
        Tm1, Tm = Tm, 2 * Z * Tm - Tm1
    end
    return Tm
end

# Tuple mapping specializes modes while sharing geometry and stencil work.
@inline chebvals(modes::Tuple, Z) = map(n -> chebT(n, Z), modes)

# Multi-mode scatter: mirrors decaykernel's qT structure but scatters the
# quadrature factor onto the parent-grid stencil instead of evaluating Fa. The
# inner integral is done in Z = D/Q rather than Δη (an η-independent source never
# needs the rapidity), so everything is algebraic (matrix_notes §8):
#     2 ∫dΔη Θ/√(Q²−D²) · 2Tₙ(Z)  =  (4/M) ∫_{Zlo}^{1} Tₙ(Z) dZ / √((1−Z²)(c²−1)),
# c = (C+QZ)/M, Zlo = (M−C)/Q. Both windows use a Chebyshev–Gauss rule with the
# inverse-√ weight removed analytically (R = √(c²−1) full, √((Q/M)(1+Z)(c+1))
# partial), so no positivity guard. Only Tₙ(Z) varies between modes.
function scatterrows!(row_buffers, modes::NTuple{K,Int}, ma, mb, Eb, pT,
                      kernel_rules::KernelRules, parent_grid, P,
                      node_prefactor) where {K}
    q_nodes, q_weights = kernel_rules.q.x, kernel_rules.q.w
    chebyshev_rule, dense_chebyshev_rule =
        kernel_rules.zc, kernel_rules.zc4
    pb = sqrt(max(Eb^2 - mb^2, 0.0))
    C = ma * Eb
    mbT = hypot(mb, pT)
    qT_min = max(ma * (Eb * pT - pb * mbT) / mb^2, 0.0)
    qT_max = ma * (Eb * pT + pb * mbT) / mb^2
    qT_split = ma * (pb * mbT - Eb * pT) / mb^2
    segments = qT_min < qT_split < qT_max ?
               ((qT_min, qT_split), (qT_split, qT_max)) :
               ((qT_min, qT_max), (qT_max, qT_max))
    parent_min, parent_max = first(parent_grid), last(parent_grid)
    zero_mode_sums = ntuple(_ -> 0.0, Val(K))

    for (qT_start, qT_stop) in segments
        qT_start < qT_stop || continue
        qT_half_width = (qT_stop - qT_start) / 2
        qT_midpoint = (qT_start + qT_stop) / 2
        @inbounds for q_index in eachindex(q_nodes)
            qT = qT_midpoint + qT_half_width * q_nodes[q_index]
            parent_min ≤ qT ≤ parent_max || continue
            M = hypot(ma, qT) * mbT
            Q = qT * pT
            M < C + Q || continue
            Zlo = (M - C) / Q
            mode_sums = zero_mode_sums
            if Zlo ≤ -1
                z_rule = Zlo > -1.5 ?
                         dense_chebyshev_rule : chebyshev_rule
                for Z in z_rule.x
                    c = (C + Q * Z) / M
                    mode_sums = mode_sums .+
                                chebvals(modes, Z) ./
                                sqrt((c - 1) * (c + 1))
                end
                mode_sums = mode_sums .* z_rule.w
            else
                # analytic cancellation (matrix_notes §8.2): endpoint weight removed,
                # regular R² = (Q/M)(1+Z)(c+1) > 0, no guard; densify near Zlo→−1.
                z_rule = Zlo < -0.5 ?
                         dense_chebyshev_rule : chebyshev_rule
                z_half_width = (1 - Zlo) / 2
                z_midpoint = (1 + Zlo) / 2
                for ζ in z_rule.x
                    Z = z_midpoint + z_half_width * ζ
                    c = (C + Q * Z) / M
                    mode_sums = mode_sums .+
                                chebvals(modes, Z) ./
                                sqrt((Q / M) * (1 + Z) * (c + 1))
                end
                mode_sums = mode_sums .* z_rule.w
            end
            common_factor =
                node_prefactor * q_weights[q_index] *
                qT_half_width * qT * (4 / M)
            stencil_start, stencil_weights = _stencil(qT, parent_grid, P)
            for mode_index in 1:K
                mode_factor = common_factor * mode_sums[mode_index]
                for (offset, stencil_weight) in pairs(stencil_weights)
                    row_buffers[mode_index][stencil_start + offset - 1] +=
                        mode_factor * stencil_weight
                end
            end
        end
    end
    return row_buffers
end

end # module
