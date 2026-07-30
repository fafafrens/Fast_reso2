# Plan: a generic spectral-function type

Status: proposed, not implemented. Background and formulae are in
`matrix_notes.tex`, §"What the width shape controls, and what it does not"
and §"A spectral-function interface".

## Why

`BreitWigner` in `DecaySpectra.jl` is a constant-width relativistic
Breit–Wigner truncated to
`[max(mR - kΓ, 0.05 GeV), mR + kΓ]`. Here `k` is the table-local
`width_window` option and defaults to 4. That is fine for narrow states and
has two defects for broad ones, which are *independent of each other*:

1. **Threshold.** A constant width does not vanish at `m = Σ m_daughter`, so
   the shape carries weight where the decay is closed. `BroadParent`
   normalises over the full support while placing decay nodes only on the
   open part — correct for channel reopening, but it means that weight is
   normalised in and emits nothing. The state yields fewer daughters per
   decay than its branching ratios say.

2. **Tail.** The weight above any given mass is set by the truncation
   constant `k`, not by `(mR, Γ)`.

Measured for the σ (`mR = 0.475`, `Γ = 0.550`), the extreme case in the
current tables:

| weight in | constant Γ | mass-dependent Γ(m) |
|---|---|---|
| below 2m_π | 2.56% | **0.00%** |
| 0.279–1.0 GeV | 69.85% | 68.53% |
| 1.0–2.0 GeV | 22.73% | 25.78% |
| 2.0–2.675 GeV | 4.86% | 5.69% |

| weight above 1 GeV | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| constant Γ | 1.7% | 19.2% | 27.6% | 31.8% |
| Γ(m) | 2.0% | 22.1% | 31.5% | 36.1% |

**A mass-dependent width fixes defect 1 and not defect 2.** For an S-wave
decay `q(m) → m/2` cancels the `mR/m` prefactor, so `Γ(m) → Γ₀ mR/2q(mR)`,
a constant *larger* than `Γ₀` (0.680 vs 0.550 for the σ). The tail keeps its
`1/m²` fall-off with a slightly bigger coefficient. Only `ℓ ≥ 1` suppresses
it. Defect 2 needs a support cap or a form factor.

Downstream effect, measured on the cascade with the σ inserted: pion dN/dy
+7.4% (π⁺) / +6.6% (π⁰), but +17% at pT = 1 GeV — and that 17% drops to
+5% at Γ = 0.30 with dN/dy unchanged (1.0738 → 1.0734). The yield comes from
the peak, the hard-pion excess from the tail.

## Design

Nothing downstream of the parent-mass average inspects the shape.
`BroadParent` consumes only `(masses, weights, normalisation)` and the kernel
average only `(ma[i], Eb[i], c[i])`. So the contract is two methods.

```julia
abstract type SpectralShape end

support(ρ::SpectralShape) = (ρ.mmin, ρ.mmax)
polemap(ρ::SpectralShape) = (ρ.mR, ρ.Γ0)   # builds the ψ map only, not the shape
(ρ::SpectralShape)(m)                      # unnormalised weight, 0 off support
parentmassnodes(ρ, rule::MassRule, threshold) -> (masses, weights, normalisation)
```

One generic `parentmassnodes` serves every shape. With
`s = mR² + mR Γ tan ψ` the constant-width case satisfies

    ϱ(m) dm = √s dψ / (2 mR)

which is exactly what `DecaySpectra.jl:458` accumulates today
(`weight * sqrt(mass_squared)`) — so the tan-substitution is doing analytic
flattening, and generalising means keeping the map and weighting by the true
shape times the Jacobian:

```julia
jacobian(s) = mR * Γ * (1 + ((s - mR^2) / (mR * Γ))^2) / (2 * sqrt(s))
# accumulate  w[i] * jacobian(s) * ρ(sqrt(s))   instead of   w[i] * sqrt(s)
```

For `BreitWigner` the product collapses back to `√s / (2 mR)`, a constant
multiple of the current weights, and the constant cancels when `BroadParent`
divides by the normalisation. Existing results are therefore reproduced
exactly; the current specialised method can stay as the fast path.

The mass-dependent shape is then another subtype carrying the daughter
masses that fix its threshold:

```julia
struct MassDependentBW <: SpectralShape
    mR::Float64; Γ0::Float64
    m1::Float64; m2::Float64
    ℓ::Int
    mmin::Float64; mmax::Float64
end

qcm(m, m1, m2) = sqrt((m^2 - (m1+m2)^2) * (m^2 - (m1-m2)^2)) / (2m)
width(ρ, m) = m > ρ.m1 + ρ.m2 ?
    ρ.Γ0 * (ρ.mR/m) * (qcm(m, ρ.m1, ρ.m2) / qcm(ρ.mR, ρ.m1, ρ.m2))^(2ρ.ℓ+1) : 0.0
```

### Call sites

Three, and no more:

- `BroadParent(build_decay, bw::BreitWigner, ...)` at `DecaySpectra.jl:477` —
  relax the annotation to `::SpectralShape`.
- `decayfor` at `CascadeTables.jl:738` — choose which shape to construct.
- the export list.

The kernel, `MassAverage`, the matrix assembly and the cascade read only the
per-node triple and cannot distinguish the shapes.

## Open question: per-channel vs total width

`Γ(m)` depends on which channel the state decays through, but the decay list
gives one *total* `Γ` per species and `ChannelKey` carries `(ma, Γ, mb, mc, …)`
per channel. Two options:

- **Per-channel threshold** — build `Γ(m)` from that channel's own daughters.
  Cheap and constructible from a `ChannelKey` alone, but a multi-channel
  resonance then has a different total width in each channel.
- **Summed partial widths** — correct, but the shape can no longer be built
  from a `ChannelKey`, and construction has to move up into `CascadePlan`.

Moot for the σ (one final state, two charge splits sharing a threshold), so
it can be deferred if the goal is only to bracket the σ.

## Staged plan

1. **Support cap first.** The single change that moves the number. Either a
   per-shape `mmax` honoured by `decayfor`, or a table-local `width_window`
   sweep.
   Measure the σ at cap = 1.0, 1.5, 2.675 GeV. Cheap, and it decides whether
   step 2 is worth doing.
2. **`SpectralShape` + generic `parentmassnodes`.** Regression: every
   existing result must be bit-identical, since `BreitWigner` keeps its
   specialised method and the generic path reproduces it up to a constant
   that cancels.
3. **`MassDependentBW`.** Validate against: Γ→0 recovering the sharp limit,
   number conservation, zero weight below threshold, and spectral
   convergence in `nma` (the same four checks the constant-width path uses,
   `matrix_notes.tex` §"The main case").
4. **Re-run the σ comparison** against
   `decays_QM2016Plus_massorder_isofix.dat` and
   `decays_QM2016Plus_massorder_isofix_sigma.dat`.

Expected outcome of steps 2–3 alone: the 2.56% threshold leak disappears and
the +17% at 1 GeV stays. If that excess is what needs bracketing, step 1 is
the answer and steps 2–3 are correctness cleanup.

## Not in scope

The physically correct treatment of the σ is not a spectral function at all
but the S-matrix / phase-shift construction, where the weight is
`dδ_I^ℓ/dm` and the repulsive I=2 channel largely cancels the attractive
I=0 one. That cannot be expressed as a row in a decay table — a cascade
needs a positive-definite parent to decay — so it belongs outside this
machinery. A tabulated `dδ/dm` would, however, satisfy the interface above
if the I=2 piece is handled separately.
