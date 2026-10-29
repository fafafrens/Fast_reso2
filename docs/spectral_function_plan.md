# Spectral-function design and remaining work

Status: **interface and mass-dependent shape implemented.** `MassDependentBW`
is the default for broad parents in both `CascadeTable` (`spectral = :massdep`)
and `ThermalModel` (`shape = :massdep`). `BreitWigner` remains the
constant-width comparison. Tail control, barrier factors, and a summed
partial-width model remain open.

The derivation is in `matrix_notes.tex`, under “What the width shape controls,
and what it does not” and “A spectral-function interface.”

## Why there are two shapes

A constant-width relativistic Breit–Wigner does not vanish at a decay
threshold. For a broad state it therefore assigns spectral weight to masses
the state cannot physically carry. Its high-mass tail is also determined by
the finite support window rather than only by `(mR, Γ)`.

The implemented S-wave mass-dependent width removes the first defect:

```text
Γ(m) ∝ (mR / m) q(m) / q(mR)
```

It does not solve the tail problem. For ℓ = 0 the asymptotic factors largely
cancel, so a support cap, form factor, or higher partial wave is still needed
when the high-mass tail matters.

For the σ example (`mR = 0.475`, `Γ = 0.550`), the constant shape formerly
placed 2.56% of its weight below `2mπ`; `MassDependentBW` places none there.
The same study found that hard-pion changes are more sensitive to the tail
than integrated yields, which is why tail control remains a separate task.

## Implemented interface

A new shape derives from `SpectralShape` and supplies:

```julia
support(shape)       # finite (mmin, mmax)
polemap(shape)       # (mR, Γ0), used by the tangent peak-flattening map
shape(m)             # unnormalised weight, zero outside support
```

A shape may additionally implement `_lowerendpointvanishes(shape)` so the
generic quadrature uses the cosine-mapped endpoint rule. `parentmassnodes` is
the generic consumer of this contract; it is not another method each subtype
must provide. `BreitWigner` keeps a specialized algebraically flattened fast
path, while other shapes use the exact shape-times-Jacobian weight.

Nothing after parent-mass quadrature inspects the shape. `BroadParent` consumes
sampled masses, weights, and normalization; `MassAverage`, matrix assembly,
packing, and cascade execution are unchanged.

## Current `MassDependentBW`

The actual representation is:

```julia
struct MassDependentBW <: SpectralShape
    mR::Float64
    Γ0::Float64
    mth::Float64
    mmin::Float64
    mmax::Float64
end
```

`mth` is the lightest nonzero decay threshold of the resolved species. With
only the summed threshold available, the current S-wave momentum is the
symmetric-split approximation

```text
q(m) = sqrt(m² - mth²) / 2.
```

There is no implemented orbital-angular-momentum field and no barrier factor.
Those require per-state parity or partial-wave data and are scoped in
`angular_momentum_plan.md`.

The threshold is resolved once on `DecayList` and preserved by channel
filters. This is necessary for two independent invariants:

- filtering topology must not change a species’ primary thermal density;
- keys for a filtered plan must remain a subset of a table built from the full
  model.

Each decay channel still applies its own kinematic closure when parent-mass
nodes are enrolled.

## Selection and fallbacks

Broad species normally use `MassDependentBW`. A pole at or below its own
species threshold has no real `q(mR)`, so reopened states fall back to the
constant shape. `spectral = :constant` or `shape = :constant` selects the
constant shape explicitly for every broad state.

Both paths use the same finite window, currently controlled by
`width_window = 4.0` in the cascade and `k = 4.0` in the thermal model.

## Validation already present

The suite covers:

- constant-shape regression and the sharp-width limit;
- zero mass-dependent weight below threshold;
- number conservation and parent-mass quadrature convergence;
- exact open-support normalization when endpoint rules differ;
- equality of cascade and thermal spectral formulas;
- channel reopening and fallback behavior;
- invariant species thresholds under topology filtering.

## Remaining decisions

1. **Tail policy.** Sweep physically motivated caps for the σ and other broad
   states before choosing a permanent support rule. A per-shape cap or form
   factor is more explicit than relying only on a global width count.
2. **Total width.** The table supplies one total pole width per species. A
   correct `Γ(m)` should sum partial widths, but partial fractions and angular
   momenta are absent. The current lightest-threshold S-wave shape is therefore
   a controlled approximation, not a channel-by-channel total-width model.
3. **Barrier factors and ℓ.** Resolve the parity/model-label data gap described
   in `angular_momentum_plan.md`; do not apply one global ℓ.
4. **σ cross-check.** Re-run the spectra and thermal-yield comparisons over
   several tail caps using the isospin-corrected table with and without the σ.

The physically complete σ treatment is an S-matrix or phase-shift
construction, including the repulsive isospin-2 channel. That signed
contribution is not naturally representable as a positive parent in a decay
cascade and remains outside this machinery.
