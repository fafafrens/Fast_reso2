# Plan: barrier factors and per-state ℓ

Status: **proposed, blocked on one decision** (§"The fork", below). Continues
`spectral_function_plan.md`, whose steps 2–3 are implemented: `SpectralShape`,
the generic `parentmassnodes`, and `MassDependentBW` at ℓ = 0 as the default in
both the cascade and the thermal model.

## Why

`MassDependentBW` implements `matrix_notes.tex` eq. (massdep-width) at **ℓ = 0
and without the Blatt–Weisskopf factor**. That fixed the defect that mattered —
a resonance carrying weight below the mass of its own decay products — with no
new parameter and no new data. It leaves the centrifugal barrier unclaimed, and
the barrier is the larger effect.

Thermal density at `T = 156.5 MeV`, ℓ = 0 against ℓ = 1 (Blatt–Weisskopf,
`R = 1 fm`):

| state | ℓ = 0 → ℓ = 1 |
|---|---:|
| ρ(770) | **−18.7%** |
| Δ(1232) | **−11.4%** |
| K*(892) | −5.8% |
| σ / f0(500) — genuinely ℓ = 0 | **+55.3%** if wrongly given ℓ = 1 |

For scale, the whole `:constant` → `:massdep` change moved the proton by −7.5%.
ρ, Δ and K* are the dominant feed-down sources, so most of the remaining
finite-width physics is here.

The σ row is the reason this cannot be a global switch. A blanket ℓ = 1 would be
**worse than the ℓ = 0 shipped today**. ℓ must be per state.

## The physics: ℓ is determined, not chosen

For a two-body decay a → b + c, conservation fixes ℓ:

- **angular momentum** — `S ∈ |J_b − J_c| … J_b + J_c`, `L ∈ |J_a − S| … J_a + S`
- **parity** — `P_a = P_b · P_c · (−1)^L`, which fixes the *parity* of L

Together these normally leave one lowest allowed value, and the barrier
suppresses the rest, so the lowest is the physical one:

| decay | J^P | S | P constraint | ℓ |
|---|---|---|---|---|
| ρ → ππ | 1⁻ → 0⁻0⁻ | 0 | (−1)^L = −1 | 1 |
| f0(500) → ππ | 0⁺ → 0⁻0⁻ | 0 | (−1)^L = +1 | 0 |
| Δ → Nπ | 3/2⁺ → 1/2⁺0⁻ | 1/2 | (−1)^L = −1 | 1 |
| K* → Kπ | 1⁻ → 0⁻0⁻ | 0 | (−1)^L = −1 | 1 |

**Isotropy survives ℓ ≥ 1.** The kernel integrates over an isotropic emission
angle (`matrix_notes.tex` §"integrating the two-body factor over the isotropic
emission angle"), and that stays valid because ℓ controls two separable things:

1. the angular distribution for a *given* parent substate, ∝ `|Y_ℓm|²`;
2. the mass dependence of the width, `Γ(m) ∝ q^(2ℓ+1) B_ℓ`.

For an **unpolarized** parent the first averages away —
`Σ_m |Y_ℓm|² = (2ℓ+1)/4π`, constant, because the density matrix is ∝ identity
and there is no axis to point along. A thermal source produces resonances
unpolarized, so ℓ enters *only* through `Γ(m)`, a scalar function of mass —
exactly and only where `MassDependentBW` puts it. The kernel never sees ℓ.

Verified rather than assumed: a p-wave shape defined against the public
`SpectralShape` contract, with no change to `DecaySpectra`, conserves daughter
number to 2e-5 — the same as ℓ = 0.

Out of scope, and the reason the above is an assumption rather than a theorem:
measured spin alignment (ρ⁰, K*⁰ with ρ₀₀ ≠ 1/3) and vorticity-induced
polarization both break the unpolarized premise, and sequential decays can
transfer polarization down a cascade. This code is spin-blind throughout.

## The data problem: parity is not in the tables

`ParticleInfo` stores `degeneracy`, so **J is recoverable** (`spin()`), but
**parity is stored nowhere** — not in the struct, not in any `.dat` file, not in
the legacy AZYHYDRO or THERMINATOR sources. The word does not appear in the
repo. Half the input for the rule above is missing.

It is not recoverable from the MC IDs either: `P = (−1)^(L+1)` from the meson
`n_L` digit works only for well-formed PDG meson IDs — not the `9xxxxxx`
specials (f0(500) is `9000221`), not baryons, and not the 1167 invented
quark-model IDs.

Coverage if parity comes from PDG: **311 of 650 broad feeders are
PDG-numbered, carrying 93.0% of the feed-down to π and p.** The head of the
list is ω, η, ρ⁰, ρ±, K*, f0(980), f2(1270) — all PDG, all unambiguous. The one
significant non-PDG entry is f0(500), which is ℓ = 0 anyway.

### Getting parity in: recommended shape

`0` is a safe sentinel for "unknown" — parity is ±1 for any eigenstate — and
unknown maps to ℓ = 0, i.e. exactly today's behaviour. The change degrades to
the status quo rather than to something wrong.

**Recommendation: a separate table, not a new column.** Column 12+ of the
canonical particle format is already claimed by aliases
(`src/readers/Canonical.jl:26`), so inserting parity there is a latent format
collision — no file uses aliases today, so nothing would break immediately,
which is what makes it dangerous. A separate `data/parities.dat` mapping
`id → P`:

- leaves all **6,374 particle rows across 9 files** untouched;
- needs no format bump and no change to the existing reader path;
- is naturally partial — absence *is* the unknown sentinel;
- follows the precedent of the charm/bottom supplements, which are separate
  files merged in.

Cost: one generated file (~351 rows from the PDG API), one loader, one optional
field on `ParticleInfo`.

## The fork (needs a decision before implementing)

ℓ is a property of a **channel**. The spectral function is a property of a
**species** — a resonance has one propagator. `MassDependentBW` is built per
species, from the lightest channel's threshold. Two measurements decide how to
reconcile that, and they pull in opposite directions:

| | |
|---|---:|
| broad keys shared by **more than one parent species** | **3241 / 3681 = 88.0%** |
| broad species with channels of **differing threshold** | **671 / 713 = 94.1%** |

ℓ is not a runtime argument under any option: it determines the matrix, so it
lives in `ChannelKey` (as `mth` already does) and is resolved at table-build
time.

**Option 1 — per-channel Γ(m).** ℓ and threshold from each channel's own
daughters, both in the key. Architecturally trivial; follows the `mth`
precedent exactly. But 94% of broad species are multi-threshold, so nearly
every resonance would get a *different spectral function per channel* — saying
the ρ has one mass distribution decaying to ππ and another decaying to anything
else. Cheap, and physically wrong in a way that is at least easy to state.

**Option 2 — summed partial widths.** `Γ(m) = Σ_c BR_c · Γ_c(m)`, one shape per
species, correct. Cannot be built from a `ChannelKey`, which knows nothing of
the parent's other channels: the shape must be constructed at `DecayList` level
and identified in the key by a small shape ID, with the table holding
`Dict{shape_id => SpectralShape}`. Correct physics, but a species-specific
shape breaks the **88%** of broad-key sharing — not `mth`'s 2.4%, most of the
broad table — and adds an indirection plus a second thing to serialize.

**Option 3 — ℓ only where unambiguous.** Apply the barrier factor only to
species whose channels share one threshold; leave the rest at ℓ = 0. Never
claims a resonance has two spectral functions, and needs no architecture
change. Measured coverage, weighting each species by `|Δn/n|` from ℓ = 0 → 1
times its feed-down volume:

```
single-threshold species:  35.9%   (includes rho0, rho+ — three of the top four)
multi-threshold species:   64.1%   (f0(500) is #1 but is genuinely l = 0;
                                    f2(1270), f0(1370), pi2(1670) are real losses)
```

So option 3 buys about a third of the effect for the smallest change. That is
less than it first appeared: weighting by feed-down volume alone suggested 30.8%
and flattered ω and η, whose widths (0.008, 0.001 GeV) put them barely over the
`width_cut` and make their shapes near-delta regardless of ℓ.

**Unresolved.** Option 2 is the only one that is right; option 3 is the only one
that is cheap; option 1 is neither but matches the existing structure. The
88%-sharing figure is the cost that has to be accepted or engineered around, and
that is a call about how much table size and complexity the correct answer is
worth.

## Implementation, once the fork is settled

1. **`data/parities.dat` + loader.** Generated from the PDG API for the
   PDG-numbered species; optional `parity` field on `ParticleInfo`, default 0.
   Regenerable and provenance-stamped like the charm/bottom supplements.
2. **ℓ from quantum numbers.** `channelangularmomentum(parent, daughters)`
   returning the lowest allowed L, or `nothing` when any parity is unknown or
   the daughter count is not 2. Two-body only — three- and four-body decays go
   through a cluster mass and have no single ℓ.
3. **`MassDependentBW` gains `ℓ` and `R`.** Non-breaking: both default to
   ℓ = 0 / unused, and the ℓ = 0 branch must stay bit-identical (the existing
   test 3e anchors it).
4. **Whichever fork option.** Key field or shape registry per §"The fork".
5. **Tests.** ℓ derivation against the table above (ρ, σ, Δ, K* by hand);
   number conservation at ℓ = 1 (the p-wave probe above, promoted into the
   suite); bit-identity of the ℓ = 0 path; `R`-sensitivity recorded so the
   choice of radius is visible rather than buried.
6. **Re-measure** the figure, README and section-10 anchors, as in
   `d9899dd`.

## Verification

```bash
julia --project -t auto runtests.jl
julia --project -t auto figures/thermal_yields.jl
CASCADE_BENCH_GRID=16 CASCADE_BENCH_SAMPLES=50 \
  julia --project -t 2 benchmarks/cascade_plan.jl   # judge by allocations, not time
```

Expected: ρ, Δ, K* densities fall by roughly the table in §"Why"; the σ is
unchanged (it is ℓ = 0 either way); `p/π` falls further from 0.0738 toward the
measured 0.05. Broad-key count and cache size move by whatever the fork option
implies — that number is the headline cost and should be reported.

Benchmark note, learned the hard way in `cascade_plan.md`: this host's
execution times carry ~13% placement noise, so A/B pairs must be interleaved and
claims rested on allocations.

## Out of scope

- **The tail cap** — `spectral_function_plan.md` step 1, still open, and
  independent: Γ(m) fixed the threshold defect and explicitly not the tail,
  which is still set by `width_window` rather than by physics.
- **Polarization** in any form (see §"The physics").
- **Mass-dependent branching ratios.** BRs stay fixed; only the shape moves.
- **Three- and four-body ℓ.** Those decays are averaged over a cluster mass and
  have no single orbital angular momentum.
