# FastReso2

Resonance **feed-down** for particle spectra in heavy-ion collisions: given a
thermal source of primary resonances, propagate every decay (2-, 3-, and
4-body) through the full feeder-before-fed cascade to obtain the final-state
spectra, including azimuthal (elliptic-flow) modulation.

The decay map is represented as a **matrix** in transverse momentum and
azimuthal Fourier mode, built once per channel and reused across the cascade.
The method — the two-body kernel and its Chebyshev–Gauss quadrature, the
effective-mass recursion behind 3-/4-body decays, the finite parent-width mass
average, and the (documented, not-yet-coded) rapidity-Toeplitz extension — is
derived in [`matrix_notes.tex`](matrix_notes.tex).

**Implemented:** 2-/3-/4-body decays, boost-invariant (η-independent) sources,
Fourier modes `n = 0, 2`, the full feeder-before-fed cascade, channel
filtering, finite parent widths with channel reopening, and reusable compiled
`CascadePlan` and `ThermalPlan` execution. Broad parents use the
threshold-vanishing `MassDependentBW` by default; the constant-width
`BreitWigner` remains available for controlled comparisons. Canonical runtime
inputs separate particle metadata from decay topology and retain source
provenance.

**Still open:** rapidity-dependent (η-resolved) sources, cluster or observed-
daughter widths, broad-state tail control beyond `width_window`, and
state-dependent angular-momentum barrier factors.

## Status and next improvements

**Phase 1 — complete.** The numerical core, canonical data conversion,
thermal-yield path, spectral-shape interface, and FastReso cross-checks are
implemented and covered by the Julia/Python suites. The former particle-
catalog, isospin-fix, and spectral-function branches have been consolidated
into `main`; the quickstart and thermal examples use `data/*_isofix.dat` as the
corrected default. The unmodified AZYHYDRO conversion is retained for
provenance, historical figures, benchmarks, and upstream comparisons, with
each such script naming its input.

**Phase 2 — foundation complete.** `src/FastReso2.jl` is the package entry
point and loads each component once. `src/DecayModel.jl` owns the resolved
model, filtering, validation, and graph-derived feeder-before-fed order;
cascade and thermal execution now share one `ParticleInfo`, `DecayList`, and
spectral-shape type world. Cycles are detected explicitly, and direct callable
decay kernels are serial by default with `threaded = true` as an explicit
opt-in.

**Next:** split the legacy assertion suite into component `Test.@testset`s and
fast/numerical/external tiers, then separate core, figure, and benchmark
dependency environments. The remaining type candidates are recorded in
[`type_simplification_plan.md`](type_simplification_plan.md); physics work then
returns to spectral tails, angular-momentum data, and the rapidity extension.

## Requirements

**Julia ≥ 1.12** — the table build uses `Threads.@threads :greedy` (1.11) and
`OncePerTask` (1.12), which gives each of that loop's tasks its own row buffers.

Dependencies are declared in `Project.toml`; instantiate once, then run
everything with `--project`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The core needs `FastGaussQuadrature` and `QuadGK` plus the standard libraries
`LinearAlgebra`, `Serialization`, `SHA`, `Dates`, and `Printf`; the figure
scripts add `Plots`.

## Quick start

Checked in as `quickstart.jl` at the repository root; run it with
`julia --project=. -t auto quickstart.jl`. The checked-in script mirrors the
artifact-free walkthrough shown below: one decay, one matrix build, and one
full cascade using the corrected isospin table and the default `:massdep`
shape. It prints diagnostics but does not write files.

```julia
## 1. One decay, on its own
using FastReso2
using Printf

ρ = TwoBodyDecay(0.7752, 0.1396, 0.1396)          # ρ → π + π  (parent, observed, partner)
Fa(qT, η, ϕ) = exp(-hypot(0.7752, qT) * cosh(η) / 0.155)     # any parent spectrum
@printf("pion from ρ at pT = 0.5: %.4e\n", ρ(Fa, 0.5, 0.0, 0.0))

grid = range(0.0, 4.0, length = 134)              # parent and daughter share the grid
W = decaymatrices(ρ, [0, 2], grid, grid)          # one matrix per Fourier mode
@printf("W[n=0] is %s — daughter = W * parent\n", size(W[1]))

## 2. The whole cascade
dl = DecayList(
    readparticles("data/particles_QM2016Plus_isofix.dat"),
    readdecays("data/decays_QM2016Plus_isofix.dat"),
)
T  = 0.155
primary(p, n, qs) =                               # thermal source, 4% v₂ on mode 2
    (n == 0 ? 1.0 : 0.04) * p.degeneracy * [exp(-hypot(p.mass, q) / T) for q in qs]

qs    = range(0.0, 4.0, length = 61)
# One shot: build/load a table, compile a plan, then execute that plan.
res   = runcascade(
    dl; primary, grid = qs, spectral = :massdep, parallel = :modes,
)

## 3. Read it out
π⁺ = 211
@printf("\n  pT    primary       +feed-down   ratio     v₂\n")
for i in (1, 16, 31)
    prim, tot = res.Fprim[π⁺][1][i], res.F[π⁺][1][i]
    @printf("%5.2f  %.4e   %.4e   %5.2f   %+.3f\n",
            qs[i], prim, tot, tot / prim, res.F[π⁺][2][i] / tot)
end
```

The printed values are deliberately not duplicated here: they depend on the
selected table and spectral-shape policy, and the script is the executable
reference.

`res.F[id]` is the spectrum per Fourier mode after feed-down and `res.Fprim[id]`
the primary part alone, both on `qs`, keyed by PDG id. Results contain only
stable particles by default; `stableparticles(dl)` lists their
`ParticleInfo` records in ID order, while `allparticles(dl)` lists every
defined species. Both accessors also accept a `CascadePlan`. Pass
`output = :all` to `runcascade` only when intermediate-resonance spectra are
needed for diagnostics. The `primary` callback is where the physics goes:
return `zeros(length(qs))` for a species to exclude it.

For production, pass `cache = "Wtables.jls"` so the packed matrices are built
once and reloaded. Serialized size and packed cache timings depend on the
selected topology, spectral policy, grid, and machine; use
`benchmarks/cascade_plan.jl` to measure the production configuration.

## Files

All package runtime sources live in `src/`. The current numerical suite remains
at the root behind the standard `test/runtests.jl` entry point. Canonical
runtime data lives in `data/`, while figure scripts plus curated PNG outputs
live in `figures/`. Benchmark notebooks keep transient outputs in ignored
benchmark-local paths.

| File | Role |
|------|------|
| `src/FastReso2.jl` | Package entry point; loads the component modules once and reexports the public API |
| `src/DecayModel.jl` | Shared resolved `DecayList`, filtering, validation, thresholds, and graph feed-down order |
| `src/DecaySpectra.jl`   | Core decay map: 2-/3-/4-body kernels; single- and fused multi-mode matrix assembly |
| `src/CascadeTables.jl`  | Persistent per-channel packed matrices, feeder-first cascade, and compiled flat plans |
| `src/RowBandMatrix.jl`  | Read-only, offset-backed row-band `AbstractMatrix` storage |
| `src/CascadeLinearAlgebra.jl` | Generic row-band `mul!` (including the mixed `Float32`/`Float64` cascade path) and short scatter kernels |
| `benchmarks/cascade_plan.jl` | Reproducible plan-compilation, reuse, and mode-threading benchmark |
| `benchmarks/fastreso_crosscheck/` | Cross-check against upstream [FastReso](https://github.com/amazeliauskas/FastReso) — see its own README |
| `src/LagrangeWeights.jl`| Lagrange weights — value, 1st/2nd derivative, combined (uniform + non-uniform grids) |
| `src/ParticleList.jl`   | `ParticleCatalog` metadata, unresolved `DecayNetwork` topology, and canonical reader mechanics |
| `src/readers/Canonical.jl` | The single split-file runtime reader used by `readparticles` and `readdecays` |
| `src/ThermalModel.jl`   | Statistical hadronization: freeze-out number densities, feed-down yields, and `ThermalPlan` for repeated evaluation |
| `quickstart.jl`     | The Quick start walkthrough above, as a runnable script |
| `test/runtests.jl`  | Standard `Pkg.test()` entry point; currently delegates to the root numerical suite |
| `runtests.jl`       | Legacy assertion-based numerical suite pending component `Test.@testset` migration |
| `cascade_plan.md`   | Design, implementation, and validation of compiled flat cascade execution |
| `spectral_function_plan.md` | The `SpectralShape` interface and mass-dependent widths — steps 2–3 implemented |
| `angular_momentum_plan.md` | Barrier factors and per-state ℓ: measurements, the parity data gap, and one open decision |
| `type_simplification_plan.md` | Where introducing a type would remove work, and where it would only move it |
| `model_labels_plan.md` | The common cause behind the parity, ℓ and quark-content gaps: labels the source model had and the schema dropped |
| `matrix_notes.tex`  | Method notes (compiled: `matrix_notes.pdf`) |
| `data/particles_QM2016Plus_isofix.dat` / `data/decays_QM2016Plus_isofix.dat` | Corrected canonical particle catalog and decay network used by the quickstart |
| `data/particles_charm_bottom.dat` / `data/decays_charm_bottom.dat` | Canonical charm+bottom supplement (150 states) |
| `tools/convert_decay_data.jl` | Converts AZYHYDRO or THERMINATOR input into the canonical split format |
| `tools/gen_heavy_flavour.py` | Regenerates and validates the charm+bottom list from the pinned PDG database |
| `figures/cascade.jl` | Full cascade driver (builds/loads the packed `Wtables.jls` cache) → `cascade.png` |
| `figures/physics_feeddown.jl` | Physics deliverable: spectra + v₂(pT) feed-down for π/K/p/Λ → `feeddown_physics.png` |
| `figures/physics_charm.jl` | Charm feed-down (D⁰/D⁺/D_s/Λc) from excited charm, SHM thermal primaries → `feeddown_charm.png` |
| `figures/spectral_shape.jl` | The spectral shape's effect on pT spectra, `Γ(m)` against constant `Γ` → `spectral_shape.png` |
| `figures/demo.jl`   | Worked example: pion feed-down from ρ→ππ and ω→πππ → `demo_feeddown.png` |
| `figures/demo_filter.jl` | Feed-down by decay topology (2 / 2+3 / 2+3+4-body) via filtering → `feeddown_topology.png`, `feeddown_spectra.png` |

With the project active, figure scripts resolve paths from their own location
(`@__DIR__`), so their data and output paths do not depend on the working
directory. Figures are written next to the script that makes them. Authored
derivation PDFs remain tracked at the root; disposable notebook plots, caches,
and historical quick-start artifacts are ignored.

## Usage

```bash
julia --project=. -e 'using Pkg; Pkg.test()'         # validate everything
julia --project=. -t auto figures/demo.jl              # minimal ρ→ππ / ω→πππ example + figure
julia --project=. -t auto figures/demo_filter.jl       # feed-down by decay topology + figures
julia --project=. -t auto figures/cascade.jl           # full cascade + figure
julia --project=. -t auto figures/physics_feeddown.jl  # spectra + v₂(pT) feed-down, π/K/p/Λ
julia --project=. -t auto figures/physics_charm.jl     # charm feed-down (needs the charm supplement)
julia --project=. -t auto figures/thermal_yields.jl    # thermal yields + feed-down, SHM style → thermal_yields.png
julia --project=. -t auto figures/spectral_shape.jl    # Γ(m) vs constant Γ in pT spectra → spectral_shape.png
CASCADE_BENCH_GRID=32 CASCADE_BENCH_SAMPLES=5 \
  julia --project=. -t 2 benchmarks/cascade_plan.jl   # reproduce the published plan benchmark
benchmarks/fastreso_crosscheck/run_small.sh        # cross-check against upstream FastReso
python3 -m unittest discover -s tests -p 'test_*.py' # offline generator/data invariants
```

The FastReso cross-check needs upstream built first, and adds two Julia
notebooks that compare against its **native** `*_Fj.out` output with no custom
C++ in the path — one on the small fixtures, one on a trimmed
`data/azyhydro/decays_QM2016Plus_massorder.dat`. See
[`benchmarks/fastreso_crosscheck/README.md`](benchmarks/fastreso_crosscheck/README.md).

`-t auto` (threads) matters for the table build, which parallelizes over
channels and fuses all Fourier modes into a single quadrature sweep. It is
also required for `runcascade(plan; parallel = :modes)` to execute modes
concurrently. Direct calls to a decay kernel are serial by default; pass
`threaded = true` only when qT-node parallelism is worthwhile.

The `Wtables.jls` cache is regenerable and git-ignored. It serializes packed
`Float32` row bands in fixed tuples aligned with `table.modes`. Fixed mode
tuples, concrete width settings, and the spectral-shape setting use cache
format 12. The package-wide shared type identity and validated `ChannelKey`
layout make older serialized caches incompatible, so they are ignored and
rebuilt automatically.

Particle metadata and decay topology are independent input layers:

- `ParticleCatalog` owns `ParticleInfo` records, canonical names, aliases, and
  ID/name lookup.
- `DecayNetwork` owns parent-to-channel rows and may refer to particle IDs from
  a catalog loaded later.
- `DecayList` resolves the two. It is the strict, runnable model accepted by
  filtering, tables, plans, validation, and cascade execution.
- `DecayModel` owns that resolution plus topology filtering and deterministic
  graph ordering, shared by both cascade and thermal consumers.

See [`data/README.md`](data/README.md) for layer composition, provenance,
legacy formats, charge-conjugation repair, and regeneration commands.

The runtime accepts one canonical split format and does not guess the source
convention:

```text
# particles
id name mass width degeneracy B S C Bot I Q [aliases...]

# decays
parent n_daughters branching_fraction daughters...
```

Aliases are optional whitespace-free labels. The daughter tail must contain
exactly `n_daughters` integer IDs. Particle metadata and decay topology load
independently and are resolved only when constructing the model:

```julia
catalog = readparticles("particles.dat")
network = readdecays("decays.dat")
dl = DecayList(catalog, network)
```

The former `DecayList(path)` / `read_decaylist(path)` combined-file API has
been removed. Convert legacy inputs once, then use the explicit split readers
above; this keeps format guessing and source-specific rules out of runtime
code.

Supplements remain independent until resolution. This lets a decay network
refer to daughters supplied by another catalog:

```julia
catalog = merge(
    readparticles("data/particles_QM2016Plus_isofix.dat"),
    readparticles("data/particles_charm_bottom.dat"),
)
network = merge(
    readdecays("data/decays_QM2016Plus_isofix.dat"),
    readdecays("data/decays_charm_bottom.dat"),
)
dl = DecayList(catalog, network)
```

Format-specific parsing is an offline conversion step under `tools/`, so new
source formats can be added without expanding the runtime API. Convert one or
more AZYHYDRO combined files with:

```bash
julia --project=. tools/convert_decay_data.jl azyhydro INPUT... \
  --particles-out particles.dat --decays-out decays.dat
```

Convert THERMINATOR particle and name-based decay files with an optional
Clebsch–Gordan table:

```bash
julia --project=. tools/convert_decay_data.jl therminator \
  particles.data decays.data --cgc CGC.dat \
  --particles-out particles.dat --decays-out decays.dat
```

The converter resolves names, derives quantum numbers, applies requested CG
coefficients, preserves source aliases in the canonical particle rows, and
refuses to replace an existing output unless `--force` is supplied. Its output
is then loaded through the same `readparticles` and `readdecays` functions as
every other model.

Raw programmatic data can also use `DecayList(catalog, network)`. Resolution
accepts only a sole `parent → parent` terminal row or runnable 2-/3-/4-body
channels; unsupported multiplicities and unresolved particle references fail
before table construction. Branching fractions must be finite but are neither
clamped nor renormalized; `validate(dl)` reports values outside the physical
`[0, 1]` range and incomplete parent sums.

`ParticleInfo.name` is the canonical display name. Catalog merges retain a
different spelling for the same ID as an alias. `particle(catalog, id)` and
`particle(catalog, name)` retrieve a unique particle; `particleids(catalog,
name)` returns every matching ID when a label is ambiguous. These lookups also
accept a resolved list directly, for example `particle(dl, 211)`. Name-based
channel filters match canonical `ParticleInfo.name` values; resolve an alias
explicitly with `fromid(particleids(dl, alias)...)` or
`toid(particleids(dl, alias)...)`. Existing
`dl.parts` and `dl.byparent` access remains available as a compatibility view
of `dl.catalog.byid` and `dl.network.byparent`. Catalog and network dictionaries
are shared as read-only model data; construct or merge a new value instead of
mutating them after resolution.

`CascadeTable(...; grid=...)` requires an `AbstractRange{Float64}` and trusts
that input. This keeps the grid compact and selects the uniform interpolation
kernel without validation scans. Standalone `DecayMatrix` objects still accept
non-uniform vectors.

`Pkg.test()` enters through `test/runtests.jl`. The current suite covers number
conservation (2-/3-/4-body), Fourier-mode
reconstruction, quadrature and Lagrange-weight convergence (incl. 1st/2nd
derivatives), matrix-vs-functor agreement, the cascade round-trip, and channel
filtering. Splitting it into structured `Test.@testset`s and execution tiers is
the next test-hygiene step.

## Channel filtering

A cascade run has three orthogonal knobs, which live in three different places:

| knob | question | where |
|------|----------|-------|
| **source** | which species emit primaries? | the `primary` callback (return `zeros(length(grid))` to exclude) |
| **topology** | which decays are allowed? | `filter(pred, dl)` before `CascadePlan` |
| **sink** | which daughters do I read out? | stable particles by default; `output = :all` for diagnostics |

Filtering acts on the `DecayList` before plan compilation. A filtered list
governs table enrollment when building a lean table and always governs the
topology compiled into `CascadePlan`; numerical execution consumes only that
plan. Filters are values of a composable `ChannelFilter` type (combine with
`&`, `|`, `!`):

```julia
t = CascadeTable(dl; cache = "Wtables.jls")              # full table, built once

two_body = filter(nbody(2), dl)
hadronic = filter(nbody(2, 3) & !todaughter("Gamma"), dl)
selected = filter(parentmass(0, 0.7) | fromparent("omega(782)"), dl)

two_body_plan = CascadePlan(t, two_body)                  # all plans reuse t
hadronic_plan = CascadePlan(t, hadronic)
selected_plan = CascadePlan(t, selected)

res = runcascade(two_body_plan; primary)
res = runcascade(hadronic_plan; primary)
res = runcascade(selected_plan; primary, parallel = :modes)

# This convenience wrapper compiles a temporary plan backed by t, then runs it.
res = runcascade(t, selected; primary)

tlean = CascadeTable(two_body; cache = "W2.jls")         # or build only 2-body matrices
leanplan = CascadePlan(tlean, two_body)
res = runcascade(leanplan; primary)

# One-shot alternative: build/load, compile, and run in one call
res = runcascade(selected; primary, grid = t.grid, cache = "Wselected.jls")

# Fast sharp-parent approximation: omit 4-body channels and width averaging
fast = filter(nbody(2, 3), dl)
res = runcascade(fast; primary, width_cut = Inf)
```

Ready-made filters (all return a `ChannelFilter`):

```
nbody(ns...)        parentmass(lo,hi)     daughtermass(lo,hi)   brabove(x)
fromparent(names…)  todaughter(names…)    fromid(ids…)          toid(ids…)
baryonicparent()    mesonicparent()       strangeparent()       charmparent()
parentstrangeness(ss…)  parentisospin(is…)  channelfilter(f)    # raw (p,ch,ds)->Bool
```

Two properties this design buys:

- **Build once, filter freely** — each filtered plan reuses the full table's
  matrices; compiling a restricted topology does not rebuild them.
- **Or build lean** — the same filter at construction builds only the matrices
  you will use (give lean tables their own cache path, since a filtered list
  keeps the parent list's hash).

Parents wider than `width_cut` (GeV) use spectral averaging. Its default is
`1.0e-3`, `Inf` disables width averaging, and `0.0` uses every positive width.
`width_window = 4.0` sets the finite integration range to at most
`mass ± width_window * width` (the lower mass is floored at `0.05 GeV`).
Both are `Float64` table keywords, concrete fields available as
`table.width_cut` and `table.width_window`, and cache settings. Their physical
ranges are caller responsibility, and there are no module-level width
constants.
`spectral = :massdep` is the default and vanishes at the resolved species
threshold; `spectral = :constant` selects the constant-width Breit–Wigner.
Species thresholds are resolved before channel filtering, so changing topology
does not change a primary thermal density or prevent a filtered plan from
reusing a full table.
Disabling four-body channels or widths does not renormalize the retained
branching fractions.

## Thermal yields

Statistical hadronization in the Andronic–Stachel sense: number densities at
chemical freeze-out, multiplied by a volume and fed down through the decay
network. This path does not touch `CascadeTable` — no grid and no matrices,
just one density integral per species and one graph-ordered pass over the
parents.

```julia
using FastReso2

dl = DecayList(readparticles("data/particles_QM2016Plus_isofix.dat"),
               readdecays("data/decays_QM2016Plus_isofix.dat"))

y = thermalyields(dl, 0.1565; muB = 0.0, V = 5300.0)     # T [GeV], V [fm³]
y.primary[211], y.feeddown[211], y.total[211]            # π⁺ 249.5, 569.7, 819.2
```

Quantum statistics (`quantum`) and mass averaging (`width`) are both on by
default and are independent keywords; `shape` picks the spectral function the
average runs over (below). `thermaldensity` and `primaryyields` are exported
for the pieces.

For repeated evaluation — a fit scanning `(T, μ, V)` — compile a plan. It
hoists everything that does not depend on the thermodynamic parameters: the
momentum quadrature, a spectral quadrature rule per broad resonance, and the
feed-down order with its channels flattened.

```julia
plan = ThermalPlan(dl)                                 # ~10 ms, once
thermalyields(plan, 0.1565; muB = 0.0, V = 5300.0)     # ~1 ms, ~40× the list path
```

The plan agrees with the direct path to `3.6e-8`, not to machine precision: it
averages each resonance on a fixed 32-node rule where the direct path drives an
adaptive integral at `rtol = 1e-6`, which is where the agreement saturates.

### What each modelling choice is worth

At `T = 156.5 MeV, μ_B = 0, V = 5300 fm³`, each row measured against the row
above it. `figures/thermal_yields.jl` prints all of it and plots the last four.

| | π⁺ | K⁺ | p | Λ | Ξ⁻ | Ω⁻ |
|---|---:|---:|---:|---:|---:|---:|
| quark-model states over PDG-numbered | +14.5% | +11.6% | +29.1% | +50.5% | +75.9% | +36.4% |
| finite widths over pole mass | +5.1% | +4.9% | +14.4% | +4.1% | +0.1% | +0.1% |
| shape: `Γ(m)` vs constant `Γ` | −1.4% | −1.8% | −7.5% | −2.7% | −0.3% | −0.02% |
| `f0(500)` | +3.3% | 0 | 0 | 0 | 0 | 0 |
| isospin fix (`isofix`) | −2.6% | 0 | 0 | 0 | 0 | 0 |

Finite widths are the largest correction and the least uniform, because a mass
average rewards the low-mass side where the Boltzmann factor is cheaper: +14%
for the proton, fed by the broad Δ(1232) and the N\* tower, against +0.1% for
the Ω⁻, whose few feeders are all narrow. That alone moves `p/π` from 0.0678 to
0.0738, so switching widths off is a different model rather than a small
perturbation.

**Which spectral function carries the width is a separate physics choice**, and
`shape` selects it. `:massdep` (default) gives each resonance a
threshold-vanishing width `Γ(m) ∝ (mR/m)·q(m)/q(mR)`, so it carries no weight
below the mass of its own decay products; `:constant` freezes `Γ(m) = Γ`. At
ℓ = 0 the two do not order universally — truncation removes low-mass weight
while the tail adds high-mass weight, and which wins depends on where the
threshold sits inside the window. Density relative to the pole mass:

| | ρ | σ | Δ(1232) | K\* |
|---|---:|---:|---:|---:|
| `:massdep` | 1.044 | **0.551** | 0.902 | 1.004 |
| `:constant` | 1.043 | **0.457** | 0.952 | 1.017 |

The σ is where they part, and it is the state the threshold treatment exists
for: a constant-width shape puts 2.56% of its weight below `2m_π`, which is
mass the state cannot have. Both shapes are available in the cascade too, as
the `spectral` table option — both paths consume the same package-level
spectral-shape types, and the suite asserts their behavior pointwise.

`isofix` and the σ are the same kind of change from opposite directions. The
three entries `isofix` patches (η, a₁(1260), K(1630)) feed pions and nothing
else, which is why both rows are zero to the digit everywhere but π⁺.

Two caveats before comparing with data. Feed-down supplies **70%** of the pions
on this list (71% with the σ), so the answer is dominated by branching ratios
that are curated
rather than measured — 1167 of the 1518 light species are quark-model states,
and `pdg_audit.md` records that only η and ω regenerate from PDG. And more
states means more feed-down, so this list sits above the published SHM ratios;
the PDG-numbered layer is reported alongside to size that effect. Agreement at
the 10% level is the realistic target.

## License

The code in this repository is released under the [MIT License](LICENSE).

The bundled physics **data** is not covered by that license and carries its own
provenance:

- `data/azyhydro/decays_QM2016Plus_massorder.dat` — a thermal-model decay list produced
  elsewhere; cite and reuse it under the terms of its original source.
- `data/azyhydro/decays_charm_bottom.dat` — generated by `tools/gen_heavy_flavour.py` from the
  [Particle Data Group](https://pdg.lbl.gov) database, whose data is licensed
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite the PDG
  *Review of Particle Physics* if you use it. The generated header records the
  exact database release, schema version, license, and citation.

The corresponding files under `data/` are deterministic canonical conversions
of these sources and retain the source path and SHA-256 in their headers. They
carry the same data provenance and licensing.

### Regenerating the charm/bottom supplement

The generator is pinned to the PDG edition used by the checked-in artifact:

```bash
python3 -m pip install -r tools/requirements-heavy-flavour.txt
python3 tools/gen_heavy_flavour.py --check
python3 tools/gen_heavy_flavour.py --force  # regenerate after reviewing drift
julia --project=. tools/convert_decay_data.jl azyhydro \
  data/azyhydro/decays_charm_bottom.dat \
  --particles-out data/particles_charm_bottom.dat \
  --decays-out data/decays_charm_bottom.dat --force
```

Generation is validated in memory and replacement is atomic. Missing required
properties abort generation, except for the documented neutral `Σ_b` mass and
width estimates obtained from their charged isospin partners. PDG upper limits
and modes without numerical branching fractions are never converted into
probabilities; tables containing only relative ratios are also omitted under
both policies. Retained absolute branching fractions are not renormalized by
default, so unsupported probability produces no modeled feed-down.
`--branching-policy conditional` provides an explicit normalized alternative
for studies that intentionally condition on the retained absolute modes.
With the default absolute policy, `validate` reports parents with `ΣBR < 1` as
normalization outliers; these quantify incomplete modeled coverage rather than
an invalid individual branching fraction.
