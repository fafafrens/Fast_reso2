# Fast_reso2

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
filtering, and **finite parent widths** (broad resonances averaged over a
Breit–Wigner spectral function, with channel reopening — applied automatically
from each species' width in the decay list), plus a compiled, grouped
`CascadePlan` used by every cascade entry point, with optional mode-level
threading. Construct the plan explicitly when the same table and topology will
be run more than once.
**Documented but not yet coded:** rapidity-dependent (η-resolved) sources, and
the cluster / observed-daughter width cases — see the note.

## Requirements

**Julia ≥ 1.12** — the table build uses `Threads.@threads :greedy` (1.11) and
`OncePerTask` (1.12), which gives each of that loop's tasks its own row buffers.

Dependencies are declared in `Project.toml`; instantiate once, then run
everything with `--project`:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The core needs only `FastGaussQuadrature` plus the standard libraries
`LinearAlgebra`, `Serialization`, `SHA`, `Dates`, and `Printf`; the tests add
`QuadGK`, and the figure scripts add `Plots`.

## Quick start

Save this at the repository root as `quickstart.jl` and run
`julia --project -t auto quickstart.jl`. It goes from a single decay to the
full 18,299-key matrix cascade and takes about 25 s from a cold start.

```julia
## 1. One decay, on its own
include("DecaySpectra.jl"); using .DecaySpectra
using Printf

ρ = TwoBodyDecay(0.7752, 0.1396, 0.1396)          # ρ → π + π  (parent, observed, partner)
Fa(qT, η, ϕ) = exp(-hypot(0.7752, qT) * cosh(η) / 0.155)     # any parent spectrum
@printf("pion from ρ at pT = 0.5: %.4e\n", ρ(Fa, 0.5, 0.0, 0.0))

grid = range(0.0, 4.0, length = 134)              # parent and daughter share the grid
W = decaymatrices(ρ, [0, 2], grid, grid)          # one matrix per Fourier mode
@printf("W[n=0] is %s — daughter = W * parent\n", size(W[1]))

## 2. The whole cascade
include("CascadeTables.jl"); using .CascadeTables

dl = DecayList("decays_QM2016Plus_massorder.dat")
T  = 0.155
primary(p, n, qs) =                               # thermal source, 4% v₂ on mode 2
    (n == 0 ? 1.0 : 0.04) * p.degeneracy * [exp(-hypot(p.mass, q) / T) for q in qs]

qs    = range(0.0, 4.0, length = 61)
# One shot: build/load a table, compile a plan, then execute that plan.
res   = runcascade(dl; primary, grid = qs, parallel = :modes)

## 3. Read it out
π⁺ = 211
@printf("\n  pT    primary       +feed-down   ratio     v₂\n")
for i in (1, 16, 31)
    prim, tot = res.Fprim[π⁺][1][i], res.F[π⁺][1][i]
    @printf("%5.2f  %.4e   %.4e   %5.2f   %+.3f\n",
            qs[i], prim, tot, tot / prim, res.F[π⁺][2][i] / tot)
end
```

```
pion from ρ at pT = 0.5: 1.7614e-03
W[n=0] is (134, 134) — daughter = W * parent

  pT    primary       +feed-down   ratio     v₂
 0.00  4.0526e-01   9.4037e-01    2.32   +0.017
 1.00  1.4817e-03   3.0861e-03    2.08   +0.035
 2.00  2.4126e-06   4.7619e-06    1.97   +0.038
```

Roughly two of every three final pions come from a decay, and the 4% v₂ put
into the source is diluted to 1.7% at pT = 0 — soft feed-down pions are
emitted off-axis — recovering to 3.8% by 2 GeV.

`res.F[id]` is the spectrum per Fourier mode after feed-down and `res.Fprim[id]`
the primary part alone, both on `qs`, keyed by PDG id. Results contain only
stable particles by default; `stableparticles(dl)` lists their
`ParticleInfo` records in ID order, while `allparticles(dl)` lists every
defined species. Both accessors also accept a `CascadePlan`. Pass
`output = :all` to `runcascade` only when intermediate-resonance spectra are
needed for diagnostics. The `primary` callback is where the physics goes:
return `zero(qs)` for a species to exclude it.

For production, pass `cache = "Wtables.jls"` so the packed matrices are built
once and reloaded. At the default 134-point grid, dense `Float32` storage would
occupy 2.4481 GiB; the production row-band census estimates 1.1012 GiB of
packed matrix payload (44.98% of dense). The serialized file size and packed
cache build/load timings depend on the system and should be measured there.

## Files

Core modules, the test suite and the data live at the root; everything that
produces a figure (script **and** its output) lives in `figures/`.

| File | Role |
|------|------|
| `DecaySpectra.jl`   | Core decay map: 2-/3-/4-body kernels; single- and fused multi-mode matrix assembly |
| `CascadeTables.jl`  | Persistent per-channel packed matrices, filtering, feeder-first cascade, compiled flat plans |
| `RowBandMatrix.jl`  | Read-only, offset-backed row-band `AbstractMatrix` storage |
| `CascadeLinearAlgebra.jl` | Generic row-band `mul!` (including the mixed `Float32`/`Float64` cascade path) and short scatter kernels |
| `benchmarks/cascade_plan.jl` | Reproducible plan-compilation, reuse, and mode-threading benchmark |
| `benchmarks/fastreso_crosscheck/` | Cross-check against upstream [FastReso](https://github.com/amazeliauskas/FastReso) — see its own README |
| `LagrangeWeights.jl`| Lagrange weights — value, 1st/2nd derivative, combined (uniform + non-uniform grids) |
| `ParticleList.jl`   | Particle / decay-list parsing |
| `runtests.jl`       | Test suite (see below) |
| `cascade_plan.md`   | Design, implementation, and validation of compiled flat cascade execution |
| `matrix_notes.tex`  | Method notes (compiled: `matrix_notes.pdf`) |
| `decays_QM2016Plus_massorder.dat` | Decay list (mass-ordered) |
| `decays_charm_bottom.dat` | Charm+bottom supplement (150 states), generated from PDG — see `tools/` |
| `tools/gen_heavy_flavour.py` | Regenerates and validates the charm+bottom list from the pinned PDG database |
| `figures/cascade.jl` | Full cascade driver (builds/loads the packed `Wtables.jls` cache) → `cascade.png` |
| `figures/physics_feeddown.jl` | Physics deliverable: spectra + v₂(pT) feed-down for π/K/p/Λ → `feeddown_physics.png` |
| `figures/physics_charm.jl` | Charm feed-down (D⁰/D⁺/D_s/Λc) from excited charm, SHM thermal primaries → `feeddown_charm.png` |
| `figures/demo.jl`   | Worked example: pion feed-down from ρ→ππ and ω→πππ → `demo_feeddown.png` |
| `figures/demo_filter.jl` | Feed-down by decay topology (2 / 2+3 / 2+3+4-body) via filtering → `feeddown_topology.png`, `feeddown_spectra.png` |

The figure scripts resolve paths from their own location (`@__DIR__`), so they
run from anywhere; data is read from the root and figures are written next to
the script that makes them.

## Usage

```bash
julia --project -t auto runtests.jl                  # validate everything
julia --project -t auto figures/demo.jl              # minimal ρ→ππ / ω→πππ example + figure
julia --project -t auto figures/demo_filter.jl       # feed-down by decay topology + figures
julia --project -t auto figures/cascade.jl           # full cascade + figure
julia --project -t auto figures/physics_feeddown.jl  # spectra + v₂(pT) feed-down, π/K/p/Λ
julia --project -t auto figures/physics_charm.jl     # charm feed-down (needs the charm supplement)
CASCADE_BENCH_GRID=32 CASCADE_BENCH_SAMPLES=5 \
  julia --project -t 2 benchmarks/cascade_plan.jl   # reproduce the published plan benchmark
benchmarks/fastreso_crosscheck/run_small.sh        # cross-check against upstream FastReso
python -m unittest discover -s tests -p 'test_*.py' # offline generator/data invariants
```

The FastReso cross-check needs upstream built first, and adds two Julia
notebooks that compare against its **native** `*_Fj.out` output with no custom
C++ in the path — one on the small fixtures, one on a trimmed
`decays_QM2016Plus_massorder.dat`. See
[`benchmarks/fastreso_crosscheck/README.md`](benchmarks/fastreso_crosscheck/README.md).

`-t auto` (threads) matters for the table build, which parallelizes over
channels and fuses all Fourier modes into a single quadrature sweep. It is
also required for `runcascade(plan; parallel = :modes)` to execute modes
concurrently.

The `Wtables.jls` cache is regenerable and git-ignored. It serializes packed
`Float32` row bands in fixed tuples aligned with `table.modes`; at the default
grid their estimated matrix payload is 1.1012 GiB, versus 2.4481 GiB for dense
storage. The previous ~7 s load and ~2 s sweep measurements describe the
superseded dense cache. Fixed mode tuples and concrete width-setting fields use
cache format 10; older caches are ignored and rebuilt automatically.

Decay-list supplements can refer to particles from a base list and be composed
without writing a temporary combined file. Repeated particle records and
parent channel blocks are deduplicated when their stored definitions agree;
conflicting definitions are rejected rather than overwritten or concatenated:

```julia
base = DecayList("decays_QM2016Plus_massorder.dat")
heavy = DecayList("decays_charm_bottom.dat")
combined = merge(base, heavy)
```

`CascadeTable(...; grid=...)` requires an `AbstractRange{Float64}` and trusts
that input. This keeps the grid compact and selects the uniform interpolation
kernel without validation scans. Standalone `DecayMatrix` objects still accept
non-uniform vectors.

The test suite covers number conservation (2-/3-/4-body), Fourier-mode
reconstruction, quadrature and Lagrange-weight convergence (incl. 1st/2nd
derivatives), matrix-vs-functor agreement, the cascade round-trip, and channel
filtering.

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
selected = filter(parentmass(0, 1) | fromparent("omega"), dl)

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

Parents wider than `width_cut` (GeV) use Breit–Wigner averaging. Its default is
`1.0e-3`, `Inf` disables width averaging, and `0.0` uses every positive width.
`width_window = 4.0` sets the finite integration range to at most
`mass ± width_window * width` (the lower mass is floored at `0.05 GeV`).
Both are `Float64` table keywords, concrete fields available as
`table.width_cut` and `table.width_window`, and cache settings. Their physical
ranges are caller responsibility, and there are no module-level width
constants.
Disabling four-body channels or widths does not renormalize the retained
branching fractions.

## License

The code in this repository is released under the [MIT License](LICENSE).

The bundled physics **data** is not covered by that license and carries its own
provenance:

- `decays_QM2016Plus_massorder.dat` — a thermal-model decay list produced
  elsewhere; cite and reuse it under the terms of its original source.
- `decays_charm_bottom.dat` — generated by `tools/gen_heavy_flavour.py` from the
  [Particle Data Group](https://pdg.lbl.gov) database, whose data is licensed
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite the PDG
  *Review of Particle Physics* if you use it. The generated header records the
  exact database release, schema version, license, and citation.

### Regenerating the charm/bottom supplement

The generator is pinned to the PDG edition used by the checked-in artifact:

```bash
python -m pip install -r tools/requirements-heavy-flavour.txt
python tools/gen_heavy_flavour.py --check
python tools/gen_heavy_flavour.py --force  # regenerate after reviewing drift
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
