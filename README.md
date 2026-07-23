# Fast_reso2

Resonance **feed-down** for particle spectra in heavy-ion collisions: given a
thermal source of primary resonances, propagate every decay (2-, 3-, and
4-body) through the full mass-ordered cascade to obtain the final-state
spectra, including azimuthal (elliptic-flow) modulation.

The decay map is represented as a **matrix** in transverse momentum and
azimuthal Fourier mode, built once per channel and reused across the cascade.
The method — the two-body kernel and its Chebyshev–Gauss quadrature, the
effective-mass recursion behind 3-/4-body decays, the finite parent-width mass
average, and the (documented, not-yet-coded) rapidity-Toeplitz extension — is
derived in [`matrix_notes.tex`](matrix_notes.tex).

**Implemented:** 2-/3-/4-body decays, boost-invariant (η-independent) sources,
Fourier modes `n = 0, 2`, the full mass-ordered cascade, channel filtering, and
**finite parent widths** (broad resonances averaged over a Breit–Wigner spectral
function, with channel reopening — applied automatically from each species'
width in the decay list).
**Documented but not yet coded:** rapidity-dependent (η-resolved) sources, and
the cluster / observed-daughter width cases — see the note.

## Requirements

**Julia ≥ 1.11** — the table build uses `Threads.@threads :greedy`, added in 1.11.

Dependencies are declared in `Project.toml`; instantiate once, then run
everything with `--project`:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The core (`DecaySpectra.jl`, `CascadeTables.jl`) needs only
`FastGaussQuadrature` plus the standard library (`Serialization`, `SHA`,
`Dates`, `Printf`); the tests add `QuadGK`; the figure scripts add `Plots`.

## Files

Core modules, the test suite and the data live at the root; everything that
produces a figure (script **and** its output) lives in `figures/`.

| File | Role |
|------|------|
| `DecaySpectra.jl`   | Core decay map: 2-/3-/4-body kernels; single- and fused multi-mode matrix assembly |
| `CascadeTables.jl`  | Persistent per-channel decay matrices, channel filtering, mass-descending cascade |
| `LagrangeWeights.jl`| Lagrange weights — value, 1st/2nd derivative, combined (uniform + non-uniform grids) |
| `ParticleList.jl`   | Particle / decay-list parsing |
| `runtests.jl`       | Test suite (see below) |
| `matrix_notes.tex`  | Method notes (compiled: `matrix_notes.pdf`) |
| `decays_QM2016Plus_massorder.dat` | Decay list (mass-ordered) |
| `decays_charm_bottom.dat` | Charm+bottom supplement (150 states), generated from PDG — see `tools/` |
| `tools/gen_heavy_flavour.py` | Regenerates the charm+bottom list from the PDG database (`pip install pdg`) |
| `figures/cascade.jl` | Full cascade driver (builds/loads `Wtables.jls`, ~2.5 GB cache) → `cascade.png` |
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
```

`-t auto` (threads) matters for the table build, which parallelizes over
channels and fuses all Fourier modes into a single quadrature sweep. The
`Wtables.jls` cache is regenerable and git-ignored: the first `cascade.jl`
builds it (~10 s on 8 threads, incl. finite parent widths) and saves it (~2.5 GB
Float32); later runs load it (~7 s), and the cascade sweep itself takes ~2 s.

The test suite covers number conservation (2-/3-/4-body), Fourier-mode
reconstruction, quadrature and Lagrange-weight convergence (incl. 1st/2nd
derivatives), matrix-vs-functor agreement, the cascade round-trip, and channel
filtering.

## Channel filtering

A cascade run has three orthogonal knobs, which live in three different places:

| knob | question | where |
|------|----------|-------|
| **source** | which species emit primaries? | the `primary` callback (return `0` to exclude) |
| **topology** | which decays are allowed? | `filter(pred, dl)` |
| **sink** | which daughters do I read out? | index `res.F[id]` afterward — trivial |

Filtering acts on the `DecayList` ("before" the cascade), not inside
`runcascade`, so one filtered list governs **both** the table build and the
apply with no change to `runcascade`/`CascadeTable`. Filters are values of a
composable `ChannelFilter` type (combine with `&`, `|`, `!`):

```julia
t   = CascadeTable(dl; cache = "Wtables.jls")            # full table, built once
res = runcascade(t, filter(nbody(2), dl); primary)       # 2-body only — REUSES the full table
res = runcascade(t, filter(nbody(2,3) & !todaughter("gamma"), dl); primary)
res = runcascade(t, filter(parentmass(0, 1) | fromparent("omega"), dl); primary)

tlean = CascadeTable(filter(nbody(2), dl); cache = "W2.jls")  # or build only 2-body matrices
```

Ready-made filters (all return a `ChannelFilter`):

```
nbody(ns...)        parentmass(lo,hi)     daughtermass(lo,hi)   brabove(x)
fromparent(names…)  todaughter(names…)    fromid(ids…)          toid(ids…)
baryonicparent()    mesonicparent()       strangeparent()       charmparent()
parentstrangeness(ss…)  parentisospin(is…)  channelfilter(f)    # raw (p,ch,ds)->Bool
```

Two properties this design buys:

- **Build once, filter freely** — a filtered list's keys are a subset of the
  cached table, so restricting an *apply* never rebuilds.
- **Or build lean** — the same filter at construction builds only the matrices
  you will use (give lean tables their own cache path, since a filtered list
  keeps the parent list's hash).

## License

The code in this repository is released under the [MIT License](LICENSE).

The bundled physics **data** is not covered by that license and carries its own
provenance:

- `decays_QM2016Plus_massorder.dat` — a thermal-model decay list produced
  elsewhere; cite and reuse it under the terms of its original source.
- `decays_charm_bottom.dat` — generated by `tools/gen_heavy_flavour.py` from the
  [Particle Data Group](https://pdg.lbl.gov) database; cite the PDG *Review of
  Particle Physics* if you use it.
