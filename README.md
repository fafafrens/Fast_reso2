# Fast_reso2

Resonance **feed-down** for particle spectra in heavy-ion collisions: given a
thermal source of primary resonances, propagate every decay (2-, 3-, and
4-body) through the full mass-ordered cascade to obtain the final-state
spectra, including azimuthal (elliptic-flow) modulation.

The decay map is represented as a **matrix** in transverse momentum and
azimuthal Fourier mode, built once per channel and reused across the cascade.
Derivations — kernels, the Toeplitz structure of the map, and the four-body /
finite-width extensions — are in [`matrix_notes.tex`](matrix_notes.tex).

## Files

| File | Role |
|------|------|
| `DecaySpectra.jl`   | Core decay map: 2-/3-/4-body kernels, spectra as functors |
| `CascadeTables.jl`  | Persistent per-channel decay matrices + mass-descending cascade |
| `LagrangeWeights.jl`| Lagrange weights — value, 1st/2nd derivative, combined (uniform + non-uniform grids) |
| `ParticleList.jl`   | Particle / decay-list parsing |
| `cascade.jl`        | Full cascade driver (builds/loads `Wtables.jls`, ~2.5 GB cache) |
| `cascade_simple.jl` | Minimal single-mode cascade — regression reference |
| `demo.jl`           | Worked example: pion feed-down from ρ→ππ and ω→πππ |
| `runtests.jl`       | Test suite: conservation, Fourier modes, quadrature & Lagrange convergence, tables |
| `decays_QM2016Plus_massorder.dat` | Decay list (mass-ordered) |

## Usage

```julia
julia runtests.jl        # validate everything
julia demo.jl            # minimal feed-down example
julia cascade.jl         # full cascade (first run builds the ~2.5 GB table cache; later runs ~2 s)
```

Only `Printf` and the standard library are required. The `Wtables.jls` cache is
regenerable and git-ignored; the first `cascade.jl` run creates it (~37 s) and
subsequent runs load it.

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
baryonicparent()    mesonicparent()       channelfilter(f)      # raw (p,ch,ds)->Bool
```

Two properties this design buys:

- **Build once, filter freely** — a filtered list's keys are a subset of the
  cached table, so restricting an *apply* never rebuilds.
- **Or build lean** — the same filter at construction builds only the matrices
  you will use (give lean tables their own cache path, since a filtered list
  keeps the parent list's hash).
