# Fast_reso2

Resonance **feed-down** for particle spectra in heavy-ion collisions: given a
thermal source of primary resonances, propagate every decay (2- and 3-body)
through the full mass-ordered cascade to obtain the final-state spectra,
including azimuthal (elliptic-flow) modulation.

The decay map is represented as a **matrix** in transverse momentum and
azimuthal Fourier mode, built once per channel and reused across the cascade.
Derivations — kernels, the Toeplitz structure of the map, and the four-body /
finite-width extensions — are in [`matrix_notes.tex`](matrix_notes.tex).

## Files

| File | Role |
|------|------|
| `DecaySpectra.jl`   | Core decay map: 2-/3-body kernels, spectra as functors |
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
