# Recorded cross-check results

These are development measurements, not universal performance claims.

## Provenance

- FastReso commit:
  `a694de368e79e3d21afa5d57b47209d5c9c55ed9`
- temperature: `0.145 GeV`
- chemical potentials: zero
- widths: disabled/ignored
- upstream native grid: 401 points
- stock upstream GSL tolerances: `epsrel=1e-6`, `epsabs=1e-10`
- local direct rule: `nq=256`, `ntheta=64`, `nmu=64`

## Static thermal spectra

| Check | Maximum relative difference |
|---|---:|
| Primary thermal mapping | `1.46e-13` |
| `rho0 -> pi+ pi-`, through 1.2 GeV | `9.95e-8` |
| `omega -> charged pion`, through 0.4 GeV | `1.69e-7` |
| `omega -> neutral pion`, through 0.4 GeV | `1.79e-7` |
| Three-body spectrum, through 1.2 GeV | `4.25e-6` |
| Combined charged-pion spectrum, through 1.2 GeV | `3.99e-7` |
| Integrated branching-yield identities | machine precision |

The larger three-body tail error is upstream numerical accuracy rather than
local quadrature:

| Upstream configuration | Error at 0.8 GeV | Error at 1.2 GeV |
|---|---:|---:|
| stock tolerances, 401 points | `2.51e-6` | `4.13e-6` |
| `1e-8/1e-16`, 401 points | `7.34e-9` | `6.20e-8` |
| `1e-8/1e-16`, 801 points | `4.56e-10` | `3.93e-9` |

Local `128/32/32 -> 256/64/64` changed the selected three-body values by at
most `1.51e-8`; `256/64/64 -> 512/128/128` changed them by at most `9.52e-10`.

## Two-step chain

For

```text
rho(1700)0 -> rho0 pi- pi+  (BR 0.15)
rho0       -> pi- pi+       (BR 1.0)
```

the integrated rho and pion yields closed exactly. With the deliberately
moderate nested-functor rule `48/12/12`, five final-pion points from
`0.05` to `0.8 GeV` had maximum relative difference `6.55e-6`.

## Projected matrix/plan path

For the isolated rho channel on a uniform 401-point grid from `0.0001` to
`5 GeV`, the packed `CascadePlan` feed-down agreed with upstream `TKernel` at
roughly:

```text
max relative difference: 1.1e-5
relative RMS:            3.5e-6
```

The dense `DecayMatrix` and packed plan agreed with each other; the remaining
difference came from interpolation/reference integration rather than the flat
plan.

On the development machine, the 401-point, three-key chain gave:

```text
CascadeTable + CascadePlan build: 0.28-0.32 s
reused runcascade median:         0.036-0.038 ms
```

Upstream timings must be kept separate by stage. Representative stock runs
were approximately `0.53 s` for the independent rho+omega rest-frame decay and
`0.27 s` for the rho(1700) chain; these are not directly equivalent to a
prebuilt matrix application.
