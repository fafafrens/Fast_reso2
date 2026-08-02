# FastReso cross-check

This directory reproduces the thermal-spectrum comparison between this
repository and
[`amazeliauskas/FastReso`](https://github.com/amazeliauskas/FastReso).
The reference used while developing it was upstream commit
`a694de368e79e3d21afa5d57b47209d5c9c55ed9`.

It contains two complementary checks:

1. `verify_static.jl` compares the continuous two- and three-body decay
   functors with upstream `Feq2` in one static fluid cell.
2. `verify_projected.jl` compares the reusable, packed `CascadePlan` with an
   upstream `TKernel` projection. This is the path to use with a longer decay
   list.

`verify_static_chain.jl` retains the more expensive direct check of the real
two-step chain

```text
rho(1700)0 -> rho0 pi- pi+ -> pi- pi+
```

and `benchmark_chain_plan.jl` measures construction and repeated execution of
the same topology through `CascadePlan`.

## Build the upstream reference exporter

Build FastReso and this exporter with the same C++ compiler and standard
library. FastReso's CMake file adds `-fopenmp`; on macOS the simplest route is
Homebrew GCC and GSL:

```bash
git clone https://github.com/amazeliauskas/FastReso.git /path/to/FastReso
git -C /path/to/FastReso checkout a694de368e79e3d21afa5d57b47209d5c9c55ed9

cmake -S /path/to/FastReso -B /path/to/FastReso/build \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build /path/to/FastReso/build

make -C benchmarks/fastreso_crosscheck \
  FASTRESO_DIR=/path/to/FastReso \
  FASTRESO_BUILD=/path/to/FastReso/build \
  CXX=/opt/homebrew/bin/g++-15 \
  GSL_PREFIX=/opt/homebrew
```

On Linux, omit `GSL_PREFIX` when GSL is on the normal compiler search path.
Do not link a GCC-built FastReso archive with Apple Clang: the
`libstdc++`/`libc++` ABI mismatch produces unresolved symbols.

`CMAKE_POLICY_VERSION_MINIMUM` is required with CMake 4: upstream's
`cmake_minimum_required` predates 3.5, which newer CMake refuses outright.

## Notebooks

Two Julia notebooks compare against upstream's **native** `*_Fj.out` output
rather than the exporter in this directory, so no custom C++ sits in the path.
They are visual benchmarks, not acceptance tests — `run_small.sh` remains the
regression check.

| notebook | input | runtime |
|---|---|---|
| `compare_fjout_fixtures.ipynb` | the four fixtures in `fixtures/` | ~15 s |
| `compare_fjout_runcascade.ipynb` | `decays_QM2016Plus_massorder.dat`, trimmed by `MASS_CUT` | ~90 s at 1.4 GeV |

Both run `run_fastreso_azyhydro.exe`, convert `Feq 2` to a spectrum, project it
over rapidity, and compare against `CascadeTable` → `CascadePlan` →
`runcascade` on the same decay list.

They need a Julia Jupyter kernel bound to this project:

```julia
using Pkg; Pkg.add("IJulia")
using IJulia; installkernel("Julia FastReso", "--project=/path/to/Fast_reso2", "--threads=auto")
```

`QuadGK`, `Interpolations` and `PyPlot` are installed by the notebooks' first
cell into the shared default environment, deliberately **not** into this
repository's `Project.toml`, which describes what the library needs rather than
what a notebook finds convenient. `PyPlot` needs a Python with matplotlib and,
because the plot style sets `usetex`, a LaTeX installation.

### Why `*_Fj.out` cannot be compared to `runcascade` directly

`*_Fj.out` holds the spectrum in one **static** fluid cell at `ur = 0`, a
function of `|p|`. `runcascade` produces the **boost-invariant mode-0**
projection, which integrates over rapidity. Compared as-is they differ by
`2.95, 2.04, 1.33, 1.12` at `p = 0.05, 0.2, 0.5, 1.0 GeV` — drifting with
momentum, so not a normalization one can divide out. The notebooks bridge them
with

```text
F_proj(pT) = 4pi * int_0^etamax deta F_static(p(eta)),   E = mT cosh(eta)
```

where the `4pi` is `2pi` (source azimuth) times `2` (folding the symmetric
`|eta|` window onto `[0, etamax]`). Integrating the symmetric range as well
double-counts and produces a clean factor of `2.00` at every momentum.

### What limits the agreement

`*_Fj.out` is written with `%15.5e` — six significant digits — so these
notebooks cannot resolve better than about `1e-5`. That is the file format, not
either code: the exporter in this directory writes 17 digits precisely to avoid
the limit, and `RESULTS.md` reports `9.95e-8` for the two-body channel.

Traps worth knowing, all handled in the notebooks:

- **The filename is not authoritative.** `run_fastreso_azyhydro.cpp` hard-codes
  `pdg[] = {211, 321, 2212, 3122, 3312, 3334}` and writes a file even when the
  species is absent from the decay list, containing whatever `getParticleByPDG`
  returned. Trimming at 1.4 GeV excludes `Omega-`, and `PDGid_3334_*` then holds
  `Sigma(1385)-`. The mc-number in the header is checked against the filename.
- **The freeze-out temperature is hard-coded** at `0.145 GeV` upstream.
- **The uniform grid must extend well past the comparison window.** Stopping at
  3.5 GeV while comparing to 3.0 truncates the parents feeding high-`pT`
  daughters and inflates the feed-down error to `7e-2`.
- **Judge `total`, not feed-down.** `feed = total - primary` subtracts two close
  numbers. For `omega_only` the feed-down is `5e-4` of the total near 2.7 GeV
  and its relative error reaches `4e-3` while `total` stays at `2e-5`; the
  fixtures notebook prints the fraction alongside so this is visible.
- **A mass cut is self-closing**: daughters are always lighter than their
  parent, so no channel is dropped and no branching ratio needs repairing. What
  is lost is feed-down from heavier resonances — a stated truncation.

## Reproduce all small checks

After instantiating the Julia project, run:

```bash
benchmarks/fastreso_crosscheck/run_small.sh
```

Generated references are placed in `benchmarks/fastreso_crosscheck/output/`,
which is git-ignored. The script performs:

- isolated `rho0 -> pi+ pi-`;
- isolated `omega -> pi+ pi- pi0`;
- both channels together;
- the two-step `rho(1700)` chain;
- a 401-point projected `CascadePlan` comparison;
- the reusable plan benchmark.

Every numerical verifier exits nonzero when its acceptance test fails.

## Compare a longer decay list

The projected comparison is deliberately generic. First export one or more
massive target particles from upstream:

```bash
benchmarks/fastreso_crosscheck/bin/fastreso_reference projected \
  /path/to/mass_ordered_decays.dat \
  benchmarks/fastreso_crosscheck/output/long.projected.tsv \
  0.145 0.0001 5.0 201 \
  211 -211 321 -321 2212 -2212
```

Then run the same decay list locally:

```bash
FASTRESO_PROJECTED_NQ=64 \
FASTRESO_PROJECTED_NTHETA=16 \
FASTRESO_PROJECTED_NMU=16 \
FASTRESO_PROJECTED_CACHE=benchmarks/fastreso_crosscheck/output/long.jls \
julia --project=. --threads=auto \
  benchmarks/fastreso_crosscheck/verify_projected.jl \
  benchmarks/fastreso_crosscheck/output/long.projected.tsv \
  /path/to/mass_ordered_decays.dat
```

The target IDs only control which final spectra are projected and written.
Upstream still computes the complete cascade. The Julia verifier:

- calculates matching zero-chemical-potential thermal primaries for every
  particle;
- keeps only two- and three-body channels;
- sets `width_cut = Inf`, so every parent is sharp;
- builds the table, compiles the plan, and reports those times separately;
- compares primary, inclusive total, and feed-down spectra;
- benchmarks repeated public `runcascade` calls.

Start with the reduced orders above for a long list, then increase to
`128/32/32`. Increase the uniform grid only after checking convergence because
table construction scales strongly with its length.

The default comparison window is `0.1 <= pT <= 3 GeV`, excluding reference
values below `1e-10` of their peak. Useful overrides are:

```text
FASTRESO_PROJECTED_PMIN
FASTRESO_PROJECTED_PMAX
FASTRESO_PROJECTED_PEAK_FLOOR
FASTRESO_PROJECTED_RTOL
FASTRESO_PROJECTED_PEAK_ATOL
FASTRESO_PROJECTED_NETA
FASTRESO_PROJECTED_SAMPLES
FASTRESO_PROJECTED_NQ
FASTRESO_PROJECTED_NTHETA
FASTRESO_PROJECTED_NMU
FASTRESO_PROJECTED_NSIGMA
FASTRESO_PROJECTED_NMA
FASTRESO_PROJECTED_CACHE
```

## Spectrum conventions

The exporter writes 17-digit, versioned TSV files. Do not substitute
FastReso's normal `*_Fj.out` files:

- they contain regularized irreducible components rather than
  `E dN/d^3p`;
- their printed precision is only about six significant digits;
- their native momentum grid is nonuniform.

Each TSV also records a decay-file fingerprint, compiler identity, native
FastReso grid parameters, GSL tolerances, and optional upstream revision/patch
labels. Set the latter before exporting when keeping a reference:

```bash
export FASTRESO_UPSTREAM_REVISION="$(git -C /path/to/FastReso rev-parse HEAD)"
export FASTRESO_UPSTREAM_PATCHES="quiet_4body"
```

`verify_projected.jl` refuses to compare a reference generated from different
decay-list bytes.

For the static comparison, upstream stores

```text
Feq2(p) = p f2(E)
```

and the exporter writes

```text
F(p) = nu/(2pi)^3 * E/p * Feq2(p).
```

For the projected comparison, `TKernel` is evaluated at `ur=0`, component
index `0`, integrated over its fixed `|eta| <= 5` window, and divided by
`(2pi)^3`. The result is already the boost-invariant mode-zero parent spectrum
consumed by `CascadePlan`; no additional degeneracy or `E/p` factor belongs in
the Julia callback.

The projected grid must start close to zero but strictly above it. Starting at
`0.02 GeV` discarded low-momentum parent support and created a false
`5.6e-3` discrepancy; `0.0001 GeV` avoids that problem.

## Matching limitations

These restrictions are intentional because they reproduce upstream FastReso:

- widths in particle headers are ignored;
- four-body rows are skipped;
- the input must be ordered feeder-before-fed for upstream;
- chemical potentials are currently fixed to zero by this harness;
- observed massless daughters should not be selected because the local decay
  kernel does not support them;
- upstream yields are inclusive accumulated densities—parents are not removed
  after they feed daughters.

The optional `patches/fastreso_quiet_4body.patch` suppresses unconditional
four-body skip messages when upstream runs with `verbose=false`.

`patches/fastreso_high_precision.patch` changes the GSL tolerances and the
native rest-frame grid from 401 to 801 points. Because `fNpbar` is embedded in
the `TParticle` layout, apply that patch only with a fresh FastReso build
directory and rebuild this exporter against the same headers. Mixing a stale
archive and exporter is an ABI error.

Keep upstream stderr when running the exporter: GSL failures trigger looser
fallback tolerances and are reported only there.

## Timing interpretation

The exporter uses separate FastReso objects for primary and total spectra so
lazy spline initialization cannot contaminate the decay timer. Its TSV records
thermal setup, primary projection, decay, and final projection separately.

Do not quote `FastReso decay time / CascadePlan apply time` as a direct
algorithmic speedup. FastReso evolves its rest-frame irreducible functions,
whereas the local plan applies a prebuilt projected matrix. Report at least:

1. upstream thermal setup;
2. upstream decay;
3. upstream projection;
4. local thermal projection;
5. `CascadeTable` construction or cache load;
6. `CascadePlan` compilation;
7. repeated `runcascade` execution.

See [RESULTS.md](RESULTS.md) for the measurements obtained while creating this
benchmark.
