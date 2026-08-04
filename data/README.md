# Canonical particle and decay tables

Two files per table: a catalog (`particles_*.dat`) and a network
(`decays_*.dat`), both in FastReso's canonical format.

```text
particles: id name mass width degeneracy B S C Bot I Q [aliases...]
decays:    parent n_daughters branching_fraction daughters...
```

Every decay row carries its parent ID, so a channel may live in a different
file from its particle header. That is how one layer refers to species defined
by another.

## Layers

Disjoint and additive — no species or channel appears twice.

| layer | species | rows | |
|---|---:|---:|---|
| `light_pdg` | 351 | 1802 | hadrons with a PDG MC ID; **loadable alone** |
| `light_qm` | 1167 | 18736 | quark-model states, no PDG ID exists |
| `sigma` | 1 | 2 | f0(500) at `m = 0.475`, `Gamma = 0.55` |
| `charm` | 91 | 370 | open charm and charmonium |
| `bottom` | 59 | 90 | open bottom and bottomonium |

```julia
names = ["light_pdg", "light_qm", "charm"]
dl = DecayList(
    reduce(merge, [readparticles("data/particles_$n.dat") for n in names]),
    reduce(merge, [readdecays("data/decays_$n.dat")   for n in names]),
)
```

Stacks, all validating with zero conservation or branching-range violations:

| stack | species | channels | equals |
|---|---:|---:|---|
| `light_pdg` | 351 | 1778 | |
| `+ light_qm` | 1518 | 20514 | `QM2016Plus_isofix` |
| `+ sigma` | 1519 | 20516 | `QM2016Plus_isofix_sigma` |
| `+ charm + bottom` | 1669 | 20878 | everything |

`QM2016Plus`, `QM2016Plus_isofix`, `QM2016Plus_isofix_sigma` and
`charm_bottom` are the whole-file originals, kept so a stack can be diffed
against what it came from. `isofix` is the definition the layers are cut from.

## Legacy formats

The sources everything above was converted from, in the formats they arrived
in. Nothing at runtime reads these — `tools/converters/` parses them once.

| | |
|---|---|
| `azyhydro/` | the four AZYHYDRO tables: `QM2016Plus_massorder` and its `isofix`, `isofix_sigma`, and `charm_bottom` |
| `therminator/` | `OpenCharmParticleList_corrJS*.txt`, an open-charm list |

The THERMINATOR files are a **14-column variant**, not the 13-column layout
`tools/converters/THERMINATOR.jl` reads: they carry an extra `deg` between
`Gamma` and `Spin`. Do not drop that column to make them fit — `deg` and
`Spin` disagree on 46 of 87 rows and `deg` is the correct one, so converting
them as 13-column THERMINATOR would give every `D*` degeneracy 1 instead of 3.
`_correct_MC` replaces 26 placeholder `666` IDs with real ones; two of its
invented IDs (`43140`, `43141`) name states PDG does number, as
`104324`/`104314` and `104322`/`104312`.

## Three things to know

**`QM2016Plus` is upstream FastReso's list with two corrected daughter
counts** (`9040221` n=2→4, `20223` n=1→2). Without them it does not load here
at all — this reader rejects nonzero padding past the declared count.

**Only 351 of 1518 light species have a PDG ID.** The rest are quark-model
predictions; `pdg_audit.md` records that only `eta` and `omega` can be
regenerated from PDG on their own.

**The heavy layers keep absolute branching ratios** — omitted probability is
not redistributed, so quarkonia sum to well under one (`Upsilon(1S)` 0.0002,
`J/psi` 0.127). `validate` flags these; it is the documented policy, not an
error.

## Regenerating

```bash
julia --project=. tools/convert_decay_data.jl azyhydro SOURCE.dat \
    --particles-out data/particles_X.dat --decays-out data/decays_X.dat

python tools/split_by_provenance.py \
    data/particles_QM2016Plus_isofix.dat data/decays_QM2016Plus_isofix.dat --stem light
python tools/split_by_provenance.py \
    data/particles_charm_bottom.dat data/decays_charm_bottom.dat --stem heavy --split flavour
```

The splitter needs the `pdg` package; see its docstring for why membership is
asked of PDG rather than read off the ID digits.
