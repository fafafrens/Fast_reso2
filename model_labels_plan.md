# Plan: recover the labels the quark model knew

Status: **proposed, speculative, nothing implemented.** This is the common
cause behind three separate blockers already scoped elsewhere. It may well not
be worth doing — §"Is it worth it" argues both sides — but the pattern is worth
recording because each of those plans hits it independently and none of them
names it.

## The pattern

Three unrelated-looking gaps turn out to be one:

| gap | scoped in | blocked on |
|---|---|---|
| barrier factors, per-state ℓ | `angular_momentum_plan.md` | **parity** is stored nowhere |
| structural flavour selection | `type_simplification_plan.md` | **quark content** is not stored |
| quantum numbers as *derived* rather than declared | `type_simplification_plan.md` §B | same |

In every case the catalog stores the **derived** numbers and the source dropped
the **generating** labels. `QM2016Plus` is a quark-model list: its 1167
non-PDG states were produced by a model that necessarily knew each state's
quark content, orbital angular momentum L, spin S and parity. The `.dat` schema
kept `mass width degeneracy B S C Bot I Q` and none of that.

## What the audit established

**The stored numbers are sound, just not generative.** Gell-Mann–Nishijima,
`Q = I₃ + (B+S+C+Bot)/2`, holds across every layer — **1,668 species, zero
violations of `|I₃| ≤ I` and zero half-integer mismatches**. Nothing here is a
data-quality problem.

**Quark content really is the more primitive object.** Derived from MC ID
digits, it reproduces `(B, Q, S, C, Bot)` exactly for **all 500 PDG-numbered
species** — 274 baryons and 226 mesons across `light_pdg`, `charm`, `bottom` —
with zero mismatches and zero unresolvable quark/antiquark assignments.

**Including the case expected to break.** 80 of those are flavour-neutral qq̄
states (π⁰, η, η′, ω, φ, J/ψ, Υ) which are physically *superpositions*, so no
single integer content describes them. But every component carries the same net
numbers — uū and dd̄ both give `B = Q = S = C = 0` — so the derivation is blind
to the mixing exactly where it needs to be. The label is wrong about the state
and right about the quantum numbers.

**The 1167 are unrecoverable from what is checked in.** Three routes, all dead:

- *MC ID digits* — the invented IDs make digit-based inference return junk.
  The public helper that exposed that inference was removed during the Phase 1
  consolidation rather than offering confident false answers.
- *Names* — the convention is `Name(mass)<deg>(<charge>)`. Of 1167, **779
  parse, and the middle digit equals the stored degeneracy in all 779**. It
  encodes only what the columns already have.
- *The legacy AZYHYDRO file* — its 12th column is the channel count, not a
  model label.

So the information must come from outside the repo, exactly as parity must.

## The proposal

One supplementary table, alongside the existing layer files rather than a
schema change:

```text
data/species_model.dat
# id  content        L  S   P     provenance
  213 u,dbar         0  1  -1     PDG
  113 (uubar-ddbar)  0  1  -1     PDG      # representative; see mixing below
9000221 (uubar+ddbar) 1 1  +1     model
```

Design constraints, all inherited from decisions already taken:

- **Supplementary, not a column.** Column 12+ of the canonical particle format
  is aliases (`readers/Canonical.jl:26`), and no file uses them today — which
  is what makes that collision dangerous rather than safe. A separate file also
  leaves all 6,374 particle rows untouched.
- **Partial by construction.** Absence *is* the unknown sentinel; no `0`
  convention needed, and coverage can grow without a format bump.
- **Read-only and derived-from.** The catalog keeps its flat `B S C Bot I Q`
  fields. `ThermalModel` duck-types records on those, and nesting or replacing
  them re-creates the double-module trap `dbe97dc` documented. This table is
  consulted *beside* the catalog, never instead of it.

## What it unblocks

- **ℓ from J and P**, per `angular_momentum_plan.md` — worth −18.7% on the ρ,
  −11.4% on the Δ, and the largest unclaimed finite-width effect.
- **Flavour-sector selection becomes structural.** `charm(qc) = qc.c − qc.c̄` against
  `qc.c + qc.c̄ > 0` — two fields rather than a paragraph warning that
  `ischarm(J/ψ)` is false.
- **`QuantumNumbers` becomes generative** rather than a view: candidate B of
  `type_simplification_plan.md` upgrades from "group five fields" to "derive
  five fields, and make an inconsistent species unrepresentable".
- **Gell-Mann–Nishijima gets something to verify** rather than confirm — a
  cross-check between two independent sources instead of an identity among
  stored columns.

## Is it worth it

**Against.** Coverage is the whole problem. PDG supplies parity and content for
351 species; the other 1167 need either the program that generated
`QM2016Plus` or hand assignment, and `pdg_audit.md` already records that those
states cannot be regenerated from PDG. A table covering 23% of species, in a
repo whose defining feature is that 77% are quark-model predictions, is a
partial answer that must be permanently guarded with fallbacks.

**For.** The 351 are not a random 23%. They carry **93.0% of the feed-down to
π and p**, and the head of the broad-feeder list — ω, η, ρ, K*, f₀(980),
f₂(1270) — is entirely PDG with textbook J^P. The one significant non-PDG
entry, f₀(500), is ℓ = 0, which the fallback gives anyway.

**The cheaper alternative.** `angular_momentum_plan.md` already proposes
`data/parities.dat`: parity only, ~351 rows, one loader. That unblocks ℓ — the
item with measured physics behind it — without committing to quark content, L,
S, or a generative rewrite of the quantum numbers. **Do that first.** This
plan's full table is worth revisiting only if a second consumer appears, or if
the `QM2016Plus` generator is recovered and the 1167 stop being a permanent
hole.

## What it does not solve

- **Mixing.** π⁰/η/η′ and ω/φ are superpositions; a single integer content is a
  representative, correct for net quantum numbers and wrong as a state. Fine
  for everything here, not a basis for anything needing the wavefunction.
- **The per-channel vs per-species shape fork** — `angular_momentum_plan.md`
  §"The fork" is a structural decision about `ChannelKey` and matrix sharing,
  untouched by having better labels.
- **The tail cap** — `spectral_function_plan.md` step 1, independent.
- **The 1167 themselves** — no amount of schema design conjures labels that
  were never written down.

## Verification, if built

```bash
julia --project -t auto runtests.jl
julia --project -t auto figures/thermal_yields.jl    # must not move
```

The table is inert until a consumer reads it, so the first commit must change
no observable. The check that it is *right* rather than merely present: for
every species it covers, the derived `(B, Q, S, C, Bot)` must equal the stored
columns exactly — the measurement above says that is achievable for all 500
PDG-numbered species, so anything less is a defect in the table, not in the
data.
