# Plan: where a type would simplify the repo

Status: **proposed, nothing implemented.** Written after auditing the repo's 50
underscore helpers, its trait-shaped predicates, and its `Symbol`-valued
options. Most of what looks like accumulated ceremony turned out to be
load-bearing; this records the few places where introducing a type would
genuinely remove work, and the places it would only move it.

The audit's method: count the sites, not the concepts. A type earns its place
when the same tuple of values is declared, passed, merged, or checked
field-by-field in several places — not when it merely names something.

## Two things already settled, so they are not revisited here

**Holy traits: no.** Every trait-shaped predicate in the repo is
instance-level. `_lowerendpointvanishes(md) = md.mmin ≤ md.mth` reads fields,
so two `MassDependentBW` values of the same type disagree on it and no
`f(::Type)` can answer it. The particle predicates (`isbaryon`, `ismeson`, …)
discriminate values of a single concrete type — a 351-species catalog contains
exactly one. Where a property *is* type-level the repo already dispatches
(`parentmassnodes(::BreitWigner)` over `(::SpectralShape)`, `_stencil` on
`AbstractRange` over `AbstractVector`, `Base.IndexStyle` on `RowBandMatrix`).

**`spectral::Symbol` → singleton type: no.** The branch runs once per key at
build time and once per species at plan time, never in a loop. A type parameter
would push `CascadeTable{G,N,M}` to `{G,N,M,S}` and propagate into
`CascadePlan{T,N,M}`, and `spectral` is serialized into the cache, where a
`Symbol` is trivial. The validation *is* duplicated four times, but three of
those guard different boundaries (constructor, deserialization which must
return `nothing` rather than throw, and a second module), so a shared validator
would need to cross the module firewall to remove one of them.

## Candidate A — `Provenance`, the strongest evidence

`path::String` and `sha::String` are declared as a pair in **three** structs
(`ParticleCatalog`, `DecayNetwork`, `DecayList`), each with a matching pair of
constructor arguments, and combined by **four** helpers: `_joinprovenance`,
`_joinhashes`, `_modelprovenance`, `_sourceidentity`. Merging a catalog joins
paths with `" + "` and hashes with `"+"` — two nearly-identical helpers whose
only difference is a separator.

```julia
struct Provenance
    path::String        # " + "-joined when several sources merge
    sha::String         # "+"-joined, same order
end
Base.merge(a::Provenance, rest::Provenance...)   # replaces both join helpers
```

Removes: two helpers outright, one pair of fields from three structs, and the
`same_source` special case in `_modelprovenance`.

**The cost, and it is the reason this is a candidate and not a decision.**
`.path` and `.sha` are read at **48 library and tool sites and 26 test sites**.
Keeping them working means `Base.getproperty` forwarding — which `DecayList`
already does for `.parts` and `.byparent`, so the pattern exists — or accepting
74 mechanical edits. Do it with forwarding, and the field pair disappears while
the accessor stays.

## Candidate B — `QuantumNumbers`, the best payoff per line

`validate` checks conservation with five near-identical blocks:

```julia
if sum(daughter -> daughter.charge, daughters)      != parent.charge      …
if sum(daughter -> daughter.baryon, daughters)      != parent.baryon      …
if sum(daughter -> daughter.strangeness, daughters) != parent.strangeness …
if sum(daughter -> daughter.charm, daughters)       != parent.charm       …
if sum(daughter -> daughter.bottomness, daughters)  != parent.bottomness  …
```

and `ThermalPlan` stores them as four parallel `Vector{Float64}` fields, read
together in one expression in the density loop. `chemicalpotential` sums four
of the five by hand — **and omits bottomness entirely**, which is defensible at
LHC energies but is invisible in the current form and would be obvious as a
missing term in a dot product.

```julia
struct QuantumNumbers          # additive, so conservation is one comparison
    baryon::Int8; charge::Int8; strangeness::Int8; charm::Int8; bottomness::Int8
end
Base.:+(a, b); Base.zero(::Type{QuantumNumbers}); Base.:(==)
quantumnumbers(p) = QuantumNumbers(p.baryon, p.charge, …)   # a *view*, not storage
```

Conservation becomes `sum(quantumnumbers, daughters) == quantumnumbers(parent)`,
and the chemical potential a dot product with a `QuantumNumbers` of potentials.

**Do it as an accessor, not as a field.** `ParticleInfo` must keep its flat
fields: `ThermalModel` duck-types records on `.mass`, `.width`, `.baryon`, …
precisely so it need not import `ParticleList` and create a second
`ParticleInfo` that live decay lists could not dispatch to. Nesting the numbers
would change that structural contract and break the firewall that `dbe97dc`
documented. A `quantumnumbers(p)` view costs nothing and leaves duck typing
intact.

## Candidate C — group `ChannelKey`'s shape fields

`ChannelKey` is at **8 fields**, up from 7 when `mth` landed in `daa9c3e`, and
`angular_momentum_plan.md` would add ℓ for a 9th. Three of them — `Γ`, `mth`,
and a future `ℓ` — exist for exactly one reason: they determine the parent's
spectral shape. The masses determine the kinematics.

```julia
struct ShapeKey          # what the spectral function is built from
    Γ::Float64
    mth::Float64
end
```

Worth doing **only alongside the ℓ decision, not before it.** The reason it is
listed here is that it makes that decision's option 2 natural: a shape registry
keyed by `ShapeKey` rather than a widening flat key is the difference between
"the key carries the shape's parameters" and "the key names a shape". Deciding
the fork first and grouping second avoids grouping for a layout that then
changes. Note this is a serialized field order, so it costs a `FORMAT` bump —
cheap to combine with the one ℓ needs anyway.

## Candidate D — not a type: name the conversion constant

`0.1973269804` appears as a bare literal in **two** functions plus a docstring,
converting fm³ to GeV⁻³. Not a units system — the repo has exactly one
conversion site and Unitful in the density loop would cost more than it
protects — just:

```julia
const ħc = 0.1973269804        # GeV fm
```

Full units were considered and rejected: the hot paths are bare `Float64`
arithmetic in `_densityatmass` and the row-band `mul!`, and the one place a
units error could hide is this single constant.

## Recommended order

1. **D**, immediately — one constant, no API surface, no risk.
2. **B** as an accessor, next — best payoff, no struct change, and it makes the
   `chemicalpotential` bottomness omission explicit rather than silent.
3. **A** with `getproperty` forwarding — real cleanup, but 74 read sites means
   it wants its own commit and a careful pass, not a drive-by.
4. **C** only when `angular_momentum_plan.md`'s fork is decided.

## Verification

```bash
julia --project -t auto runtests.jl
julia --project -t auto figures/thermal_yields.jl   # numbers must not move
```

Every item here is a refactor: **no observable may change.** The check for B is
that `validate`'s reported counts are identical on the production list, and for
A that the provenance strings recorded in table metadata are byte-identical.
For D, that yields agree to the last digit.
