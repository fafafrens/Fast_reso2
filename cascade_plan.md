# Unified Compiled and Partially Threaded Cascade Plan

Status: implementation and correctness validation complete. Deterministic
full-topology benchmarks are recorded below; resident production-cache and
one/two/four-thread scaling measurements remain pending. All public entry
points now converge on one executor: `runcascade(decaylist)` builds or loads a
table and compiles a plan, `runcascade(table, decaylist)` compiles a plan backed
by that table, and `runcascade(plan)` executes it.

## 1. Goal

The cascade table remains keyed by `ChannelKey` while it is built, cached,
filtered, and extended. Every execution operates on a compiled plan that
resolves the graph, particle identifiers, branching weights, and matrix
lookups. Callers construct that plan explicitly when they want to reuse the
same table and topology.

The intended API is

```julia
selected = filter(nbody(2, 3) & !todaughter("Gamma"), decaylist)
plan = CascadePlan(table, selected)
result = runcascade(plan; primary, parallel = :modes)
```

The result remains

```julia
(F = Dict(id => spectra), Fprim = Dict(id => primary_spectra))
```

and contains stable particles by default, so callers do not need to change
how they read final spectra. `output = :all` retains intermediate resonances
for diagnostics. Dictionaries are used at the API boundary, but not in the
numerical loop.

This plan has four goals:

1. compile topological sorting, channel enumeration, key construction, and
   dictionary lookup once per plan;
2. combine repeated applications of the same decay matrix;
3. allow safe partial multithreading;
4. preserve keyed table-storage semantics and numerical accuracy.

The table has since moved to packed row bands and fixed per-mode tuples without
changing the plan architecture.

## 2. Why a separate plan

`CascadeTable.W` is a good construction and storage representation:

```julia
Dict{ChannelKey,NTuple{N,M}} where {N,M<:RowBandMatrix{Float32}}
```

It supports filtered decay lists, incremental `ensure!`, and `prune!`.
Changing it into one positional array would make those operations more
fragile.

During a cascade, however, the topology and matrix choices do not change.
A direct executor over the keyed table would repeat work that can be compiled
once:

- enumerate the enrolled daughter slots and their dependency graph;
- topologically order the decaying parents;
- reconstruct `ChannelKey` values;
- look up `table.W[key]`;
- look up parent and daughter spectra by particle ID.

A `CascadePlan` is therefore an in-memory, topology-specific view of the
table.  It stores dense particle indices and direct references to the existing
matrices. It does not copy matrix payloads and is not serialized. The
decay-list and table convenience wrappers both compile this representation and
delegate to its executor rather than maintaining a second numerical loop.

## 3. Implemented representation

```julia
struct CascadeTarget
    daughter::Int
    weight::Float64
end

struct CascadeGroup{N,M<:RowBandMatrix{Float32}}
    matrices::NTuple{N,M}
    targets::UnitRange{Int}
end

struct CascadeStep
    parent::Int
    groups::UnitRange{Int}
end

struct CascadePlan{T<:CascadeTable,N,M<:RowBandMatrix{Float32}}
    table::T
    ids::Vector{Int}
    particles::Vector{ParticleInfo}
    stable_indices::Vector{Int}
    steps::Vector{CascadeStep}
    groups::Vector{CascadeGroup{N,M}}
    targets::Vector{CascadeTarget}
end
```

The fields have the following meanings:

- `ids[i]` is the public particle ID corresponding to dense index `i`;
- `particles[i]` is the matching `ParticleInfo`, used to initialize primaries;
- `stable_indices` selects the terminal particles returned by default;
- `steps` stores the feeder-before-fed order computed from the enrolled
  dependency graph during plan construction;
- one `CascadeGroup` represents one exact `(parent, ChannelKey)` pair;
- `matrices[m]` is the already-built decay matrix for mode `m`;
- each target contains a dense daughter index and its total branching weight;
- step and group ranges index contiguous flat `groups` and `targets` arrays,
  avoiding one separately allocated vector per step or group.

Keeping `table` in the plan explicitly owns the referenced matrices.  This is
also the correct lifetime model if table matrices later become mmap-backed
views.

## 4. Plan construction

Filtering happens before plan construction because the selected channels
define both the numerical operations and their dependency graph.

`CascadePlan(table, decaylist)` then performs the following work once:

1. collect particles into a stable dense order and build a temporary
   `particle_id => dense_index` dictionary;
2. enumerate the enrolled observed-daughter slots once, using the table's
   finite-width settings;
3. during that traversal, collect grouped operations and unique
   `parent => decaying daughter` dependency edges;
4. attach `table.W[key]` directly to each new parent/key group;
5. topologically order the completed parent blocks;
6. within each group, sum weights separately for every daughter.

The ordering is required by the one-pass cascade.  For

```text
A -> B -> C
```

`A` must update `B` before `B` decays, otherwise the contribution from `A`
never reaches `C`.  Dictionary/file order is not a physics contract, and pure
descending pole mass fails for a broad-tail reopened channel that feeds a
heavier daughter.  The existing reopened-channel test demonstrates that case.

Dependency construction and topological ordering belong only in plan
construction. `runcascade(plan)` walks the already ordered `steps` and never
rebuilds either.

The daughter weight is

```math
alpha[parent, key, daughter]
    = sum(branching ratio for every enrolled matching daughter slot).
```

Equivalently, construction executes

```julia
target_weight[daughter] += channel.br
```

once per enrolled daughter slot.  Thus a channel containing the same daughter
twice automatically contributes `2 * channel.br`.  Multiple equivalent
channels also add correctly.  No multiplicity field or change to
`DecayChannel` is required.

Grouping must never cross parent boundaries: equal decay matrices belonging
to two parents act on different parent spectra.  Parent steps must retain the
topological order.  Stable first-seen ordering of groups and targets should be
used for reproducibility.

A missing table entry naturally throws during construction at `table.W[key]`.
Matrix count is encoded by `NTuple{N,M}`. Cache restoration validates matrix
shape and packed storage before use. No key, grid, topology, or hash validation
belongs in the hot execution path.

The dependency edges should be deduplicated even when several slots or
channels connect the same parent and daughter. A dependency cycle has no exact
single-sweep order; construction warns once and uses the deterministic
mass-descending approximate fallback, with particle ID as the tie-break.

## 5. Serial execution

At the beginning of each run:

1. initialize every species needed by the numerical sweep and create public
   output dictionaries for stable particles;
2. create a dense vector whose entries reference the same spectrum vectors;
3. copy `Fprim`;
4. allocate run-local scratch storage.

For one Fourier mode, the numerical loop is conceptually

```julia
for step in plan.steps
    Fp = denseF[step.parent][mode_index]

    for group_index in step.groups
        group = plan.groups[group_index]
        W = group.matrices[mode_index]

        if length(group.targets) == 1
            target = plan.targets[first(group.targets)]
            Fd = denseF[target.daughter][mode_index]
            mul!(Fd, W, Fp, target.weight, 1.0)
        else
            mul!(scratch, W, Fp)

            for target_index in group.targets
                target = plan.targets[target_index]
                Fd = denseF[target.daughter][mode_index]
                _cascade_axpy!(target.weight, scratch, Fd)
            end
        end
    end
end
```

The singleton path uses allocation-free five-argument `mul!`. A multi-target
group computes `W * Fp` once with three-argument `mul!` and scatters with
`_cascade_axpy!`. `RowBandMatrix` supplies the mixed-precision standard
`LinearAlgebra.mul!` fast path; the private scatter retains an `axpy!`
fallback.

The parent spectrum is read-only during all of its outgoing groups. Enrollment
excludes `daughter == parent`, so the input cannot alias an output.

## 6. Operation-count census

For the bundled QM2016 decay list:

| representation | matrix applications per mode |
|---|---:|
| current enrolled daughter slots | 44,643 |
| identical `(parent, key, daughter)` transfers | 42,573 |
| grouped `(parent, key)` operations | 39,075 |

The complete grouping removes 5,568 matrix applications per mode, or 12.47%.
For modes `0` and `2`, the count falls from 89,286 to 78,150.

Of the 39,075 groups:

- 35,798 have one destination and use direct five-argument `mul!`;
- 3,129 have two destinations;
- 75 have three destinations;
- 73 have four destinations.

Only 3,277 groups therefore need scratch-and-scatter execution.  The reduction
in matrix applications does not guarantee the same percentage reduction in
wall time: the full cascade is memory-bandwidth sensitive, and the scatter
adds vector traffic.  It must be measured on resident production-sized
matrices.

## 7. Safe partial multithreading

### 7.1 Recommended first level: Fourier modes

Fourier modes are completely independent:

- mode `0` reads and writes only mode-`0` spectra;
- mode `2` reads and writes only mode-`2` spectra;
- each mode has its own matrix and scratch vector;
- there is no reduction between modes.

The whole cascade for each mode can therefore run on a separate task:

```julia
function apply_plan!(denseF, plan; parallel)
    if parallel === :modes && Threads.nthreads() > 1 &&
       length(plan.table.modes) > 1
        @sync for m in eachindex(plan.table.modes)
            Threads.@spawn begin
                scratch = Vector{Float64}(undef, length(plan.table.grid))
                apply_mode!(denseF, plan, m, scratch)
            end
        end
    else
        scratch = Vector{Float64}(undef, length(plan.table.grid))
        for m in eachindex(plan.table.modes)
            apply_mode!(denseF, plan, m, scratch)
        end
    end
end
```

The task boundary must be outside the parent/group loops.  Creating a task for
each of the 39,075 small matrix applications would cost more than it saves.

Primary spectra should be initialized before launching these tasks.  This
keeps arbitrary user-provided `primary` callbacks serial and avoids imposing a
new thread-safety requirement on them.

With the currently common modes `[0, 2]`, this provides at most two-way
parallelism.  The actual gain will be below the ideal factor of two if both
tasks saturate memory bandwidth.  It is nevertheless the safest first
threaded implementation because:

- no spectrum vector is written by two tasks;
- the arithmetic order inside each mode is unchanged;
- serial and threaded plan execution can be bitwise identical;
- no locks, atomics, or reduction buffers are needed.

### 7.2 Parallel independent cascade calls

When many primary parameter sets or events use the same plan, parallelizing
whole calls is even simpler:

```julia
Threads.@threads for i in eachindex(primary_models)
    results[i] = runcascade(
        plan;
        primary = primary_models[i],
        parallel = :serial,
    )
end
```

The plan is read-only.  Outputs and scratch vectors belong to each invocation,
so concurrent calls are safe.  Inner mode threading should normally be
disabled in this case to avoid nested oversubscription.

Scratch storage must never be owned and mutated by `CascadePlan`.  It must be
local to an invocation or task.

If repeated runs must also avoid output allocation, a later
`CascadeWorkspace(plan)` can own `F`, `Fprim`, dense spectrum references, and
scratch.  A workspace would belong to one invocation at a time; the immutable
plan could still be shared by many workspaces and tasks.

### 7.3 Why groups cannot simply be threaded

Groups belonging to one parent all read the same completed parent spectrum,
but two different groups can feed the same daughter:

```text
group A ----\
             +--> F[daughter]
group B ----/
```

Running both direct updates concurrently creates a data race in
`F[daughter]`.  Float64 atomics would add substantial element-level overhead
and would make accumulation order nondeterministic.  Locks around an entire
matrix application would mostly serialize the work.

Therefore the first implementation must not place `Threads.@threads` directly
around `step.groups`.

### 7.4 Optional later level: topological frontiers

More than two threads can be used by partitioning the decay graph into
topological frontiers.  Parents in one frontier have no dependency on each
other, so their matrix products may execute concurrently.  Multiple parents
can still feed the same daughter, however, so their outputs require one of:

1. per-task daughter accumulation buffers followed by a fixed-order reduction;
2. compile-time conflict-free batches whose destination sets do not overlap;
3. products written to temporary buffers followed by a serial scatter.

A deterministic per-task reduction is the most general option:

```text
completed spectra
       |
       v
parallel parent frontier
  |       |       |
local 0 local 1 local 2
  \       |       /
   fixed-order reduction
           |
           v
    next frontier
```

This adds barriers, buffer clearing, and spectrum-vector traffic.  Since the
current workload is dominated by streaming matrix values, it may or may not
beat the simpler mode-level implementation.  It should be attempted only
after profiling shows useful remaining compute parallelism.

The bundled full graph has 40 Kahn topological levels.  Their median width is
9 parents and the widest contains 465, but 31 of the 40 levels contain
shared-daughter conflicts.  In the first level alone, 137 daughter IDs receive
from more than one parent.  This confirms that a plain parallel loop over a
frontier is not race-free.

At the production dimensions `G = 134`, `S = 1518`, and two modes, one dense
Float64 delta spectrum costs about 3.1 MiB per logical lane.  Four lanes would
need about 12.4 MiB, small beside the matrix table, but clearing and reducing
those buffers introduces extra bandwidth.  Logical lane assignment and
reduction order must be fixed if reproducibility across runs is required.
Changing the lane count can still change roundoff relative to the serial
parent order.

Private buffers should be associated with explicit tasks/chunks, not indexed
by `Threads.threadid()`, because Julia tasks may migrate between threads.

### 7.5 Row-level matrix threading

Rows of one matrix-vector product are independent, but launching a threaded
row loop for every small GEMV would introduce tens of thousands of
synchronization points.  A row-parallel kernel would require a persistent
worker design or larger fused batches.  It is a later numerical-kernel
experiment, not part of the first `CascadePlan`.

## 8. Threading and BLAS policy

The `RowBandMatrix{Float32} * Vector{Float64}` operation is mixed precision
and does not use the same-type SGEMV/DGEMV fast path. Its `LinearAlgebra.mul!`
method walks each stored row band contiguously, converts values to `Float64`,
and accumulates with SIMD `muladd`. Nevertheless benchmarks should set

```julia
BLAS.set_num_threads(1)
```

so future storage changes cannot silently combine BLAS threading with Julia
mode/task threading.

The initial public choices should be explicit:

```julia
parallel = :serial
parallel = :modes
```

An `:auto` default should be considered only after benchmarks.  Explicit
selection prevents accidental nested threading when callers already
parallelize over events or parameter points.

The table and its matrix values must be treated as read-only while a plan is
executing.  Concurrent `ensure!`, `prune!`, or matrix mutation is outside the
thread-safe contract.

## 9. Lifetime and invalidation

A plan is a snapshot of:

- the selected/filtered decay topology;
- cascade order;
- branching ratios and multiplicities;
- table matrix references;
- grid and mode layout.

Compile a new plan after changing the decay list, branching ratios, filter, or
table.  Adding unrelated matrices to the same table does not invalidate an
existing plan.  Removing dictionary entries also cannot dangle existing Julia
matrix references, although a live plan keeps those matrices in memory.

Plans should not be stored in the existing serialized cache.  Recompilation is
small compared with loading the matrix payload, and an ephemeral plan avoids a
second cache format and pointer/offset validity problems.

## 10. Correctness tests

The table and decay-list wrappers must be checked against the explicit plan to
which they delegate:

```julia
plan = CascadePlan(table, decaylist)
runcascade(table, decaylist; primary) == runcascade(plan; primary)
```

Because these calls share one executor, wrapper equivalence should be bitwise.
Grouping itself needs an independent test-only ungrouped slot interpreter,
direct small-cascade calculations, or pinned expected results; a public wrapper
that compiles the same plan is not an independent numerical reference.

Required cases:

1. repeated daughters such as `rho -> pi + pi`, verifying weight `2 * BR`;
2. one `(parent, key)` feeding several distinct daughters;
3. multiple channels contributing to the same `(parent, key, daughter)`;
4. filtered decay lists backed by a full table;
5. broad-parent reopened channels and feeder-before-fed ordering;
6. one mode, reversed mode order, and modes `[0, 2]`;
7. two successive calls with different primary callbacks, proving no state is
   retained in the plan;
8. two concurrent calls using the same plan and different primaries;
9. serial versus `parallel = :modes`;
10. immediate construction failure when a required table key is missing.

Grouping changes the order of some floating-point scaling/addition operations.
Ungrouped-reference-versus-grouped comparison should therefore use

```julia
norm(Fplan - Fref, Inf) / max(norm(Fref, Inf), 1) <= 1e-12
```

rather than bitwise equality. `Fprim` should remain exact. Wrapper delegation
and serial versus mode-threaded execution of the same compiled plan should be
bitwise equal because each mode retains its serial operation order.

## 11. Benchmark plan

Compilation and execution must be measured separately:

1. plan compilation against an already resident table;
2. the table wrapper, which includes plan compilation and one execution;
3. serial execution of a reused plan;
4. reused-plan execution with `parallel = :modes`;
5. the full decay-list one-shot path, including table build/load where relevant;
6. parallel independent plan calls where relevant.

For every path report:

- median time and allocations after warm-up;
- slot, transfer, and group counts;
- Julia thread count and BLAS thread count;
- grid size and active modes;
- correctness outside the timed region.

Use four workloads:

1. a tiny synthetic cascade for API and correctness;
2. the complete topology with tiny deterministic matrices to isolate
   orchestration;
3. resident 20-parent and 50-parent production-grid subsets;
4. the real full cache only as an opt-in benchmark after a RAM preflight.

Thread scaling should be measured with one, two, and four Julia threads even
though mode-level execution can use only `min(number_of_modes, nthreads())`.
Interleave serial and threaded trials to reduce cache-state bias.

Initial acceptance criteria:

- wrapper results bitwise equal to their explicitly constructed plan;
- normalized error against an independent ungrouped or direct reference no
  larger than `1e-12`, with exact `Fprim`;
- no copied decay matrices in a plan; storage migrations use an explicit
  cache-format bump;
- no representative resident workload more than 3% slower;
- plan compilation reported separately from reused execution;
- a measurable mode-threading gain on two-mode workloads, reported without
  assuming ideal two-way scaling.

### Historical deterministic benchmark

Before the public paths were unified, `benchmarks/cascade_plan.jl` exercised
the complete bundled 1,518-particle topology with distinct resident matrices
and used the now-removed direct keyed executor as an independent baseline. On
the implementation host with Julia 1.12.6, two Julia threads, one BLAS thread,
`G = 16`, two modes, and three median samples, it recorded:

| historical path | median | allocations | speedup vs keyed |
|---|---:|---:|---:|
| removed keyed executor | 59.179 ms | 32.183 MiB | 1.00x |
| reused plan, serial | 23.121 ms | 2.056 MiB | 2.56x |
| reused plan, mode-threaded | 12.240 ms | 2.058 MiB | 4.83x |

Plan compilation was 72.559 ms and 77.812 MiB, reported separately because it
is amortized across repeated runs. The isolated compiled apply measured
21.213 ms serial and 10.640 ms mode-threaded, a 1.99x threading gain. The
normalized error against the former keyed reference was `1.443e-15`; serial
and mode-threaded outputs were bitwise equal. These historical measurements
include row-band matrix-vector applications with deterministic payloads while
excluding construction and cache I/O. They are not a `G = 134`
production-cache benchmark. Current measurements should label the table
wrapper as compile-plus-execute and the reused plan as execution-only.

### Fixed-tuple table storage benchmark

Changing each table value from `Vector{M}` to `NTuple{N,M}` removes the
plan-time tuple adapter and its cache. On the same full topology with Julia
1.12.6, two Julia threads, one BLAS thread, `G = 16`, two modes, and 50 median
samples:

| path | vector-backed | tuple-backed | vector allocations | tuple allocations |
|---|---:|---:|---:|---:|
| plan compilation | 32.596 ms | 29.081 ms | 77.892 MiB | 67.736 MiB |
| table wrapper: compile + run | 40.926 ms | 36.958 ms | 79.948 MiB | 69.793 MiB |
| reused plan, serial | 8.497 ms | 8.533 ms | 2.057 MiB | 2.057 MiB |
| reused plan, mode-threaded | 4.569 ms | 4.500 ms | 2.058 MiB | 2.058 MiB |

Plan compilation is about 11% faster with 13% fewer allocations; the
compile-plus-run wrapper is about 10% faster with 13% fewer allocations.
Execution of an already compiled plan is materially unchanged. All compared
outputs were bitwise equal.

The vector-backed variant no longer exists, so that column cannot be
re-measured; the table is kept as the record of the change. Its tuple-backed
allocations still reproduce exactly — see below.

### Where the compile-side numbers moved

The two compile-side rows above are stale. They moved at `cf1bd10` ("Convert
the tables to one canonical format"), and have been flat since. Measured on an
Apple M4 Max, Julia 1.12.6, two Julia threads, one BLAS thread, `G = 16`, two
modes, 50 median samples, as three interleaved A/B pairs:

| path | `cf1bd10~1` | current | change |
|---|---:|---:|---:|
| plan compilation | 33.712 ms / 67.768 MiB | 27.326 ms / 62.261 MiB | −19% / −8.1% |
| table wrapper: compile + run | 42.838 ms / 69.824 MiB | 37.181 ms / 64.317 MiB | −13% / −7.9% |
| reused plan, serial | 9.574 ms / 2.057 MiB | 9.723 ms / 2.057 MiB | +1.6% / — |
| reused plan, mode-threaded | 4.955 ms / 2.058 MiB | 5.050 ms / 2.058 MiB | +1.9% / — |

Execution allocations are byte-identical and the workload is unchanged:
`keys 18299; transfers 44643; groups 39075; targets 42573; multi-target 3277`
before and after. Execution *time* is consistently ~2% slower — every current
run fell outside the range of every `cf1bd10~1` run across three pairs — but
this is three samples per side on an effect small enough to be code layout, so
it is recorded rather than explained.

`cf1bd10` changed both the reader and 254 lines of `CascadeTables.jl`, so the
compile-side gain is not attributable to either alone. Two candidate mechanisms
were checked and ruled out: the `iszero(channel.br) && continue` added to
`eachchannelkey` skips nothing on this list, since the counts above are
identical; and the parent-ID reordering between the legacy and canonical files
is neutralized by `_cascadeorder`, which sorts keys before use.

Two notes on method, learned by getting them wrong first. Single invocations of
this benchmark vary by up to 15% on this host despite its own 50-sample median,
so A/B pairs must be interleaved rather than compared across separate sessions.
And allocations are the reliable cross-host quantity: the tuple-backed
`67.736 MiB` published above reproduced to the byte at `cf1bd10~1` on different
hardware, while none of the times did.

## 12. Implementation stages

### Stage 1: compiled serial plan — complete

- add dense indices, direct matrix references, and ordered parent steps;
- preserve the current slot operation order;
- establish the reference tests and benchmark harness.

### Stage 2: grouped serial plan — complete

- coalesce identical `(parent, key, daughter)` transfers;
- group all destinations of one `(parent, key)`;
- use direct `mul!` for singleton groups and
  `mul!`/`_cascade_axpy!` scratch/scatter otherwise.

### Stage 3: mode-level threading — complete

- move the task boundary outside the entire per-mode cascade;
- allocate one scratch vector per mode task;
- keep primary initialization serial;
- add serial/threaded determinism and concurrent-call tests.

### Stage 4: unified wrappers — complete

`runcascade(table, decaylist; primary)` constructs
`CascadePlan(table, decaylist)` and delegates to the plan executor.
`runcascade(decaylist; primary, table_options...)` first builds or loads the
table and then follows the same route. There is no separate keyed numerical
executor. Repeated workloads construct and reuse `CascadePlan` explicitly.

### Stage 5: optional wider parallelism

Only if profiling justifies the complexity, prototype deterministic
topological-frontier reductions or a persistent row-parallel kernel.  Keep
these experiments separate from the initial plan so their memory and
rounding-order costs are independently measurable.
