# Compiled and Partially Threaded Cascade Plan

Status: design proposal; not yet implemented.

## 1. Goal

The cascade table should remain keyed by `ChannelKey` while it is built,
cached, filtered, and extended.  Repeated execution should use a separate
compiled plan that resolves the graph, particle identifiers, branching
weights, and matrix lookups once.

The intended API is

```julia
selected = filter(nbody(2, 3) & !todaughter("Gamma"), decaylist)
plan = CascadePlan(table, selected)
result = runcascade(plan; primary, parallel = :modes)
```

The existing result remains

```julia
(F = Dict(id => spectra), Fprim = Dict(id => primary_spectra))
```

so callers do not need to change how they read spectra.  Dictionaries are
used at the API boundary, but not in the numerical loop.

This plan has four goals:

1. remove repeated topological sorting, channel enumeration, key construction,
   and dictionary lookup from repeated cascades;
2. combine repeated applications of the same decay matrix;
3. allow safe partial multithreading;
4. preserve the current table/cache representation and numerical accuracy.

It does not initially change the decay-matrix representation, cache format, or
physics kernels.

## 2. Why a separate plan

`CascadeTable.W` is a good construction and storage representation:

```julia
Dict{ChannelKey, Vector{Matrix{Float32}}}
```

It supports filtered decay lists, incremental `ensure!`, and `prune!`.
Changing it into one positional array would make those operations more
fragile.

During a cascade, however, the topology and matrix choices do not change.
The current executor repeats work that can be compiled once:

- compute `cascadeorder(decaylist)`;
- walk `eachchannelkey`;
- reconstruct `ChannelKey` values;
- look up `table.W[key]`;
- look up parent and daughter spectra by particle ID.

A `CascadePlan` is therefore an in-memory, topology-specific view of the
table.  It stores dense particle indices and direct references to the existing
matrices.  It does not copy matrix payloads and is not serialized.

## 3. Proposed representation

```julia
struct CascadeTarget
    daughter::Int
    weight::Float64
end

struct CascadeGroup
    matrices::Vector{Matrix{Float32}}
    targets::Vector{CascadeTarget}
end

struct CascadeStep
    parent::Int
    groups::Vector{CascadeGroup}
end

struct CascadePlan{T<:CascadeTable}
    table::T
    ids::Vector{Int}
    particles::Vector{ParticleInfo}
    steps::Vector{CascadeStep}
end
```

The fields have the following meanings:

- `ids[i]` is the public particle ID corresponding to dense index `i`;
- `particles[i]` is the matching `ParticleInfo`, used to initialize primaries;
- `steps` follows the exact feeder-before-fed order from `cascadeorder`;
- one `CascadeGroup` represents one exact `(parent, ChannelKey)` pair;
- `matrices[m]` is the already-built decay matrix for mode `m`;
- each target contains a dense daughter index and its total branching weight.

Keeping `table` in the plan explicitly owns the referenced matrices.  This is
also the correct lifetime model if table matrices later become mmap-backed
views.

## 4. Plan construction

Filtering happens before plan construction because the selected channels
define both the numerical operations and their dependency graph.

`CascadePlan(table, decaylist)` then performs the following work once:

1. collect particles into a stable dense order and build a temporary
   `particle_id => dense_index` dictionary;
2. traverse the enrolled slot stream with `eachchannelkey` once;
3. during that traversal, collect grouped operations and unique
   `parent => decaying daughter` dependency edges;
4. resolve `table.W[key]` once for each encountered key;
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

This dependency work belongs only in plan construction.  Repeated
`runcascade(plan)` calls walk the already ordered `steps` and never call
`cascadeorder`.

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

A missing table entry naturally throws during construction at
`table.W[key]`; no separate validation layer is needed.  No grid, topology, or
hash validation belongs in the hot execution path.

The dependency edges should be deduplicated even when several slots or
channels connect the same parent and daughter.  A dependency cycle has no
exact single-sweep order; plan construction should report it rather than hide
it inside repeated execution.

## 5. Serial execution

At the beginning of each run:

1. create the normal output dictionaries;
2. create a dense vector whose entries reference the same spectrum vectors;
3. copy `Fprim`;
4. allocate run-local scratch storage.

For one Fourier mode, the numerical loop is conceptually

```julia
for step in plan.steps
    Fp = denseF[step.parent][mode_index]

    for group in step.groups
        W = group.matrices[mode_index]

        if length(group.targets) == 1
            target = only(group.targets)
            Fd = denseF[target.daughter][mode_index]
            mul!(Fd, W, Fp, target.weight, 1.0)
        else
            mul!(scratch, W, Fp)

            for target in group.targets
                Fd = denseF[target.daughter][mode_index]
                axpy!(target.weight, scratch, Fd)
            end
        end
    end
end
```

The singleton path retains the current allocation-free, five-argument
`mul!`.  A multi-target group computes `W * Fp` once and scatters the result
with `axpy!`.

The parent spectrum is read-only during all of its outgoing groups.
`eachchannelkey` already excludes `daughter == parent`, so the input cannot
alias an output.

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

- 35,798 have one destination and use direct `mul!`;
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
                scratch = zeros(Float64, length(plan.table.grid))
                apply_mode!(denseF, plan, m, scratch)
            end
        end
    else
        scratch = zeros(Float64, length(plan.table.grid))
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

The current `Matrix{Float32} * Vector{Float64}` operation is mixed precision
and does not use the same-type SGEMV/DGEMV fast path.  Nevertheless benchmarks
should set

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

Keep the current `runcascade(table, decaylist; primary)` implementation as a
reference during development.  Compare every species and mode against
`runcascade(plan; primary)`.

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
Legacy-versus-grouped comparison should therefore use

```julia
norm(Fplan - Fref, Inf) / max(norm(Fref, Inf), 1) <= 1e-12
```

rather than bitwise equality.  `Fprim` should remain exact.  Serial and
mode-threaded execution of the same compiled plan should be bitwise equal
because each mode retains its serial operation order.

## 11. Benchmark plan

Compilation and execution must be measured separately:

1. current one-shot/reference cascade;
2. plan compilation;
3. serial repeated plan application;
4. grouped serial application;
5. grouped `parallel = :modes` application;
6. parallel independent calls where relevant.

For every path report:

- median time and allocations after warm-up;
- slot, transfer, and group counts;
- effective matrix bytes per second;
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

- normalized error no larger than `1e-12`, with exact `Fprim`;
- no copied decay matrices and no cache-format change;
- reused-plan allocations at least 50% below the reference graph walk;
- no representative resident workload more than 3% slower;
- at least 3--5% median improvement on the weighted resident workloads before
  replacing the default executor;
- a measurable mode-threading gain on two-mode workloads, reported without
  assuming ideal two-way scaling.

## 12. Implementation stages

### Stage 1: compiled serial plan

- add dense indices, direct matrix references, and ordered parent steps;
- preserve the current slot operation order;
- establish the reference tests and benchmark harness.

### Stage 2: grouped serial plan

- coalesce identical `(parent, key, daughter)` transfers;
- group all destinations of one `(parent, key)`;
- use direct `mul!` for singleton groups and scratch/scatter otherwise.

### Stage 3: mode-level threading

- move the task boundary outside the entire per-mode cascade;
- allocate one scratch vector per mode task;
- keep primary initialization serial;
- add serial/threaded determinism and concurrent-call tests.

### Stage 4: decide compatibility behavior

After measuring compilation cost, decide whether

```julia
runcascade(table, decaylist; primary)
```

should construct a temporary plan or retain the one-shot implementation.
Repeated workloads should explicitly construct and reuse `CascadePlan`.

### Stage 5: optional wider parallelism

Only if profiling justifies the complexity, prototype deterministic
topological-frontier reductions or a persistent row-parallel kernel.  Keep
these experiments separate from the initial plan so their memory and
rounding-order costs are independently measurable.
