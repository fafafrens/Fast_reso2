"""
    ParticleList

Particle metadata and unresolved decay topology. Shared input mechanics live
here; only the canonical split-file reader is part of the runtime module.
External formats are converted by standalone tools.
"""
module ParticleList

using SHA

export ParticleInfo, DecayChannel,
       ParticleCatalog, DecayNetwork,
       readparticles, readdecays,
       particle, particleids,
       isospin, spin, isbaryon, ismeson, isantiparticle, isstrange, ischarm,
       isbottom, quarkdigits, hasflavour

struct ParticleInfo
    id::Int
    name::String
    mass::Float64
    width::Float64
    degeneracy::Float64
    baryon::Int
    charge::Int
    strangeness::Int      # S
    charm::Int            # C
    bottomness::Int       # bottom flavour quantum number
    twiceisospin::Int     # 2I
end

# Compatibility constructors used by programmatic decay lists.
ParticleInfo(id, name, mass, width, degeneracy, baryon, charge) =
    ParticleInfo(id, name, mass, width, degeneracy, baryon, charge, 0, 0, 0, 0)

ParticleInfo(
        id, name, mass, width, degeneracy, baryon, charge,
        strangeness, charm, twiceisospin,
    ) = ParticleInfo(
        id, name, mass, width, degeneracy, baryon, charge,
        strangeness, charm, 0, twiceisospin,
    )

"Physical isospin I (half-integer), recovered from the stored 2I."
isospin(particle::ParticleInfo) = particle.twiceisospin / 2
_spindegeneracyoffset(mass) = mass > 0.001 ? 1 : 0

"Physical spin J, recovered from the massive 2J+1 or massless 2J degeneracy."
spin(particle::ParticleInfo) =
    (particle.degeneracy - _spindegeneracyoffset(particle.mass)) / 2
"True for a baryon (nonzero baryon number)."
isbaryon(particle::ParticleInfo) = particle.baryon != 0
"True for a meson (zero baryon number)."
ismeson(particle::ParticleInfo) = particle.baryon == 0
"True for an antiparticle (negative id in this list's convention)."
isantiparticle(particle::ParticleInfo) = particle.id < 0
"True for a strange hadron (nonzero strangeness)."
isstrange(particle::ParticleInfo) = particle.strangeness != 0
"True for a charm hadron (nonzero charm)."
ischarm(particle::ParticleInfo) = particle.charm != 0
"True for a bottom hadron (nonzero bottomness)."
isbottom(particle::ParticleInfo) = particle.bottomness != 0

"""
    quarkdigits(id) -> (q1, q2, q3)

Quark-content digits of a Monte-Carlo ID, most significant first, using the
standard `n_q1 n_q2 n_q3 n_J` layout: `q1` is zero for a meson and names the
third quark of a baryon. Flavours are `1 = d, 2 = u, 3 = s, 4 = c, 5 = b`.

Returns zeros for anything below 100 — leptons and gauge bosons have no quark
content, and reading their digits would report the photon (22) as carrying an
up quark.
"""
function quarkdigits(id::Integer)
    magnitude = abs(id)
    magnitude < 100 && return (0, 0, 0)
    # The three digits left of n_J. Division and modulo read them directly;
    # going through a padded string would allocate one per call.
    return (
        (magnitude ÷ 1000) % 10,
        (magnitude ÷ 100) % 10,
        (magnitude ÷ 10) % 10,
    )
end
quarkdigits(particle::ParticleInfo) = quarkdigits(particle.id)

"""
    hasflavour(particle, flavour) -> Bool

True when the species contains a quark of this flavour, as quark *or*
antiquark.

This is a different question from `ischarm`/`isbottom`, which test the net
flavour quantum number. Quarkonia carry zero net flavour because quark and
antiquark cancel, so `ischarm(J/psi)` is false while `hasflavour(J/psi, 4)` is
true. Use the net-number predicates for open flavour and this one to select a
sector: partitioning a charm+bottom list by net flavour files every J/psi,
Upsilon, chi_c and chi_b under neither heading.
"""
hasflavour(particle::ParticleInfo, flavour::Integer) =
    flavour in quarkdigits(particle)
hasflavour(id::Integer, flavour::Integer) = flavour in quarkdigits(id)

struct DecayChannel
    parent::Int
    br::Float64
    daughters::Vector{Int}
end

# ---------------------------------------------------------------------------
# Catalog and unresolved topology
# ---------------------------------------------------------------------------

struct _ValidatedInputToken end
const _VALIDATED_INPUT = _ValidatedInputToken()

"""
    ParticleCatalog(particles; aliases = Dict(), path = "", sha = "")

Particle metadata keyed by integer particle ID. `ParticleInfo.name` is the
canonical display name; `aliases` may map an ID to additional exact labels.
Names are a multimap because distinct particle IDs may legitimately share a
label. Use [`particleids`](@ref) for name lookup when ambiguity is possible.
"""
struct ParticleCatalog
    byid::Dict{Int, ParticleInfo}
    byname::Dict{String, Vector{Int}}
    path::String
    sha::String

    function ParticleCatalog(
            byid::Dict{Int, ParticleInfo},
            byname::Dict{String, Vector{Int}},
            path::String,
            sha::String,
            ::_ValidatedInputToken,
        )
        return new(byid, byname, path, sha)
    end
end

function _nameindex(
        particles::Dict{Int, ParticleInfo},
        aliases,
    )
    byname = Dict{String, Vector{Int}}()
    for particle_id in sort!(collect(keys(particles)))
        particle = particles[particle_id]
        particle.id == particle_id || throw(ArgumentError(
            "particle dictionary key $particle_id disagrees with " *
                "ParticleInfo.id=$(particle.id)",
        ))
        isempty(particle.name) || push!(
            get!(byname, particle.name, Int[]), particle_id,
        )
    end
    for (particle_id, labels) in aliases
        haskey(particles, particle_id) || throw(ArgumentError(
            "alias supplied for unknown particle ID $particle_id",
        ))
        for label in labels
            isempty(label) || push!(
                get!(byname, String(label), Int[]), particle_id,
            )
        end
    end
    for ids in values(byname)
        sort!(unique!(ids))
    end
    return byname
end

function ParticleCatalog(
        particles::Dict{Int, ParticleInfo};
        aliases = Dict{Int, Vector{String}}(),
        path::AbstractString = "",
        sha::AbstractString = "",
    )
    return ParticleCatalog(
        particles,
        _nameindex(particles, aliases),
        String(path),
        String(sha),
        _VALIDATED_INPUT,
    )
end

"Return the particle with integer `id`."
particle(catalog::ParticleCatalog, id::Integer) = catalog.byid[Int(id)]

"Return all particle IDs carrying the exact canonical name or alias `label`."
function particleids(catalog::ParticleCatalog, label::AbstractString)
    ids = get(catalog.byname, String(label), nothing)
    return ids === nothing ? Int[] : copy(ids)
end

"Return the unique particle carrying `label`, rejecting unknown or ambiguous names."
function particle(catalog::ParticleCatalog, label::AbstractString)
    ids = get(catalog.byname, String(label), nothing)
    ids === nothing && throw(ArgumentError("unknown particle name: $label"))
    length(ids) == 1 || throw(ArgumentError(
        "particle name '$label' is ambiguous; matching IDs: $(join(ids, ", "))",
    ))
    return catalog.byid[only(ids)]
end

function _differentparticlefields(left::ParticleInfo, right::ParticleInfo)
    return [
        field for field in fieldnames(ParticleInfo)
            if field != :name &&
                !isequal(getfield(left, field), getfield(right, field))
    ]
end

function _replaceparticlename(particle::ParticleInfo, name::String)
    return ParticleInfo(
        particle.id,
        name,
        particle.mass,
        particle.width,
        particle.degeneracy,
        particle.baryon,
        particle.charge,
        particle.strangeness,
        particle.charm,
        particle.bottomness,
        particle.twiceisospin,
    )
end

function _joinprovenance(values)
    selected = String[value for value in values if !isempty(value)]
    return join(selected, " + ")
end

function _joinhashes(values)
    selected = String[value for value in values if !isempty(value)]
    return join(selected, "+")
end

"Merge catalogs by particle ID, retaining alternate names as aliases."
function Base.merge(
        first::ParticleCatalog,
        remaining::ParticleCatalog...,
    )
    catalogs = (first, remaining...)
    particles = Dict{Int, ParticleInfo}()
    aliases = Dict{Int, Vector{String}}()

    for catalog in catalogs
        for (label, particle_ids) in catalog.byname
            for particle_id in particle_ids
                haskey(catalog.byid, particle_id) || throw(ArgumentError(
                    "name index '$label' refers to unknown particle ID " *
                        "$particle_id",
                ))
                push!(get!(aliases, particle_id, String[]), label)
            end
        end
        for particle_id in sort!(collect(keys(catalog.byid)))
            incoming = catalog.byid[particle_id]

            if haskey(particles, particle_id)
                current = particles[particle_id]
                fields = _differentparticlefields(current, incoming)
                isempty(fields) || throw(ArgumentError(
                    "conflicting particle definition for ID $particle_id; " *
                        "different fields: $(join(fields, ", "))",
                ))
                if isempty(current.name) && !isempty(incoming.name)
                    particles[particle_id] =
                        _replaceparticlename(current, incoming.name)
                end
            else
                particles[particle_id] = incoming
            end
        end
    end

    return ParticleCatalog(
        particles;
        aliases,
        path = _joinprovenance(catalog.path for catalog in catalogs),
        sha = _joinhashes(catalog.sha for catalog in catalogs),
    )
end

_isselfchannel(channel::DecayChannel, parent_id::Integer) =
    length(channel.daughters) == 1 && only(channel.daughters) == parent_id

_isterminalblock(channels, parent_id) =
    !isempty(channels) &&
    all(channel -> _isselfchannel(channel, parent_id), channels)

function _channelsignature(channel::DecayChannel)
    return (channel.parent, channel.br, Tuple(sort(channel.daughters)))
end

function _samechannels(left, right)
    left === right && return true
    length(left) == length(right) || return false
    left_signatures = sort!([_channelsignature(channel) for channel in left])
    right_signatures = sort!([_channelsignature(channel) for channel in right])
    return isequal(left_signatures, right_signatures)
end

"""
    DecayNetwork(channels_by_parent; path = "", sha = "")

Decay topology keyed by parent ID. A network deliberately does not own
particle metadata, so daughter IDs may be resolved only after several sources
are merged. `decaying_ids` survives topology filtering and prevents a filtered
parent from becoming a terminal output accidentally. Raw input may retain a
legacy five-body row, but resolving it into a runnable `DecayList` rejects
unsupported multiplicities.
"""
struct DecayNetwork
    byparent::Dict{Int, Vector{DecayChannel}}
    decaying_ids::Vector{Int}
    terminal_ids::Vector{Int}
    path::String
    sha::String

    function DecayNetwork(
            byparent::Dict{Int, Vector{DecayChannel}},
            decaying_ids::Vector{Int},
            terminal_ids::Vector{Int},
            path::String,
            sha::String,
            ::_ValidatedInputToken,
        )
        return new(byparent, decaying_ids, terminal_ids, path, sha)
    end
end

function DecayNetwork(
        channels_by_parent::Dict{Int, Vector{DecayChannel}};
        decaying_ids = nothing,
        terminal_ids = nothing,
        path::AbstractString = "",
        sha::AbstractString = "",
    )
    inferred_decaying = Int[]
    inferred_terminal = Int[]
    for (parent_id, channels) in channels_by_parent
        all(channel -> channel.parent == parent_id, channels) ||
            throw(ArgumentError(
                "channel parent disagrees with dictionary key $parent_id",
            ))
        if _isterminalblock(channels, parent_id)
            push!(inferred_terminal, parent_id)
        elseif !isempty(channels)
            push!(inferred_decaying, parent_id)
        end
    end

    selected_decaying = collect(Int, something(decaying_ids, ()))
    append!(selected_decaying, inferred_decaying)
    sort!(unique!(selected_decaying))

    selected_terminal = collect(Int, something(terminal_ids, ()))
    append!(selected_terminal, inferred_terminal)
    sort!(unique!(selected_terminal))

    conflicting_ids = intersect(selected_decaying, selected_terminal)
    isempty(conflicting_ids) || throw(ArgumentError(
        "particle IDs cannot be both decaying and terminal: " *
            join(conflicting_ids, ", "),
    ))

    return DecayNetwork(
        channels_by_parent,
        selected_decaying,
        selected_terminal,
        String(path),
        String(sha),
        _VALIDATED_INPUT,
    )
end

"Merge unresolved networks without requiring their particle catalogs."
function Base.merge(
        first::DecayNetwork,
        remaining::DecayNetwork...,
    )
    networks = (first, remaining...)
    channels_by_parent = Dict{Int, Vector{DecayChannel}}()
    decaying_ids = Int[]
    terminal_ids = Int[]

    for network in networks
        append!(decaying_ids, network.decaying_ids)
        append!(terminal_ids, network.terminal_ids)
    end

    decaying_set = Set(decaying_ids)
    for network in networks
        for parent_id in sort!(collect(keys(network.byparent)))
            channels = network.byparent[parent_id]
            parent_id in decaying_set &&
                _isterminalblock(channels, parent_id) && continue
            if haskey(channels_by_parent, parent_id)
                _samechannels(channels_by_parent[parent_id], channels) ||
                    throw(ArgumentError(
                        "conflicting channel definitions for parent ID " *
                            "$parent_id",
                    ))
            else
                channels_by_parent[parent_id] = channels
            end
        end
    end

    filter!(particle_id -> particle_id ∉ decaying_set, terminal_ids)

    return DecayNetwork(
        channels_by_parent;
        decaying_ids,
        terminal_ids,
        path = _joinprovenance(network.path for network in networks),
        sha = _joinhashes(network.sha for network in networks),
    )
end

# ---------------------------------------------------------------------------
# Canonical split-file reader
# ---------------------------------------------------------------------------

function _sourceidentity(path::AbstractString)
    source_path = String(path)
    return source_path, bytes2hex(sha256(read(path)))
end

function _datarows(path::AbstractString)
    rows = Tuple{Int, Vector{SubString{String}}}[]
    for (line_number, line) in enumerate(eachline(path))
        content = first(split(line, '#'; limit = 2))
        isempty(strip(content)) && continue
        push!(rows, (line_number, split(content)))
    end
    return rows
end

function _expectcolumns(fields, count, location, kind)
    length(fields) == count || throw(ArgumentError(
        "$location: expected $count $kind columns, got $(length(fields))",
    ))
end

function _parseat(action, location, kind)
    try
        return action()
    catch error
        error isa InterruptException && rethrow()
        message = sprint(showerror, error)
        startswith(message, "$location:") && rethrow()
        throw(ArgumentError("$location: invalid $kind row: $message"))
    end
end

_parsequantum(value) = round(Int, parse(Float64, value))

function _parsecanonicalparticle(fields, location, kind)
    # Shared with the AZYHYDRO converter, whose first 11 columns are this exact
    # layout -- hence `kind`, which names the format in any error.
    #
    # No column check here: both callers validate the row width and then slice
    # exactly 11 fields, so one could never fire.
    return _parseat(location, "$kind particle") do
        particle_id = parse(Int, fields[1])
        particle_id != 0 || throw(ArgumentError(
            "particle ID 0 is reserved and cannot identify a particle",
        ))
        return ParticleInfo(
            particle_id,
            String(fields[2]),
            parse(Float64, fields[3]),
            parse(Float64, fields[4]),
            parse(Float64, fields[5]),
            _parsequantum(fields[6]),
            _parsequantum(fields[11]),
            _parsequantum(fields[7]),
            _parsequantum(fields[8]),
            _parsequantum(fields[9]),
            round(Int, 2 * parse(Float64, fields[10])),
        )
    end
end

function _insertparticle!(particles, particle_info, location)
    haskey(particles, particle_info.id) && throw(ArgumentError(
        "$location: duplicate particle ID $(particle_info.id)",
    ))
    particles[particle_info.id] = particle_info
    return particles
end

function _pushchannel!(channels_by_parent, channel)
    push!(
        get!(channels_by_parent, channel.parent, DecayChannel[]),
        channel,
    )
    return channels_by_parent
end

function _parsedaughters(values, daughter_count)
    1 <= daughter_count <= 5 || throw(ArgumentError(
        "channel n_daughters=$daughter_count out of range 1-5",
    ))
    daughters = [parse(Int, values[index]) for index in 1:daughter_count]
    any(==(0), daughters) && throw(ArgumentError(
        "particle ID 0 is reserved and cannot identify a daughter",
    ))
    return daughters
end

include(joinpath(@__DIR__, "readers", "Canonical.jl"))

end # module
