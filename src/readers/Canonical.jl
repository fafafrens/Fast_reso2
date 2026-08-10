# Canonical split-file reader, included inside `ParticleList`.

"""
    readparticles(particle_path) -> ParticleCatalog

Read the canonical particle catalog. Rows are

    id name mass width degeneracy B S C Bot I Q [aliases...]

Aliases are optional whitespace-free labels associated with the particle ID.
"""
function readparticles(particle_path::AbstractString)
    particles = Dict{Int, ParticleInfo}()
    aliases = Dict{Int, Vector{String}}()
    for (line_number, fields) in _datarows(particle_path)
        location = "$(String(particle_path)):$line_number"
        length(fields) >= 11 || throw(ArgumentError(
            "$location: expected at least 11 canonical particle columns, " *
                "got $(length(fields))",
        ))
        particle_info = _parsecanonicalparticle(
            fields[1:11], location, "canonical",
        )
        _insertparticle!(particles, particle_info, location)
        if length(fields) > 11
            aliases[particle_info.id] = String.(fields[12:end])
        end
    end

    particle_source, particle_sha = _sourceidentity(particle_path)
    return ParticleCatalog(
        particles;
        aliases,
        path = particle_source,
        sha = particle_sha,
    )
end

"""
    readdecays(decay_path) -> DecayNetwork

Read canonical decay topology. Rows are

    parent n_daughters branching_fraction daughters...

The declared daughter count must exactly match the variable daughter tail.
Particle IDs remain unresolved until the network is combined with a
`ParticleCatalog`.
"""
function readdecays(decay_path::AbstractString)
    channels_by_parent = Dict{Int, Vector{DecayChannel}}()
    for (line_number, fields) in _datarows(decay_path)
        location = "$(String(decay_path)):$line_number"
        length(fields) >= 4 || throw(ArgumentError(
            "$location: expected at least 4 canonical decay columns, " *
                "got $(length(fields))",
        ))
        channel = _parseat(location, "canonical decay") do
            daughter_count = parse(Int, fields[2])
            expected = 3 + daughter_count
            length(fields) == expected || throw(ArgumentError(
                "channel declares $daughter_count daughters but the row has " *
                    "$(length(fields) - 3) daughter fields",
            ))
            return DecayChannel(
                parse(Int, fields[1]),
                parse(Float64, fields[3]),
                _parsedaughters(fields[4:end], daughter_count),
            )
        end
        _pushchannel!(channels_by_parent, channel)
    end

    decay_source, decay_sha = _sourceidentity(decay_path)
    return DecayNetwork(
        channels_by_parent;
        path = decay_source,
        sha = decay_sha,
    )
end
