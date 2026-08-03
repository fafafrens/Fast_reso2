# Fixed generic split-file adapter, included inside `ParticleList`.

"""
    _readgenericparticles(particle_path) -> ParticleCatalog

Read particle metadata from the fixed generic convention. Particle rows are

    id name mass width degeneracy B S C Bot I Q
"""
function _readgenericparticles(particle_path::AbstractString)
    particles = Dict{Int, ParticleInfo}()
    for (line_number, fields) in _datarows(particle_path)
        location = "$(String(particle_path)):$line_number"
        _insertparticle!(
            particles,
            _parsecanonicalparticle(fields, location, "generic"),
            location,
        )
    end

    particle_source, particle_sha = _sourceidentity(particle_path)
    return ParticleCatalog(
        particles;
        path = particle_source,
        sha = particle_sha,
    )
end

"""
    _readgenericdecays(decay_path) -> DecayNetwork

Read decay topology from the fixed generic convention. Decay rows are

    parent n_daughters branching_fraction daughters...

The declared daughter count must exactly match the variable daughter tail.
Particle IDs are deliberately left unresolved until the network is combined
with a `ParticleCatalog`.
"""
function _readgenericdecays(decay_path::AbstractString)
    channels_by_parent = Dict{Int, Vector{DecayChannel}}()
    for (line_number, fields) in _datarows(decay_path)
        location = "$(String(decay_path)):$line_number"
        length(fields) >= 4 || throw(ArgumentError(
            "$location: expected at least 4 generic decay columns, " *
                "got $(length(fields))",
        ))
        channel = _parseat(location, "generic decay") do
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
