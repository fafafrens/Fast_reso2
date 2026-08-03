# Fixed AZYHYDRO combined-file adapter, included inside `ParticleList`.

function _parseazyhydrochannel(fields, location)
    _expectcolumns(fields, 8, location, "AZYHYDRO decay")
    return _parseat(location, "AZYHYDRO decay") do
        daughter_count = parse(Int, fields[2])
        daughters = _parsedaughters(fields[4:8], daughter_count)
        all(value -> parse(Int, value) == 0, fields[4 + daughter_count:8]) ||
            throw(ArgumentError(
                "channel declares $daughter_count daughters but has " *
                    "nonzero padding",
            ))
        return DecayChannel(
            parse(Int, fields[1]),
            parse(Float64, fields[3]),
            daughters,
        )
    end
end

function _readazyhydrofile(path::AbstractString)
    rows = _datarows(path)
    particles = Dict{Int, ParticleInfo}()
    channels_by_parent = Dict{Int, Vector{DecayChannel}}()
    row_index = 1

    while row_index <= length(rows)
        line_number, fields = rows[row_index]
        location = "$(String(path)):$line_number"
        _expectcolumns(fields, 12, location, "AZYHYDRO particle")
        particle_info = _parsecanonicalparticle(
            fields[1:11], location, "AZYHYDRO",
        )
        channel_count = _parseat(location, "AZYHYDRO particle") do
            count = parse(Int, fields[12])
            count >= 0 || throw(ArgumentError(
                "channel count must be nonnegative",
            ))
            return count
        end
        _insertparticle!(particles, particle_info, location)

        available = length(rows) - row_index
        available >= channel_count || throw(ArgumentError(
            "$location: particle $(particle_info.id) declares $channel_count " *
                "channels but only $available data rows remain",
        ))
        for offset in 1:channel_count
            channel_line, channel_fields = rows[row_index + offset]
            channel_location = "$(String(path)):$channel_line"
            channel = _parseazyhydrochannel(channel_fields, channel_location)
            channel.parent == particle_info.id || throw(ArgumentError(
                "$channel_location: channel parent $(channel.parent) does not " *
                    "match block parent $(particle_info.id)",
            ))
            _pushchannel!(channels_by_parent, channel)
        end
        row_index += channel_count + 1
    end

    source_path, source_sha = _sourceidentity(path)
    return (
        catalog = ParticleCatalog(
            particles;
            path = source_path,
            sha = source_sha,
        ),
        network = DecayNetwork(
            channels_by_parent;
            path = source_path,
            sha = source_sha,
        ),
    )
end

"""
    _readazyhydro(first_path, remaining_paths...) -> (catalog, network)

Read and merge one or more fixed AZYHYDRO combined files. Each particle header

    id name mass width degeneracy B S C Bot I Q n_channels

is followed by exactly `n_channels` padded decay rows

    parent n_daughters BR d1 d2 d3 d4 d5
"""
function _readazyhydro(
        first_path::AbstractString,
        remaining_paths::AbstractString...,
    )
    sources = map(_readazyhydrofile, (first_path, remaining_paths...))
    length(sources) == 1 && return only(sources)
    return (
        catalog = merge((source.catalog for source in sources)...),
        network = merge((source.network for source in sources)...),
    )
end
