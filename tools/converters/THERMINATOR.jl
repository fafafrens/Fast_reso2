# THERMINATOR name-based conversion. Included inside `DecayDataConverter`.

_roundtherminator(value) = round(Int, value, RoundNearestTiesAway)

function _parsetherminatorparticle(fields, location)
    PL._expectcolumns(fields, 13, location, "THERMINATOR particle")
    return PL._parseat(location, "THERMINATOR particle") do
        mass = parse(Float64, fields[2])
        mass >= 0 || throw(ArgumentError("particle mass must be nonnegative"))
        spin_value = parse(Float64, fields[4])
        isospin_value = parse(Float64, fields[5])
        i3 = parse(Float64, fields[6])
        nq = parse(Float64, fields[7])
        ns = parse(Float64, fields[8])
        naq = parse(Float64, fields[9])
        nas = parse(Float64, fields[10])
        nc = parse(Float64, fields[11])
        nac = parse(Float64, fields[12])
        baryon = _roundtherminator(
            (nq + ns + nc - naq - nas - nac) / 3,
        )
        strangeness = _roundtherminator(nas - ns)
        charm = _roundtherminator(nc - nac)
        charge = _roundtherminator(
            i3 + (baryon + strangeness + charm) / 2,
        )
        degeneracy = 2 * spin_value + PL._spindegeneracyoffset(mass)
        return (
            name = String(fields[1]),
            mass,
            width = parse(Float64, fields[3]),
            spin = spin_value,
            isospin = isospin_value,
            i3,
            baryon,
            charge,
            strangeness,
            charm,
            degeneracy,
            mc = parse(Int, fields[13]),
            location = String(location),
        )
    end
end

function _therminatorids(rows)
    counts = Dict{Int, Int}()
    for row in rows
        counts[row.mc] = get(counts, row.mc, 0) + 1
    end

    ids = [row.mc for row in rows]
    ambiguous = [
        index for (index, row) in pairs(rows)
            if row.mc == 0 || counts[row.mc] > 1
    ]
    isempty(ambiguous) && return ids
    sort!(ambiguous; by = index -> (rows[index].mc, rows[index].name))

    largest = maximum(
        (abs(Int128(row.mc)) for row in rows);
        init = Int128(0),
    )
    length(ambiguous) <= typemax(Int) - largest || throw(ArgumentError(
        "THERMINATOR MC codes leave no room for collision-free internal IDs",
    ))
    for (rank, index) in pairs(ambiguous)
        magnitude = Int(largest + rank)
        ids[index] = rows[index].mc < 0 ? -magnitude : magnitude
    end
    return ids
end

function _cgcindex(path::AbstractString)
    coefficients = Dict{NTuple{6, Int}, Float64}()
    for (line_number, fields) in PL._datarows(path)
        location = "$(String(path)):$line_number"
        PL._expectcolumns(fields, 7, location, "CGC")
        key, coefficient = PL._parseat(location, "CGC") do
            values = ntuple(index -> parse(Float64, fields[index]), 6)
            entry = (
                ntuple(index -> _roundtherminator(2 * values[index]), 6),
                parse(Float64, fields[7]),
            )
            isfinite(last(entry)) || throw(ArgumentError(
                "CGC coefficient must be finite",
            ))
            return entry
        end
        haskey(coefficients, key) && throw(ArgumentError(
            "$location: duplicate CGC entry",
        ))
        coefficients[key] = coefficient
    end
    return coefficients
end

function _therminatorbranching(
        branching,
        flag,
        parent,
        daughters,
        cgc,
        location,
    )
    isfinite(branching) || throw(ArgumentError(
        "$location: THERMINATOR branching fraction must be finite, got " *
            "$branching",
    ))
    flag == 0 && return branching
    flag == 1 || throw(ArgumentError(
        "$location: unsupported THERMINATOR CG flag $flag",
    ))
    length(daughters) == 2 || throw(ArgumentError(
        "$location: THERMINATOR CG rescaling is supported only for " *
            "two-body rows",
    ))
    cgc === nothing && throw(ArgumentError(
        "$location: THERMINATOR row requests CG rescaling; supply --cgc",
    ))

    first_daughter, second_daughter = daughters
    isapprox(
        parent.i3,
        first_daughter.i3 + second_daughter.i3;
        atol = 1e-12,
        rtol = 0,
    ) || throw(ArgumentError(
        "$location: parent I3=$(parent.i3) differs from daughter " *
            "I3=$(first_daughter.i3 + second_daughter.i3)",
    ))
    key = (
        _roundtherminator(2 * first_daughter.isospin),
        _roundtherminator(2 * first_daughter.i3),
        _roundtherminator(2 * second_daughter.isospin),
        _roundtherminator(2 * second_daughter.i3),
        _roundtherminator(2 * parent.isospin),
        _roundtherminator(2 * parent.i3),
    )
    # FastReso uses map indexing, whose missing-key value is zero.
    coefficient = get(cgc, key, 0.0)
    adjusted = branching * coefficient

    same_multiplet =
        abs(first_daughter.mass - second_daughter.mass) < 0.01 &&
        abs(first_daughter.spin - second_daughter.spin) < 0.01 &&
        first_daughter.charm == second_daughter.charm &&
        first_daughter.strangeness == second_daughter.strangeness &&
        abs(first_daughter.i3 - second_daughter.i3) > 0.01
    result = same_multiplet ? 2 * adjusted : adjusted
    isfinite(result) || throw(ArgumentError(
        "$location: THERMINATOR CG rescaling produced a nonfinite branching " *
            "fraction",
    ))
    return result
end

function _loadtherminator(
        particle_path::AbstractString,
        decay_path::AbstractString;
        cgc_path::Union{Nothing, AbstractString} = nothing,
    )
    rows = [
        _parsetherminatorparticle(
            fields,
            "$(String(particle_path)):$line_number",
        ) for (line_number, fields) in PL._datarows(particle_path)
    ]
    names = Dict{String, Int}()
    for (index, row) in pairs(rows)
        startswith(row.name, "MC:") && throw(ArgumentError(
            "$(row.location): THERMINATOR particle names beginning with " *
                "'MC:' are reserved for source-code aliases",
        ))
        haskey(names, row.name) && throw(ArgumentError(
            "$(row.location): duplicate THERMINATOR particle name " *
                "$(repr(row.name))",
        ))
        names[row.name] = index
    end

    ids = _therminatorids(rows)
    particles = Dict{Int, PL.ParticleInfo}()
    aliases = Dict{Int, Vector{String}}()
    for (row, particle_id) in zip(rows, ids)
        particles[particle_id] = PL.ParticleInfo(
            particle_id,
            row.name,
            row.mass,
            row.width,
            row.degeneracy,
            row.baryon,
            row.charge,
            row.strangeness,
            row.charm,
            0,
            _roundtherminator(2 * row.isospin),
        )
        aliases[particle_id] = ["MC:$(row.mc)"]
    end

    cgc = cgc_path === nothing ? nothing : _cgcindex(cgc_path)
    channels_by_parent = Dict{Int, Vector{PL.DecayChannel}}()
    for (line_number, fields) in PL._datarows(decay_path)
        location = "$(String(decay_path)):$line_number"
        length(fields) in (5, 6) || throw(ArgumentError(
            "$location: expected 5 or 6 THERMINATOR decay columns, got " *
                "$(length(fields))",
        ))
        channel = PL._parseat(location, "THERMINATOR decay") do
            daughter_count = length(fields) - 3
            parent_name = String(fields[1])
            daughter_names = String.(fields[2:1 + daughter_count])
            parent_index = get(names, parent_name, nothing)
            parent_index === nothing && throw(ArgumentError(
                "unknown parent name $(repr(parent_name))",
            ))
            daughter_indices = Int[]
            for daughter_name in daughter_names
                daughter_index = get(names, daughter_name, nothing)
                daughter_index === nothing && throw(ArgumentError(
                    "unknown daughter name $(repr(daughter_name))",
                ))
                push!(daughter_indices, daughter_index)
            end
            branching_index = 2 + daughter_count
            branching = parse(Float64, fields[branching_index])
            flag = parse(Int, fields[branching_index + 1])
            source_daughters = rows[daughter_indices]
            branching = _therminatorbranching(
                branching,
                flag,
                rows[parent_index],
                source_daughters,
                cgc,
                location,
            )
            return PL.DecayChannel(
                ids[parent_index],
                branching,
                ids[daughter_indices],
            )
        end
        PL._pushchannel!(channels_by_parent, channel)
    end

    particle_source, particle_sha = PL._sourceidentity(particle_path)
    decay_source, decay_sha = PL._sourceidentity(decay_path)
    if cgc_path !== nothing
        cgc_source, cgc_sha = PL._sourceidentity(cgc_path)
        decay_source = PL._joinprovenance((decay_source, cgc_source))
        decay_sha = PL._joinhashes((decay_sha, cgc_sha))
    end
    return (
        catalog = PL.ParticleCatalog(
            particles;
            aliases,
            path = particle_source,
            sha = particle_sha,
        ),
        network = PL.DecayNetwork(
            channels_by_parent;
            path = decay_source,
            sha = decay_sha,
        ),
    )
end
