module CrosscheckTools

using Printf
using Statistics

export SpectrumRow, ReferenceData, ErrorSummary,
       read_reference, reference_grid, metadata_float,
       closest_rows, error_summary, print_summary

struct SpectrumRow
    momentum::Float64
    primary::Float64
    total::Float64
end

struct ReferenceData
    metadata::Dict{String,String}
    spectra::Dict{Int,Vector{SpectrumRow}}
    yields::Dict{Int,Tuple{Float64,Float64}}
end

struct ErrorSummary
    count::Int
    max_relative::Float64
    rms_relative::Float64
    scaled_max_absolute::Float64
    worst_index::Int
end

function read_reference(path::AbstractString)
    metadata = Dict{String,String}()
    spectra = Dict{Int,Vector{SpectrumRow}}()
    yields = Dict{Int,Tuple{Float64,Float64}}()

    for (line_number, line) in enumerate(eachline(path))
        fields = split(line)
        isempty(fields) && continue

        if fields[1] == "#"
            length(fields) ≥ 2 || continue
            key = fields[2]
            if key == "yield"
                length(fields) == 5 ||
                    error("$path:$line_number: malformed yield row")
                particle_id = parse(Int, fields[3])
                haskey(yields, particle_id) &&
                    error("$path:$line_number: duplicate yield for PDG $particle_id")
                primary = parse(Float64, fields[4])
                total = parse(Float64, fields[5])
                isfinite(primary) && isfinite(total) ||
                    error("$path:$line_number: non-finite yield")
                yields[particle_id] = (primary, total)
            elseif length(fields) ≥ 3
                metadata[key] = join(fields[3:end], " ")
            end
            continue
        end

        length(fields) == 4 ||
            error(
                "$path:$line_number: expected four numerical columns; " *
                "raw FastReso *_Fj.out files are not accepted",
            )
        particle_id = parse(Int, fields[1])
        values = parse.(Float64, fields[2:4])
        all(isfinite, values) ||
            error("$path:$line_number: non-finite spectrum value")
        push!(
            get!(spectra, particle_id, SpectrumRow[]),
            SpectrumRow(values...),
        )
    end

    get(metadata, "fastreso_crosscheck_version", nothing) == "1" ||
        error("$path is not a fastreso_crosscheck v1 reference")
    representation = get(metadata, "representation", nothing)
    representation in ("static", "projected") ||
        error("$path has unknown representation $(repr(representation))")
    isempty(spectra) && error("$path contains no spectra")

    for (particle_id, rows) in spectra
        all(
            index -> rows[index - 1].momentum < rows[index].momentum,
            2:length(rows),
        ) || error("$path: momentum is not strictly increasing for PDG $particle_id")
        haskey(yields, particle_id) ||
            error("$path: missing yield row for PDG $particle_id")
    end
    return ReferenceData(metadata, spectra, yields)
end

function metadata_float(reference::ReferenceData, key::AbstractString)
    value = get(reference.metadata, String(key), nothing)
    value === nothing && error("reference is missing metadata '$key'")
    return parse(Float64, value)
end

function reference_grid(reference::ReferenceData)
    get(reference.metadata, "representation", nothing) == "projected" ||
        error("only projected references have a CascadeTable-compatible grid")
    get(reference.metadata, "grid", nothing) == "uniform" ||
        error("projected reference grid is not declared uniform")

    first_rows = first(values(reference.spectra))
    point_count = length(first_rows)
    point_count ≥ 2 || error("reference grid needs at least two points")
    grid = range(
        first_rows[1].momentum,
        first_rows[end].momentum;
        length = point_count,
    )
    tolerance = 256eps(Float64) * max(abs(last(grid)), 1.0)

    for (particle_id, rows) in reference.spectra
        length(rows) == point_count ||
            error("PDG $particle_id uses a different grid length")
        for index in eachindex(rows)
            abs(rows[index].momentum - grid[index]) ≤ tolerance ||
                error(
                    "PDG $particle_id does not use one full-precision " *
                    "uniform grid at point $index",
                )
        end
    end
    return grid
end

function closest_rows(rows::Vector{SpectrumRow}, requested_momenta)
    return [
        rows[argmin(abs(row.momentum - requested) for row in rows)]
        for requested in requested_momenta
    ]
end

function error_summary(
        candidate,
        reference,
        momenta;
        peak_floor = 1.0e-10,
        minimum_momentum = -Inf,
        maximum_momentum = Inf,
    )
    length(candidate) == length(reference) == length(momenta) ||
        throw(DimensionMismatch("candidate, reference, and grid lengths differ"))
    reference_scale = maximum(abs, reference)
    reference_scale > 0 ||
        error("cannot form relative metrics for an identically zero reference")
    selected = [
        index for index in eachindex(reference)
        if minimum_momentum ≤ momenta[index] ≤ maximum_momentum &&
           abs(reference[index]) > peak_floor * reference_scale
    ]
    isempty(selected) &&
        error("no reference points remain after momentum/tail selection")

    difference = candidate[selected] .- reference[selected]
    relative = abs.(difference) ./ abs.(reference[selected])
    worst_local_index = argmax(relative)
    return ErrorSummary(
        length(selected),
        relative[worst_local_index],
        sqrt(sum(abs2, relative) / length(relative)),
        maximum(abs, difference) / reference_scale,
        selected[worst_local_index],
    )
end

function print_summary(
        label,
        summary::ErrorSummary,
        momenta,
        candidate,
        reference,
    )
    index = summary.worst_index
    @printf(
        "%-22s n=%4d max_rel=%10.3e rms_rel=%10.3e peak_abs=%10.3e worst_p=%8.4f\n",
        label,
        summary.count,
        summary.max_relative,
        summary.rms_relative,
        summary.scaled_max_absolute,
        momenta[index],
    )
    @printf(
        "  candidate=% .12e reference=% .12e\n",
        candidate[index],
        reference[index],
    )
end

end # module
