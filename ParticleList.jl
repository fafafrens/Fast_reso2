"""
    ParticleList

Reader for the hydro-code decay list format (`decays_QM2016Plus_massorder.dat`):
per species a 12-field header

    id  name  mass  width  degeneracy  B  S  C  Bot  I  Q  n_channels

followed by `n_channels` 8-field lines

    id  n_daughters  BR  d1 d2 d3 d4 d5      (unused slots are 0)

Stable particles carry one self-channel (n_daughters = 1, daughter = id).
"""
module ParticleList

export ParticleInfo, DecayChannel, read_decaylist,
       isospin, spin, isbaryon, ismeson, isantiparticle, isstrange

struct ParticleInfo
    id::Int
    name::String
    mass::Float64
    width::Float64
    degeneracy::Float64
    baryon::Int
    charge::Int
    strangeness::Int      # S, file field 7
    charm::Int            # C, file field 8
    twiceisospin::Int     # twice I, derived from file field 10
end

# Constructor with zero strangeness, charm, and isospin.
ParticleInfo(id, name, mass, width, degeneracy, baryon, charge) =
    ParticleInfo(id, name, mass, width, degeneracy, baryon, charge, 0, 0, 0)

"Physical isospin I (half-integer), recovered from the stored 2I."
isospin(particle::ParticleInfo) = particle.twiceisospin / 2
"Physical spin J (half-integer), recovered from the 2J+1 spin degeneracy."
spin(particle::ParticleInfo) = (particle.degeneracy - 1) / 2
"True for a baryon (nonzero baryon number)."
isbaryon(particle::ParticleInfo) = particle.baryon != 0
"True for a meson (zero baryon number)."
ismeson(particle::ParticleInfo) = particle.baryon == 0
"True for an antiparticle (negative id in this list's convention)."
isantiparticle(particle::ParticleInfo) = particle.id < 0
"True for a strange hadron (nonzero strangeness)."
isstrange(particle::ParticleInfo) = particle.strangeness != 0

struct DecayChannel
    parent::Int
    br::Float64
    daughters::Vector{Int}
end

"""
    read_decaylist(path) -> (particles, channels_by_parent)

`particles :: Dict{Int,ParticleInfo}` keyed by id, and
`channels_by_parent :: Dict{Int,Vector{DecayChannel}}` groups channels by
parent id (absent key = no channel rows for that id).
"""
function read_decaylist(path::AbstractString)
    particles = Dict{Int,ParticleInfo}()
    channels_by_parent = Dict{Int,Vector{DecayChannel}}()

    for line in eachline(path)
        fields = split(line)
        (isempty(fields) || startswith(fields[1], "#")) && continue

        # Field count identifies the row type because names contain no whitespace.
        if length(fields) == 12
            particle_id = parse(Int, fields[1])
            name = String(fields[2])
            mass = parse(Float64, fields[3])
            width = parse(Float64, fields[4])
            degeneracy = parse(Float64, fields[5])
            baryon = round(Int, parse(Float64, fields[6]))
            charge = round(Int, parse(Float64, fields[11]))
            strangeness = round(Int, parse(Float64, fields[7]))
            charm = round(Int, parse(Float64, fields[8]))
            twice_isospin = round(Int, 2 * parse(Float64, fields[10]))

            # Bottomness (field 9) and the declared channel count (12) are not stored.
            particles[particle_id] = ParticleInfo(
                particle_id, name, mass, width, degeneracy, baryon, charge,
                strangeness, charm, twice_isospin,
            )
        elseif length(fields) == 8
            parent_id = parse(Int, fields[1])
            daughter_count = parse(Int, fields[2])
            1 ≤ daughter_count ≤ 5 ||
                error("channel n_daughters=$daughter_count out of range 1–5: $line")

            branching_ratio = parse(Float64, fields[3])
            daughter_ids = [parse(Int, fields[3 + index])
                            for index in 1:daughter_count]

            # Reject nonzero padding so under-declared daughters are not dropped.
            all(index -> parse(Int, fields[3 + index]) == 0,
                daughter_count + 1:5) ||
                error("channel declares $daughter_count daughters but has nonzero padding: $line")

            channel = DecayChannel(parent_id, branching_ratio, daughter_ids)
            parent_channels = get!(channels_by_parent, parent_id, DecayChannel[])
            push!(parent_channels, channel)
        else
            error("unrecognized line ($(length(fields)) fields): $line")
        end
    end
    return particles, channels_by_parent
end

end # module
