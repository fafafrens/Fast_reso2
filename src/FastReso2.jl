"""
    FastReso2

Resonance feed-down for particle spectra and thermal yields. The package root
loads each public component once so particle, decay-model, cascade, and thermal
APIs share the same Julia types.
"""
module FastReso2

include("ParticleList.jl")
include("DecaySpectra.jl")
include("DecayModel.jl")
include("CascadeTables.jl")
include("ThermalModel.jl")

using .ParticleList
using .DecaySpectra
using .DecayModel
using .CascadeTables
using .ThermalModel

export ParticleInfo, DecayChannel, ParticleCatalog, DecayNetwork,
    readparticles, readdecays, particle, particleids,
    isospin, spin, isbaryon, ismeson, isantiparticle, isstrange, ischarm,
    isbottom,
    kallen, MassAverage, TwoBodyDecay, ThreeBodyDecay, FourBodyDecay,
    DecayMatrix, decaymatrices, decaymatrices!, rowscratch, SpectralShape,
    BreitWigner, MassDependentBW, BroadParent, Rules,
    DecayList, validate, feeddown_order, allparticles, stableparticles,
    ChannelKey, CascadeTable, CascadePlan, runcascade, prune!,
    savetables, loadtables,
    ChannelFilter, channelfilter, nbody, parentmass, daughtermass, brabove,
    fromparent, todaughter, fromid, toid, baryonicparent, mesonicparent,
    parentstrangeness, strangeparent, charmparent, parentisospin,
    chemicalpotential, thermaldensity, primaryyields, thermalyields,
    ThermalPlan

end # module FastReso2
