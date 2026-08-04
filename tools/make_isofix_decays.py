"""Patch three isospin-inconsistent entries in the QM2016+ decay list.

Header (12 fields):  id name mass width degeneracy B S C Bot I Q n_channels
Channel (8 fields):  id n_daughters BR d1 d2 d3 d4 d5     (unused slots 0)

Reads  data/azyhydro/decays_QM2016Plus_massorder.dat
Writes data/azyhydro/decays_QM2016Plus_massorder_isofix.dat        isospin fixes only
       data/azyhydro/decays_QM2016Plus_massorder_isofix_sigma.dat  the same, plus the sigma

The two outputs differ by exactly three lines, so the corrections below -- which
are unambiguous fixes -- can be used without also buying into the sigma, which
is a modelling choice.  Every other line is byte-identical to the input.

Each patch replaces a placeholder parametrisation with the isospin
decomposition its multiplet requires.  Summed over a multiplet, an
isospin-symmetric table must give <pi0> = <pi+> = <pi->; as shipped, the a1 and
the K(1630) do not, which shows up as a spurious pi0 excess above ~1 GeV.

  eta (221)
      Shipped: 3pi0 0.55, pi+pi-pi0 0.38, pi+pi-gamma 0.07 -- eta -> gamma
      gamma is missing and the hadronic modes are renormalised to 1, which
      inflates eta feed-down (all of it pi0-rich) by 1/(1-0.394) ~ 1.6.
      The *relative* hadronic rates are fine: 0.55/0.38 = 1.45 vs PDG
      32.56/23.02 = 1.41.  Restoring gamma gamma is purely a normalisation fix.
      eta -> 3pi is genuinely isospin-violating (dI=1, driven by m_d - m_u),
      so the multiplet imbalance it produces is physical and stays.

  a1(1260) (20213, 20113, -20213)
      Shipped: a0(980) gamma 0.7 + a direct 3pi 0.3, with a1+ -> pi+pi0pi0 but
      a1_0 -> pi0pi0pi0.  The latter cannot occur at all: a J=1 state has no
      amplitude to three identical pseudoscalars (the same Bose-symmetry
      argument that forbids omega -> 3pi0).  The a0(980) gamma branch is also a
      placeholder (real a1 -> pi gamma is ~1e-3) and, since a0(980) -> eta pi
      at BR 1, it manufactures an eta in 70% of a1 decays -- a second, indirect
      pi0 enhancement.
      Patched to the physical mode a1 -> rho pi at BR 1.  The epsilon_abc
      coupling forbids a1_0 -> rho0 pi0, giving 0.5/0.5 per charge state.  (The
      other real mode, sigma pi at ~18%, is not representable: no f0(500) in
      this table.)

  K(1630) (4000002, 4000003, -4000002, -4000003)
      Shipped: all four charge states -> K pi+ pi- at BR 1, with no K pi0 pi0
      partner -- the single largest direct imbalance in the table.
      J^P is unmeasured and PDG lists only "seen in K pi pi", so the minimal
      assumption is that the pi pi pair carries I=0 and the kaon carries the
      parent's isospin, giving pi+pi- : pi0pi0 = 2/3 : 1/3 and leaving the
      kaon charge untouched.

The second output additionally carries the sigma, which this table lacks:

  f0(500) / sigma (9000221, an id no entry in this list uses)
      Added at the PDG pole, m = 0.475, Gamma = 0.550, decaying to pi pi with
      the I=0 split 2/3 : 1/3 -- the same decomposition f0(980) already uses.
      --sigma-mass/--sigma-width move it.

      What it does, measured against this file without it: pion dN/dy rises
      7.4% (pi+) and 6.6% (pi0), protons are untouched to 4 digits, and
      pi0/pi+ barely moves (1.458 -> 1.436 at pT=0) since sigma -> pi pi is
      isospin symmetric.  Almost none of that comes from the a1 -> sigma pi
      channel it unlocks (routing 23% of a1 decays there moves dN/dy by 0.0002);
      it is the sigma's own thermal abundance.

      Two caveats worth carrying.  (1) The mass sets the yield -- 5% at m=0.55,
      10% at m=0.40 -- while the width sets the *shape*: at fixed mass, Gamma
      0.55 vs 0.30 changes pi+ at 1 GeV by 17% vs 5%, because the +-4 Gamma
      Breit-Wigner window reaches m + 4 Gamma = 2.675 GeV and those tails emit
      hard pions.  A sigma with strength at 2.7 GeV is unphysical; its spectral
      function is not a Breit-Wigner.  (2) The sigma is the I=0 s-wave pi pi
      interaction, and those pions are already in the gas as free pions --
      adding it as a resonance double counts, and in the S-matrix formulation
      the attractive I=0 contribution is largely cancelled by the repulsive I=2
      channel, which has no resonance to stand for it.  That is why tables in
      this lineage omit it.

Branching ratios below are PDG averages; edit ETA_MODES if you want different
inputs.  Run with no arguments:  python3 tools/make_isofix_decays.py
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("RESO_REPO", os.path.dirname(HERE))
SRC = os.path.join(REPO, "data/azyhydro/decays_QM2016Plus_massorder.dat")
OUT = os.path.join(REPO, "data/azyhydro/decays_QM2016Plus_massorder_isofix.dat")
OUT_SIGMA = os.path.join(REPO, "data/azyhydro/decays_QM2016Plus_massorder_isofix_sigma.dat")

GAMMA, PI0, PIP, PIM = 22, 111, 211, -211
RHOP, RHO0, RHOM = 213, 113, -213

SIGMA_ID = 9000221                # PDG id for f0(500); unused by this list
SIGMA_MASS, SIGMA_WIDTH = 0.475, 0.550                  # PDG T-matrix pole

# PDG branching fractions; they sum to 0.9916, so they are renormalised to 1
# (the ~0.8% remainder is e+e-gamma, mu+mu-gamma and other rare modes).
ETA_MODES = [
    (0.3936, [GAMMA, GAMMA]),
    (0.3256, [PI0, PI0, PI0]),
    (0.2302, [PIP, PIM, PI0]),
    (0.0422, [PIP, PIM, GAMMA]),
]

TWO_THIRDS, ONE_THIRD = 2.0 / 3.0, 1.0 / 3.0


def normalise(modes):
    """Scale branching ratios to sum to exactly 1 after rounding."""
    total = sum(br for br, _ in modes)
    scaled = [(round(br / total, 9), ds) for br, ds in modes]
    drift = round(1.0 - sum(br for br, _ in scaled), 9)
    return [(round(scaled[0][0] + drift, 9), scaled[0][1])] + scaled[1:]


def channels(parent, modes):
    """Format 8-field channel rows, zero-padding the unused daughter slots."""
    rows = []
    for br, daughters in modes:
        padded = daughters + [0] * (5 - len(daughters))
        fields = [parent, len(daughters), _fmt(br)] + padded
        rows.append("\t".join(str(f) for f in fields))
    return rows


def _fmt(br):
    return f"{br:.9f}".rstrip("0").rstrip(".")


# parent id -> replacement channel list, as (BR, [daughter ids])
PATCHES = {
    221: normalise(ETA_MODES),

    20213: [(0.5, [RHOP, PI0]), (0.5, [RHO0, PIP])],      # a1+
    20113: [(0.5, [RHOP, PIM]), (0.5, [RHOM, PIP])],      # a1_0
    -20213: [(0.5, [RHOM, PI0]), (0.5, [RHO0, PIM])],     # a1-

    4000002: [(TWO_THIRDS, [321, PIP, PIM]),              # K(1630)+
              (ONE_THIRD, [321, PI0, PI0])],
    4000003: [(TWO_THIRDS, [311, PIP, PIM]),              # K(1630)0
              (ONE_THIRD, [311, PI0, PI0])],
    -4000002: [(TWO_THIRDS, [-321, PIP, PIM]),            # anti-K(1630)+
               (ONE_THIRD, [-321, PI0, PI0])],
    -4000003: [(TWO_THIRDS, [-311, PIP, PIM]),            # anti-K(1630)0
               (ONE_THIRD, [-311, PI0, PI0])],
}


def sigma_block(mass, width):
    """Header + I=0 pi pi channels for the f0(500)."""
    header = "\t".join([str(SIGMA_ID), "f0(500)", f"{mass:g}", f"{width:g}",
                        "1", "0", "0", "0", "0", "0", "0", "2"])
    return [header] + channels(SIGMA_ID, [(TWO_THIRDS, [PIP, PIM]),
                                          (ONE_THIRD, [PI0, PI0])])


def build(sigma):
    """Patched line list. `sigma` is (mass, width) to insert f0(500), or None.

    The two outputs differ by exactly the three sigma lines, so the isospin
    corrections -- which are unambiguous fixes -- stay separable from the
    sigma, which is a modelling choice.
    """
    out, skipping, patched, placed = [], None, {}, sigma is None
    for line in open(SRC).read().split("\n"):
        fields = line.split()
        if len(fields) == 12:                    # particle header
            pid = int(fields[0])
            # the list is mass-descending: the sigma goes before the first
            # entry lighter than it
            if not placed and float(fields[2]) < sigma[0]:
                out.extend(sigma_block(*sigma))
                placed = True
            skipping = pid if pid in PATCHES else None
            if skipping is not None:
                modes = PATCHES[pid]
                fields[11] = str(len(modes))     # declared channel count
                out.append("\t".join(fields))
                out.extend(channels(pid, modes))
                patched[pid] = fields[1]
                continue
        elif len(fields) == 8:                   # channel row
            if skipping is not None and int(fields[0]) == skipping:
                continue                         # drop the shipped channels
        out.append(line)

    if not placed:
        raise SystemExit("no entry lighter than the sigma: nowhere to insert it")
    missing = set(PATCHES) - set(patched)
    if missing:
        raise SystemExit(f"parent ids not found in {SRC}: {sorted(missing)}")
    return out, patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-mass", type=float, default=SIGMA_MASS)
    ap.add_argument("--sigma-width", type=float, default=SIGMA_WIDTH)
    ap.add_argument("--out", default=OUT, help="output without the sigma")
    ap.add_argument("--out-sigma", default=OUT_SIGMA, help="output with it")
    args = ap.parse_args()

    for path, sigma in ((args.out, None),
                        (args.out_sigma, (args.sigma_mass, args.sigma_width))):
        lines, patched = build(sigma)
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
        print(f"wrote {path}")
        if sigma:
            print(f"  {SIGMA_ID:>9} f0(500){'':<10} 2 channels, "
                  f"m = {sigma[0]:g}, width = {sigma[1]:g}")
        for pid, name in sorted(patched.items(), key=lambda kv: kv[1]):
            modes = PATCHES[pid]
            print(f"  {pid:>9} {name:<16} {len(modes)} channels, "
                  f"BR sum {sum(br for br, _ in modes):.6f}")


if __name__ == "__main__":
    main()
