# PDG audit of `data/azyhydro/decays_QM2016Plus_massorder.dat`

PDG edition 2026. Report only -- no `.dat` file was modified.

`importance` = largest fraction of the pi0 or pi+ spectrum (at 0.33 or 1.0 GeV) that disappears when this particle's decays are switched off.

## Coverage

| class | particles |
|---|---|
| ad-hoc id (no PDG entry) | 1109 |
| antiparticle (conjugate) | 187 |
| PDG modes: cannot constrain (unmeasured/generic) | 169 |
| standard id, absent from PDG | 48 |
| PDG modes: none | 3 |
| PDG modes: usable (measured, representable) | 2 |

Particles whose channel list PDG can regenerate on its own: eta (`221`), omega(782) (`223`).

## Method, and why most entries cannot be refreshed

A mode is only compared when PDG measures enough of the width to renormalise
against: identifiable, charge-consistent modes must cover >= 90% of it, and the
modes this table can represent must be >= 50% of that. Three structural reasons
block the rest.

1. **No number on the dominant mode.** `rho --> pi pi` is quoted as ~100% in the
   listings text but carries no value in the database; the rho's measured modes
   sum to 1.1%, all of them radiative. Renormalising over what *is* measured
   would give the rho a 91% branch to `pi+ pi- gamma`. Same for `K*(892) --> K pi`,
   `Delta(1232) --> N pi` (0.994, but generic) and `a_1(1260)`, which has no
   numeric mode at all.
2. **Generic multiplet rows.** `pi pi`, `3 pi`, `N pi`, `Lambda pi` name a
   multiplet, not charge states. Splitting them needs Clebsch-Gordan, which PDG
   does not supply -- exactly the gap the a1 and K(1630) patches fill by hand.
3. **Grouping rows.** `exclusive_branching_fractions()` returns PDG's top-level
   partition: for the eta that is {neutral modes, charged modes}, and `2gamma` /
   `3pi0` never appear. Descending to the real modes and dropping the groupings
   is what the explicit-mcid filter in `Pdg.modes` does.

So a PDG refresh is a per-entry judgement, not an import. What the audit is good
for is catching the cases where PDG *does* have the numbers and the table
disagrees.

## Findings (39 particles)

### eta (`221`), m = 0.547, importance 13.6%

- **header** width 1.180e-03 -> PDG 1.308e-06 (902x)
- MISSING  `Gamma Gamma` — PDG 0.3936, renormalised 0.397
- BR-DIFF  `pi0 pi0 pi0` — table 0.550 vs PDG 0.328
- BR-DIFF  `Anti-pi+ pi0 pi+` — table 0.380 vs PDG 0.232

### rho0 (`113`), m = 0.775, importance 12.8%

- _PDG cannot constrain these branching ratios: measured modes cover only 1.1% of the width._ `rho(770) --> pi pi = no value`; `rho(770) --> pi+- pi0 = no value`; `rho(770) --> pi+- gamma = 0.00045`; `rho(770) --> pi+ pi- = no value`

### rho+ (`213`), m = 0.775, importance 11.4%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `rho(770) --> pi pi = no value`; `rho(770) --> pi+- pi0 = no value`; `rho(770) --> pi+- gamma = 0.00045`

### K*(892)+ (`323`), m = 0.892, importance 3.8%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `K^*(892) --> K pi = no value`; `K^*(892) --> (K pi)+- = 0.999`; `K^*(892) --> (K pi)0 = 0.9975`

### f2(1270) (`225`), m = 1.275, importance 2.6%

- _PDG cannot constrain these branching ratios: measured modes cover only 11.2% of the width._ `f_2(1270) --> pi pi = 0.8426`; `f_2(1270) --> K Kbar = 0.04573`; `f_2(1270) --> eta pi pi = limit`

### a2(1320)+ (`215`), m = 1.318, importance 1.8%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `a_2(1320) --> 3 pi = 0.7006`; `a_2(1320) --> eta pi = 0.1448`; `a_2(1320) --> omega pi pi = 0.1058`

### K*(892)0 (`313`), m = 0.896, importance 1.6%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.2% of the width._ `K^*(892) --> K pi = no value`; `K^*(892) --> (K pi)+- = 0.999`; `K^*(892) --> (K pi)0 = 0.9975`

### a1(1260)0 (`20113`), m = 1.230, importance 1.4%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `a_1(1260) --> 3 pi = no value`; `a_1(1260) --> ( rho pi )(S-wave) , rho --> pi pi = no value`; `a_1(1260) --> ( rho pi )(D-wave) , rho --> pi pi = no value`; `a_1(1260) --> pi+ pi- pi0 = no value`; `a_1(1260) --> pi0 pi0 pi0 = no value`

### h1(1170) (`10223`), m = 1.170, importance 1.3%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `h_1(1170) --> rho pi = no value`

### a2(1320)0 (`115`), m = 1.318, importance 1.2%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `a_2(1320) --> 3 pi = 0.7006`; `a_2(1320) --> eta pi = 0.1448`; `a_2(1320) --> omega pi pi = 0.1058`

### b1(1235)+ (`10213`), m = 1.230, importance 1.2%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `b_1(1235) --> omega pi = no value`; `b_1(1235) --> pi+- gamma = 0.001615`; `b_1(1235) --> eta rho = no value`; `b_1(1235) --> pi+ pi+ pi- pi0 = no value`

### K2*(1430)+ (`325`), m = 1.426, importance 1.0%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.2% of the width._ `K_2^*(1430) --> K pi = 0.4987`; `K_2^*(1430) --> K^*(892) pi = 0.2471`; `K_2^*(1430) --> K^*(892) pi pi = 0.1345`

### Delta(1232)++ (`2224`), m = 1.232, importance 0.9%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `Delta(1232) --> N pi = 0.994`; `Delta(1232) --> N gamma = limit`; `Delta(1232) --> N gamma, helicity=1/2 = limit`

### b1(1235)0 (`10113`), m = 1.230, importance 0.9%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `b_1(1235) --> omega pi = no value`; `b_1(1235) --> pi+- gamma = 0.001615`; `b_1(1235) --> eta rho = no value`

### f0(1370) (`10221`), m = 1.350, importance 0.8%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `f_0(1370) --> pi pi = no value`; `f_0(1370) --> 4pi = no value`; `f_0(1370) --> rho rho = no value`; `f_0(1370) --> 4pi0 = no value`; `f_0(1370) --> 2pi+ 2pi- = no value`; `f_0(1370) --> pi+ pi- 2pi0 = no value`

### a1(1260)+ (`20213`), m = 1.230, importance 0.8%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `a_1(1260) --> 3 pi = no value`; `a_1(1260) --> ( rho pi )(S-wave) , rho --> pi pi = no value`; `a_1(1260) --> ( rho pi )(D-wave) , rho --> pi pi = no value`

### rho(1450)+ (`100213`), m = 1.465, importance 0.7%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `rho(1450) --> pi pi = no value`; `rho(1450) --> 4pi = no value`; `rho(1450) --> eta rho = no value`

### rho(1450)0 (`100113`), m = 1.465, importance 0.7%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `rho(1450) --> pi pi = no value`; `rho(1450) --> 4pi = no value`; `rho(1450) --> eta rho = no value`; `rho(1450) --> pi+ pi- = no value`; `rho(1450) --> K+ K- = no value`; `rho(1450) --> pi0 gamma = no value`

### rho3(1690)+ (`217`), m = 1.691, importance 0.6%

- **header** mass 1.6910 -> PDG 1.6856
- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `rho_3(1690) --> 4pi = 0.7107`; `rho_3(1690) --> pi+- pi+ pi- pi0 = 0.6743`; `rho_3(1690) --> omega pi = 0.1573`

### eta'(958) (`331`), m = 0.958, importance 0.6%

- _PDG cannot constrain these branching ratios: measured modes cover only 68.4% of the width._ `eta^'(958) --> rho0 gamma (including non-resonant pi+ pi- gamma) = 0.2948`; `eta^'(958) --> omega gamma = 0.02519`; `eta^'(958) --> omega e+ e- = 0.000197`

### f1(1285) (`20223`), m = 1.282, importance 0.6%

- _PDG cannot constrain these branching ratios: measured modes cover only 67.9% of the width._ `f_1(1285) --> 4pi = 0.3272`; `f_1(1285) --> rho0 pi+ pi- = 0.1091`; `f_1(1285) --> rho0 rho0 = no value`; `f_1(1285) --> phi gamma = no value`

### h1(1380) (`10333`), m = 1.386, importance 0.6%

- **header** mass 1.3861 -> PDG 1.4091

### Delta(1232)+ (`2214`), m = 1.232, importance 0.5%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `Delta(1232) --> N pi = 0.994`; `Delta(1232) --> N gamma = limit`; `Delta(1232) --> N gamma, helicity=1/2 = limit`

### Delta(1232)0 (`2114`), m = 1.232, importance 0.5%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `Delta(1232) --> N pi = 0.994`; `Delta(1232) --> N gamma = limit`; `Delta(1232) --> N gamma, helicity=1/2 = limit`

### pi2(1670)+ (`10215`), m = 1.670, importance 0.5%

- _PDG cannot constrain these branching ratios: measured modes cover only 0.0% of the width._ `pi_2(1670) --> 3pi = 0.9578`; `pi_2(1670) --> f_2(1270) pi = 0.5628`; `pi_2(1670) --> rho pi = 0.3082`

### rho3(1690)0 (`117`), m = 1.691, importance 0.5%

- **header** mass 1.6910 -> PDG 1.6856

### K0*(1430)+ (`10321`), m = 1.412, importance 0.3%

- **header** mass 1.4120 -> PDG 1.4250

### omega(1420) (`100223`), m = 1.419, importance 0.3%

- **header** mass 1.4190 -> PDG 1.4100
- **header** width 1.740e-01 -> PDG 2.900e-01 (1x)

### a0(1450)+ (`10211`), m = 1.474, importance 0.3%

- **header** mass 1.4740 -> PDG 1.4387

### a0(1450)0 (`10111`), m = 1.474, importance 0.2%

- **header** mass 1.4740 -> PDG 1.4387

