import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import gen_heavy_flavour as generator


class FakeParticle:
    def __init__(
        self,
        mcid,
        name,
        mass,
        width,
        charge,
        spin,
        isospin,
        *,
        self_conjugate=True,
        modes=(),
    ):
        self.mcid = mcid
        self.name = name
        self.mass = mass
        self.width = width
        self.charge = charge
        self.quantum_J = spin
        self.quantum_I = isospin
        self.self_conjugate = self_conjugate
        self._modes = list(modes)

    def exclusive_branching_fractions(self):
        return self._modes


class MissingMassParticle(FakeParticle):
    def __init__(self, *args, **kwargs):
        mass = kwargs.pop("mass", None)
        super().__init__(*args, mass=mass, **kwargs)
        del self.mass

    @property
    def mass(self):
        raise generator.PdgNoDataError("no mass summary")

    @mass.setter
    def mass(self, value):
        self._ignored_mass = value

    @mass.deleter
    def mass(self):
        pass


class FakeItem:
    def __init__(self, particle):
        self.name = particle.name
        self.has_particle = True
        self.particle = particle


class FakeDecayProduct:
    def __init__(self, particle, multiplier=1):
        self.item = FakeItem(particle)
        self.multiplier = multiplier


class FakeBranchingFraction:
    def __init__(
        self,
        value,
        daughters,
        *,
        limit=False,
        description="mode",
        value_text=None,
    ):
        self.value = value
        self.value_text = value_text
        self.decay_products = [
            FakeDecayProduct(daughter) for daughter in daughters
        ]
        self.is_limit = limit
        self.description = description


class FakeApi:
    def __init__(self, particles):
        self.particles = list(particles)
        self.by_id = {particle.mcid: particle for particle in particles}

    def get_particles(self):
        return self.particles

    def get_particle_by_mcid(self, mcid):
        return self.by_id[mcid]


def particle_data(
    mcid,
    name,
    mass,
    *,
    width=0.0,
    degeneracy=1,
    baryon=0,
    strangeness=0,
    charm=0,
    bottomness=0,
    isospin=0.0,
    charge=0,
):
    return generator.ParticleData(
        mcid,
        name,
        mass,
        width,
        degeneracy,
        baryon,
        strangeness,
        charm,
        bottomness,
        isospin,
        charge,
    )


class GeneratorUnitTests(unittest.TestCase):
    provenance = generator.PdgProvenance(
        package_version="2026.0",
        edition="2026",
        data_release="test-release",
        schema_version="test-schema",
        license="CC BY 4.0",
        citation="Test citation",
    )

    def test_flavour_quantum_numbers(self):
        self.assertEqual(generator.flavour(411), (0, 0, 1, 0))
        self.assertEqual(generator.flavour(-411), (0, 0, -1, 0))
        self.assertEqual(generator.flavour(431), (0, 1, 1, 0))
        self.assertEqual(generator.flavour(511), (0, 0, 0, 1))
        self.assertEqual(generator.flavour(5122), (1, 0, 0, -1))
        self.assertEqual(generator.flavour(-5122), (-1, 0, 0, 1))
        self.assertEqual(generator.flavour(443), (0, 0, 0, 0))
        self.assertTrue(generator.heavy(443))
        self.assertFalse(generator.heavy(211))

    def test_quantum_fields_use_the_current_pdg_attribute_names(self):
        particle = FakeParticle(443, "J/psi", 3.1, 0.001, 0, "3/2", "1")
        report = generator.GenerationReport()
        properties = generator.ParticleProperties({443: particle}, report)
        data = properties.particle_data(particle)
        self.assertEqual(data.degeneracy, 4)
        self.assertEqual(data.isospin, 1.0)

    def test_neutral_sigma_b_uses_explicit_isospin_estimates(self):
        sigma_minus = FakeParticle(
            5112, "Sigma_b-", 5.8156367005, 0.0053137290, -1, "1/2", "1"
        )
        sigma_plus = FakeParticle(
            5222, "Sigma_b+", 5.8105633721, 0.0049522353, 1, "1/2", "1"
        )
        sigma_zero = MissingMassParticle(
            5212,
            "Sigma_b0",
            width=0.0,
            charge=0,
            spin="1/2",
            isospin="1",
        )
        particles = {
            particle.mcid: particle
            for particle in (sigma_minus, sigma_plus, sigma_zero)
        }
        report = generator.GenerationReport()
        properties = generator.ParticleProperties(particles, report)

        self.assertAlmostEqual(
            properties.mass(sigma_zero), 5.8131000363, places=9
        )
        self.assertAlmostEqual(
            properties.width_info(sigma_zero)[0], 0.00513298215, places=11
        )
        self.assertIn((5212, "mass"), report.estimates)
        self.assertIn((5212, "width"), report.estimates)

    def test_an_unexpected_missing_mass_is_an_error(self):
        particle = MissingMassParticle(
            413,
            "Dstar",
            width=0.001,
            charge=1,
            spin="1",
            isospin="1/2",
        )
        properties = generator.ParticleProperties(
            {413: particle}, generator.GenerationReport()
        )
        with self.assertRaises(generator.GenerationError):
            properties.mass(particle)

    @staticmethod
    def branching_fixture(policy):
        pion_plus = FakeParticle(211, "pi+", 0.139, 0.0, 1, "0", "1")
        pion_minus = FakeParticle(-211, "pi-", 0.139, 0.0, -1, "0", "1")
        photon = FakeParticle(22, "gamma", 0.0, 0.0, 0, "1", "0")
        pion_zero = FakeParticle(111, "pi0", 0.135, 0.0, 0, "0", "1")
        parent = FakeParticle(443, "J/psi", 3.1, 0.1, 0, "1", "0")
        parent._modes = [
            FakeBranchingFraction(
                0.2, (pion_plus, pion_minus), description="measured hadrons"
            ),
            FakeBranchingFraction(
                0.3, (photon, photon), description="measured photons"
            ),
            FakeBranchingFraction(
                0.8,
                (pion_plus, pion_minus),
                limit=True,
                description="upper limit",
            ),
            FakeBranchingFraction(
                None, (pion_zero, pion_zero), description="seen only"
            ),
        ]
        main = {
            pion.mcid: particle_data(
                pion.mcid, pion.name, pion.mass, charge=round(pion.charge)
            )
            for pion in (pion_plus, pion_minus, photon, pion_zero)
        }
        return generator.HeavyFlavourGenerator(
            FakeApi([parent]), main, branching_policy=policy
        ), parent

    def test_absolute_branching_ratios_are_not_renormalized(self):
        instance, parent = self.branching_fixture("absolute")
        channels = dict(
            (daughters, branching_ratio)
            for branching_ratio, daughters in instance.resolve_channels(parent)
        )
        self.assertEqual(channels[(211, -211)], 0.2)
        self.assertEqual(channels[(22, 22)], 0.3)
        self.assertEqual(instance.report.dropped_modes["limit"], 1)
        self.assertEqual(
            instance.report.dropped_modes["unquantified-branching-ratio"], 1
        )
        self.assertEqual(instance.report.branching_totals[443], 0.5)

    def test_conditional_policy_is_explicit(self):
        instance, parent = self.branching_fixture("conditional")
        channels = dict(
            (daughters, branching_ratio)
            for branching_ratio, daughters in instance.resolve_channels(parent)
        )
        self.assertAlmostEqual(channels[(211, -211)], 0.4)
        self.assertAlmostEqual(channels[(22, 22)], 0.6)
        self.assertIn(443, instance.report.conditional_parents)

    def test_relative_branching_ratio_tables_are_not_probabilities(self):
        instance, parent = self.branching_fixture("absolute")
        pion_plus = FakeParticle(211, "pi+", 0.139, 0.0, 1, "0", "1")
        pion_minus = FakeParticle(-211, "pi-", 0.139, 0.0, -1, "0", "1")
        photon = FakeParticle(22, "gamma", 0.0, 0.0, 0, "1", "0")
        parent._modes = [
            FakeBranchingFraction(
                None,
                (pion_plus, pion_minus),
                value_text="DEFINED AS 1",
            ),
            FakeBranchingFraction(0.25, (photon, photon)),
        ]
        self.assertEqual(instance.resolve_channels(parent), ())
        self.assertEqual(
            instance.report.dropped_modes["relative-branching-ratio"], 2
        )

    def test_malformed_numerical_branching_ratio_is_an_error(self):
        branching_fraction = FakeBranchingFraction(
            float("nan"),
            (),
            description="bad mode",
        )
        with self.assertRaisesRegex(generator.GenerationError, "bad mode"):
            generator.HeavyFlavourGenerator._known_branching_value(
                branching_fraction
            )

    def test_atomic_writer_replaces_complete_content(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "supplement.dat"
            generator.write_atomic(output, "first\n")
            generator.write_atomic(output, "second\n")
            self.assertEqual(output.read_text(), "second\n")
            self.assertEqual(output.stat().st_mode & 0o777, 0o644)

    def test_terminal_channel_cannot_be_mixed_with_decays(self):
        parent = particle_data(443, "J/psi", 3.1)
        pion_plus = particle_data(211, "pi+", 0.139, charge=1)
        pion_minus = particle_data(-211, "pi-", 0.139, charge=-1)
        entry = generator.Entry(
            parent,
            (
                (1.0, (443,)),
                (0.2, (211, -211)),
            ),
        )
        with self.assertRaises(generator.GenerationError):
            generator.validate_entries(
                [entry], {211: pion_plus, -211: pion_minus}
            )

    def test_rendered_rounding_is_revalidated(self):
        parent = particle_data(443, "parent", 1.000004)
        daughter = particle_data(111, "daughter", 0.5)
        entry = generator.Entry(parent, ((1.0, (111, 111)),))
        main = {111: daughter}

        # Full precision is open, but the five-decimal on-disk parent mass
        # rounds below the parser's 1e-6 threshold margin.
        generator.validate_entries([entry], main)
        content = generator.render(
            [entry],
            generator.GenerationReport(),
            provenance=self.provenance,
            branching_policy="absolute",
        )
        self.assertIn(
            "# PDG data release=test-release, schema=test-schema, "
            "license=CC BY 4.0.",
            content,
        )
        rendered_entries = generator.parse_rendered_entries(content)
        with self.assertRaises(generator.GenerationError):
            generator.validate_entries(rendered_entries, main)

    def test_build_conjugates_antiparticle_channels(self):
        d_zero = FakeParticle(421, "D0", 1.86, 0.0, 0, "0", "1/2")
        dbar_zero = FakeParticle(-421, "Dbar0", 1.86, 0.0, 0, "0", "1/2")
        pion_plus = FakeParticle(211, "pi+", 0.139, 0.0, 1, "0", "1")
        pion_minus = FakeParticle(-211, "pi-", 0.139, 0.0, -1, "0", "1")
        parent = FakeParticle(
            413,
            "Dstar+",
            2.01,
            0.001,
            1,
            "1",
            "1/2",
            self_conjugate=False,
        )
        antiparticle = FakeParticle(
            -413,
            "Dstar-",
            2.01,
            0.001,
            -1,
            "1",
            "1/2",
            self_conjugate=False,
        )
        parent._modes = [
            FakeBranchingFraction(0.7, (d_zero, pion_plus))
        ]
        main = {
            particle.mcid: particle_data(
                particle.mcid,
                particle.name,
                particle.mass,
                charm=1 if particle.mcid == 421 else (
                    -1 if particle.mcid == -421 else 0
                ),
                charge=round(particle.charge),
            )
            for particle in (d_zero, dbar_zero, pion_plus, pion_minus)
        }
        instance = generator.HeavyFlavourGenerator(
            FakeApi([parent, antiparticle]), main
        )
        entries = {
            entry.particle.id: entry for entry in instance.build()
        }
        self.assertEqual(entries[413].channels, ((0.7, (421, 211)),))
        self.assertEqual(entries[-413].channels, ((0.7, (-421, -211)),))

    def test_both_antiparticle_directions_are_required(self):
        for mcid in (413, -413):
            particle = FakeParticle(
                mcid,
                "Dstar",
                2.01,
                0.001,
                1 if mcid > 0 else -1,
                "1",
                "1/2",
                self_conjugate=False,
            )
            with self.assertRaises(generator.GenerationError):
                generator.HeavyFlavourGenerator(FakeApi([particle]), {})


class CheckedInArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.supplement = ROOT / "decays_charm_bottom.dat"
        cls.main = ROOT / "decays_QM2016Plus_massorder.dat"
        cls.headers = {}
        cls.header_order = []
        cls.channels = {}
        declared_counts = {}

        for line_number, line in enumerate(
            cls.supplement.read_text().splitlines(), start=1
        ):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) == 12:
                mcid = int(fields[0])
                if mcid in cls.headers:
                    raise AssertionError(f"duplicate supplement ID {mcid}")
                cls.header_order.append(mcid)
                cls.headers[mcid] = {
                    "name": fields[1],
                    "mass": float(fields[2]),
                    "width": float(fields[3]),
                    "degeneracy": int(fields[4]),
                    "baryon": int(fields[5]),
                    "strangeness": int(fields[6]),
                    "charm": int(fields[7]),
                    "bottomness": int(fields[8]),
                    "isospin": float(fields[9]),
                    "charge": int(fields[10]),
                }
                declared_counts[mcid] = int(fields[11])
                cls.channels[mcid] = []
            elif len(fields) == 8:
                parent = int(fields[0])
                if parent not in cls.headers:
                    raise AssertionError(
                        f"line {line_number} precedes parent header {parent}"
                    )
                daughter_count = int(fields[1])
                branching_ratio = float(fields[2])
                slots = [int(value) for value in fields[3:]]
                if not all(
                    value == 0 for value in slots[daughter_count:]
                ):
                    raise AssertionError(
                        f"line {line_number} has nonzero padding"
                    )
                cls.channels[parent].append(
                    (branching_ratio, tuple(slots[:daughter_count]))
                )
            else:
                raise AssertionError(
                    f"{cls.supplement}:{line_number}: malformed row"
                )

        for mcid, expected in declared_counts.items():
            actual = len(cls.channels[mcid])
            if actual != expected:
                raise AssertionError(
                    f"{mcid}: declared {expected} channels, found {actual}"
                )

        cls.main_headers = {}
        for line in cls.main.read_text().splitlines():
            fields = line.split()
            if len(fields) != 12:
                continue
            cls.main_headers[int(fields[0])] = {
                "name": fields[1],
                "mass": float(fields[2]),
                "width": float(fields[3]),
                "degeneracy": round(float(fields[4])),
                "baryon": round(float(fields[5])),
                "strangeness": round(float(fields[6])),
                "charm": round(float(fields[7])),
                "bottomness": round(float(fields[8])),
                "isospin": float(fields[9]),
                "charge": round(float(fields[10])),
            }

    def test_particle_metadata_is_physical(self):
        self.assertEqual(len(self.headers), 150)
        self.assertTrue(all(data["mass"] > 0 for data in self.headers.values()))
        self.assertTrue(all(data["width"] >= 0 for data in self.headers.values()))
        self.assertTrue(
            all(data["degeneracy"] > 0 for data in self.headers.values())
        )
        self.assertAlmostEqual(self.headers[5212]["mass"], 5.81310)
        self.assertAlmostEqual(self.headers[5214]["mass"], 5.83253)
        self.assertEqual(self.headers[413]["degeneracy"], 3)
        self.assertEqual(self.headers[4122]["degeneracy"], 2)
        self.assertEqual(self.headers[4222]["isospin"], 1)

    def test_order_is_deterministic(self):
        masses = [self.headers[mcid]["mass"] for mcid in self.header_order]
        self.assertEqual(masses, sorted(masses, reverse=True))

    def test_channels_resolve_and_conserve_quantum_numbers(self):
        self.assertFalse(set(self.headers) & set(self.main_headers))
        particles = dict(self.main_headers)
        particles.update(self.headers)

        for parent_id, channels in self.channels.items():
            parent = particles[parent_id]
            if any(len(daughters) == 1 for _, daughters in channels):
                self.assertEqual(len(channels), 1)
            nonself_sum = 0.0
            for branching_ratio, daughters in channels:
                self.assertGreater(branching_ratio, 0)
                self.assertLessEqual(branching_ratio, 1)
                if len(daughters) == 1:
                    self.assertEqual(daughters, (parent_id,))
                    self.assertEqual(branching_ratio, 1)
                    continue

                self.assertIn(len(daughters), (2, 3, 4))
                self.assertTrue(all(daughter in particles for daughter in daughters))
                nonself_sum += branching_ratio
                daughter_data = [particles[daughter] for daughter in daughters]
                for field in (
                    "baryon",
                    "strangeness",
                    "charm",
                    "bottomness",
                    "charge",
                ):
                    self.assertEqual(
                        sum(data[field] for data in daughter_data),
                        parent[field],
                        f"{parent_id} -> {daughters} violates {field}",
                    )

                threshold = (
                    math.fsum(data["mass"] for data in daughter_data) + 1e-6
                )
                self.assertTrue(
                    parent["mass"] > threshold
                    or (
                        parent["width"] > generator.WIDTH_CUT
                        and parent["mass"]
                        + generator.WIDTH_WINDOW * parent["width"]
                        > threshold
                    ),
                    f"{parent_id} -> {daughters} is closed",
                )
            self.assertLessEqual(nonself_sum, 1 + 2e-8)

    def test_limits_are_not_turned_into_probabilities(self):
        lambda_c_2595 = self.channels[14122]
        self.assertEqual(
            lambda_c_2595,
            [
                (0.595, (4122, 111, 111)),
                (0.405, (4122, 211, -211)),
            ],
        )

        ds_star_total = sum(
            branching_ratio
            for branching_ratio, daughters in self.channels[433]
            if len(daughters) > 1
        )
        self.assertAlmostEqual(ds_star_total, 0.9932641698)
        self.assertLess(ds_star_total, 1)


if __name__ == "__main__":
    unittest.main()
