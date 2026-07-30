#include <TFastReso_AZYHYDRO.h>
#include <TKernel.h>
#include <gsl/gsl_errno.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum class Representation {
  kStatic,
  kProjected,
};

struct Options {
  Representation representation;
  std::string decay_list;
  std::string output;
  double temperature;
  double minimum_momentum = 0.0;
  double maximum_momentum = 0.0;
  int point_count = 0;
  std::vector<int> target_ids;
};

struct TargetResult {
  std::string name;
  double mass = 0.0;
  double width = 0.0;
  int degeneracy = 0;
  std::string statistic;
  double primary_yield = 0.0;
  double total_yield = 0.0;
  std::vector<double> momenta;
  std::vector<double> primary;
  std::vector<double> total;
};

double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

std::string environment_or(const char *name, const char *fallback) {
  const char *value = std::getenv(name);
  return value == nullptr || value[0] == '\0' ? fallback : value;
}

std::string file_fnv1a64(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot hash decay list: " + path);
  }
  std::uint64_t hash = 14695981039346656037ULL;
  char byte = 0;
  while (input.get(byte)) {
    hash ^= static_cast<unsigned char>(byte);
    hash *= 1099511628211ULL;
  }
  std::ostringstream text;
  text << std::hex << std::setfill('0') << std::setw(16) << hash;
  return text.str();
}

[[noreturn]] void usage(const char *program, const std::string &message = {}) {
  if (!message.empty()) {
    std::cerr << "error: " << message << "\n\n";
  }
  std::cerr
      << "usage:\n"
      << "  " << program
      << " static DECAY_LIST OUTPUT_TSV TEMPERATURE PDG [PDG ...]\n"
      << "  " << program
      << " projected DECAY_LIST OUTPUT_TSV TEMPERATURE QMIN QMAX NPOINTS"
         " PDG [PDG ...]\n\n"
      << "static exports E*dN/d^3p from FastReso Feq2 on its native"
         " pbar grid.\n"
      << "projected exports TKernel mode 0 on a uniform pT grid for direct"
         " CascadePlan input.\n";
  std::exit(2);
}

double parse_double(const char *text, const std::string &name) {
  try {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (text[consumed] != '\0' || !std::isfinite(value)) {
      throw std::invalid_argument("trailing or non-finite value");
    }
    return value;
  } catch (const std::exception &) {
    throw std::runtime_error("invalid " + name + ": " + text);
  }
}

int parse_int(const char *text, const std::string &name) {
  try {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (text[consumed] != '\0') {
      throw std::invalid_argument("trailing value");
    }
    return value;
  } catch (const std::exception &) {
    throw std::runtime_error("invalid " + name + ": " + text);
  }
}

Options parse_options(int argc, char **argv) {
  if (argc < 6) {
    usage(argv[0]);
  }

  Options options;
  const std::string representation = argv[1];
  if (representation == "static") {
    options.representation = Representation::kStatic;
  } else if (representation == "projected") {
    options.representation = Representation::kProjected;
  } else {
    usage(argv[0], "representation must be static or projected");
  }

  options.decay_list = argv[2];
  options.output = argv[3];
  options.temperature = parse_double(argv[4], "temperature");
  if (options.temperature <= 0.0) {
    usage(argv[0], "temperature must be positive");
  }

  int first_id = 5;
  if (options.representation == Representation::kProjected) {
    if (argc < 9) {
      usage(argv[0], "projected mode needs QMIN QMAX NPOINTS and a PDG ID");
    }
    options.minimum_momentum = parse_double(argv[5], "QMIN");
    options.maximum_momentum = parse_double(argv[6], "QMAX");
    options.point_count = parse_int(argv[7], "NPOINTS");
    if (!(options.minimum_momentum > 0.0 &&
          options.maximum_momentum > options.minimum_momentum &&
          options.point_count >= 2)) {
      usage(argv[0], "projected grid needs 0 < QMIN < QMAX and NPOINTS >= 2");
    }
    first_id = 8;
  }

  std::map<int, bool> seen_ids;
  for (int index = first_id; index < argc; ++index) {
    const int id = parse_int(argv[index], "PDG ID");
    if (!seen_ids[id]) {
      options.target_ids.push_back(id);
      seen_ids[id] = true;
    }
  }
  if (options.target_ids.empty()) {
    usage(argv[0], "at least one target PDG ID is required");
  }
  return options;
}

std::vector<double> uniform_grid(double minimum, double maximum, int count) {
  std::vector<double> grid(count);
  for (int index = 0; index < count; ++index) {
    grid[index] =
        minimum + (maximum - minimum) * index / static_cast<double>(count - 1);
  }
  return grid;
}

double invariant_spectrum(TParticle *particle, int grid_index) {
  const double momentum = particle->getPbar(grid_index);
  const double energy = particle->getEbar(grid_index);
  const double regularized_feq2 = particle->get_Fj(1, energy);
  const double phase_space = std::pow(2.0 * std::acos(-1.0), 3);
  return particle->getNu() * energy * regularized_feq2 /
         (phase_space * momentum);
}

double phase_space_factor() {
  return std::pow(2.0 * std::acos(-1.0), 3);
}

std::vector<double> static_spectrum(TParticle *particle) {
  std::vector<double> result;
  result.reserve(particle->getNpbar());
  for (int index = 0; index < particle->getNpbar(); ++index) {
    result.push_back(invariant_spectrum(particle, index));
  }
  return result;
}

std::vector<double> native_grid(TParticle *particle) {
  std::vector<double> result;
  result.reserve(particle->getNpbar());
  for (int index = 0; index < particle->getNpbar(); ++index) {
    result.push_back(particle->getPbar(index));
  }
  return result;
}

std::vector<double> projected_spectrum(
    TParticle *particle,
    const std::vector<double> &momenta
) {
  TKernel kernel(particle);
  std::vector<double> result;
  result.reserve(momenta.size());
  for (const double momentum : momenta) {
    result.push_back(
        kernel.get_K_mu_qag(momentum, 0.0, 0) / phase_space_factor()
    );
  }
  return result;
}

std::map<int, TParticle *> particle_index(TFastReso_AZYHYDRO &fastreso) {
  std::map<int, TParticle *> particles;
  for (int index = 0; index < fastreso.getNumberOfParticles(); ++index) {
    TParticle *particle = fastreso.getParticle(index);
    const auto inserted = particles.emplace(particle->getPDG(), particle);
    if (!inserted.second) {
      throw std::runtime_error(
          "duplicate PDG ID " + std::to_string(particle->getPDG())
      );
    }
  }
  return particles;
}

const char *statistic_name(EParticleType type) {
  switch (type) {
    case EParticleType::kBoson:
      return "boson";
    case EParticleType::kFermion:
      return "fermion";
    case EParticleType::kBoltzman:
      return "boltzmann";
  }
  return "unknown";
}

void check_spectrum(
    const std::vector<double> &values,
    int particle_id,
    const std::string &stage
) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      throw std::runtime_error(
          "non-finite " + stage + " spectrum for PDG " +
          std::to_string(particle_id) + " at point " +
          std::to_string(index)
      );
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    gsl_set_error_handler_off();

    std::map<int, TargetResult> results;
    for (const int id : options.target_ids) {
      results.emplace(id, TargetResult{});
    }

    const std::vector<double> projected_grid =
        options.representation == Representation::kProjected
            ? uniform_grid(
                  options.minimum_momentum,
                  options.maximum_momentum,
                  options.point_count
              )
            : std::vector<double>{};

    // Use separate, scoped objects for primary and total export. get_Fj and
    // TKernel initialize splines lazily; exporting selected primaries on the
    // total object would make the decay timing depend on the selected IDs.
    Clock::time_point primary_setup_start;
    Clock::time_point primary_setup_stop;
    Clock::time_point primary_start;
    Clock::time_point primary_stop;
    {
      primary_setup_start = Clock::now();
      TFastReso_AZYHYDRO primary_fastreso;
      primary_fastreso.read_particles_data(
          options.decay_list,
          "Feq",
          false
      );
      primary_fastreso.do_thermal(options.temperature);
      primary_setup_stop = Clock::now();

      const auto primary_particles = particle_index(primary_fastreso);
      primary_start = Clock::now();
      for (const int id : options.target_ids) {
        const auto found = primary_particles.find(id);
        if (found == primary_particles.end()) {
          throw std::runtime_error(
              "target PDG ID " + std::to_string(id) +
              " is absent from the list"
          );
        }
        TParticle *particle = found->second;
        TargetResult &target = results[id];
        target.name = particle->getName();
        target.mass = particle->getM();
        target.width = particle->getGamma();
        target.degeneracy = particle->getNu();
        target.statistic = statistic_name(particle->getType());
        target.primary_yield = particle->getN();
        if (options.representation == Representation::kStatic) {
          target.momenta = native_grid(particle);
          target.primary = static_spectrum(particle);
        } else {
          target.momenta = projected_grid;
          target.primary = projected_spectrum(particle, projected_grid);
        }
        check_spectrum(target.primary, id, "primary");
      }
      primary_stop = Clock::now();
    }

    Clock::time_point total_setup_start;
    Clock::time_point total_setup_stop;
    Clock::time_point decay_start;
    Clock::time_point decay_stop;
    Clock::time_point total_start;
    Clock::time_point total_stop;
    {
      total_setup_start = Clock::now();
      TFastReso_AZYHYDRO total_fastreso;
      total_fastreso.read_particles_data(options.decay_list, "Feq", false);
      total_fastreso.do_thermal(options.temperature);
      total_setup_stop = Clock::now();
      const auto total_particles = particle_index(total_fastreso);
      for (const int id : options.target_ids) {
        if (total_particles.find(id) == total_particles.end()) {
          throw std::runtime_error(
              "target PDG ID " + std::to_string(id) +
              " disappeared in the total instance"
          );
        }
      }

      decay_start = Clock::now();
      total_fastreso.do_decays(options.decay_list, false);
      decay_stop = Clock::now();

      total_start = Clock::now();
      for (const int id : options.target_ids) {
        TParticle *particle = total_particles.at(id);
        TargetResult &target = results[id];
        target.total_yield = particle->getN();
        target.total =
            options.representation == Representation::kStatic
                ? static_spectrum(particle)
                : projected_spectrum(particle, projected_grid);
        check_spectrum(target.total, id, "total");
        if (!std::isfinite(target.primary_yield) ||
            !std::isfinite(target.total_yield)) {
          throw std::runtime_error(
              "non-finite yield for PDG " + std::to_string(id)
          );
        }
      }
      total_stop = Clock::now();
    }

    std::ofstream output(options.output);
    if (!output) {
      throw std::runtime_error("cannot open output file: " + options.output);
    }
    output << std::scientific << std::setprecision(17);
    output << "# fastreso_crosscheck_version 1\n";
    output << "# representation "
           << (options.representation == Representation::kStatic
                   ? "static"
                   : "projected")
           << '\n';
    output << "# temperature " << options.temperature << '\n';
    output << "# decay_list " << options.decay_list << '\n';
    output << "# decay_fnv1a64 " << file_fnv1a64(options.decay_list) << '\n';
    output << "# upstream_revision "
           << environment_or("FASTRESO_UPSTREAM_REVISION", "unknown") << '\n';
    output << "# upstream_patches "
           << environment_or("FASTRESO_UPSTREAM_PATCHES", "unknown") << '\n';
    output << "# compiler " << __VERSION__ << '\n';
    output << "# components Feq\n";
    output << "# native_npbar " << grid_params::fNpbar << '\n';
    output << "# native_pbar_max " << grid_params::fPbarMax << '\n';
    output << "# qag_epsrel " << qag_params::fEpsRel << '\n';
    output << "# qag_epsabs " << qag_params::fEpsAbs << '\n';
    output << "# grid "
           << (options.representation == Representation::kStatic
                   ? "native_tangent"
                   : "uniform")
           << '\n';
    output << "# spectrum invariant_E_dN_d3p\n";
    output << "# normalization one_over_2pi_cubed\n";
    output << "# topology two_and_three_body\n";
    output << "# width_policy ignored_sharp\n";
    output << "# yield_semantics inclusive_density\n";
    if (options.representation == Representation::kProjected) {
      output << "# eta_max 5.0\n";
    }
    output << "# point_count " << results.begin()->second.momenta.size() << '\n';
    output << "# timing_ms primary_setup "
           << elapsed_ms(primary_setup_start, primary_setup_stop)
           << " primary " << elapsed_ms(primary_start, primary_stop)
           << " total_setup "
           << elapsed_ms(total_setup_start, total_setup_stop)
           << " decay " << elapsed_ms(decay_start, decay_stop)
           << " total " << elapsed_ms(total_start, total_stop) << '\n';
    output << "# columns pdg momentum primary total\n";
    for (const int id : options.target_ids) {
      const TargetResult &target = results.at(id);
      output << "# particle " << id << ' '
             << target.name << ' '
             << target.mass << ' '
             << target.width << ' '
             << target.degeneracy << ' '
             << target.statistic << '\n';
      output << "# yield " << id << ' ' << target.primary_yield << ' '
             << target.total_yield << '\n';
    }
    for (const int id : options.target_ids) {
      const TargetResult &target = results.at(id);
      for (std::size_t index = 0; index < target.momenta.size(); ++index) {
        output << id << ' ' << target.momenta[index] << ' '
               << target.primary[index] << ' ' << target.total[index] << '\n';
      }
    }

    std::clog << std::fixed << std::setprecision(3)
              << "timing_ms primary_setup="
              << elapsed_ms(primary_setup_start, primary_setup_stop)
              << " primary=" << elapsed_ms(primary_start, primary_stop)
              << " total_setup="
              << elapsed_ms(total_setup_start, total_setup_stop)
              << " decay=" << elapsed_ms(decay_start, decay_stop)
              << " total=" << elapsed_ms(total_start, total_stop) << '\n';
    return output ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "fastreso_reference: " << error.what() << '\n';
    return 1;
  }
}
