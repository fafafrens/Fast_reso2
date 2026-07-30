#!/usr/bin/env bash
set -euo pipefail

benchmark_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$benchmark_dir/../.." && pwd)"
reference_binary="${FASTRESO_REFERENCE:-$benchmark_dir/bin/fastreso_reference}"
output_dir="${FASTRESO_CROSSCHECK_OUTPUT:-$benchmark_dir/output}"
julia_binary="${JULIA:-julia}"

if [[ ! -x "$reference_binary" ]]; then
  echo "Missing reference exporter: $reference_binary" >&2
  echo "Build it with: make -C $benchmark_dir FASTRESO_DIR=/path/to/FastReso" >&2
  exit 2
fi

mkdir -p "$output_dir"

common_ids=(223 113 211 -211 111)
"$reference_binary" static \
  "$benchmark_dir/fixtures/rho_only.dat" \
  "$output_dir/rho.static.tsv" \
  0.145 "${common_ids[@]}"
"$reference_binary" static \
  "$benchmark_dir/fixtures/omega_only.dat" \
  "$output_dir/omega.static.tsv" \
  0.145 "${common_ids[@]}"
"$reference_binary" static \
  "$benchmark_dir/fixtures/small_thermal.dat" \
  "$output_dir/combined.static.tsv" \
  0.145 "${common_ids[@]}"

"$julia_binary" --project="$repository_root" --threads=1 \
  "$benchmark_dir/verify_static.jl" \
  "$output_dir/rho.static.tsv" \
  "$output_dir/omega.static.tsv" \
  "$output_dir/combined.static.tsv"

"$reference_binary" static \
  "$benchmark_dir/fixtures/rho_chain.dat" \
  "$output_dir/rho_chain.static.tsv" \
  0.145 30113 113 211 -211
"$julia_binary" --project="$repository_root" --threads=1 \
  "$benchmark_dir/verify_static_chain.jl" \
  "$output_dir/rho_chain.static.tsv"

"$reference_binary" projected \
  "$benchmark_dir/fixtures/rho_only.dat" \
  "$output_dir/rho.projected.tsv" \
  0.145 0.0001 5.0 401 211 -211
"$julia_binary" --project="$repository_root" --threads=auto \
  "$benchmark_dir/verify_projected.jl" \
  "$output_dir/rho.projected.tsv" \
  "$benchmark_dir/fixtures/rho_only.dat"

"$julia_binary" --project="$repository_root" --threads=auto \
  "$benchmark_dir/benchmark_chain_plan.jl"
