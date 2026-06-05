#include "model.hh"
#include "worker.hh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr float kDefaultAbsTolerance = 1e-5f;
constexpr float kDefaultRelTolerance = 1e-5f;
constexpr float kRankPerturbationScale = 1e-3f;

AutoencoderModel make_rank_variant(AutoencoderModel base, int rank) {
  auto weights = base.get_weights();
  const float rank_scale = (rank + 1) * kRankPerturbationScale;
  size_t idx = 0;
  for (auto &kv : weights) {
    kv.second.array() += rank_scale * static_cast<float>(idx + 1);
    ++idx;
  }
  base.set_weights(weights);
  return base;
}

template <typename UpdateFn>
double benchmark_weight_updates(const std::string &label,
                                AutoencoderModel &model, int world_size,
                                int iterations, int warmup, int rank,
                                UpdateFn update_fn) {
  if (rank == 0) {
    std::cout << "\n=== Benchmarking " << label << " ===\n";
  }

  for (int i = 0; i < warmup; ++i)
    update_fn(model, 0, world_size, false);

  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  for (int i = 0; i < iterations; ++i)
    update_fn(model, 0, world_size, false);
  MPI_Barrier(MPI_COMM_WORLD);

  const double elapsed = MPI_Wtime() - start;
  const double avg_time = elapsed / static_cast<double>(iterations);

  if (rank == 0) {
    std::cout << "Total time: " << elapsed << " s\n";
    std::cout << "Avg time per call: " << avg_time << " s\n";
  }

  return avg_time;
}

struct CorrectnessSummary {
  bool passed;
  float max_abs_diff;
  float max_rel_diff;
};

struct BenchmarkOutputConfig {
  std::string out_file;
  std::string out_format = "json";
};

struct BenchmarkReport {
  int world_size = 0;
  int warmup_iterations = 0;
  int bench_iterations = 0;
  double time_multiple = 0.0;
  double time_single = 0.0;
  double overhead_pct = 0.0;
  bool correctness_passed = false;
  float max_abs_diff = 0.0f;
  float max_rel_diff = 0.0f;
  size_t total_params = 0;
  size_t msg_size_bytes = 0;
  double total_traffic_mb = 0.0;
  double bandwidth_mb_s = 0.0;
  double bandwidth_gb_s = 0.0;
  std::optional<double> speedup;
  std::optional<double> parallel_efficiency;
  std::optional<double> serial_fraction;
};

std::string json_escape(const std::string &value) {
  std::ostringstream escaped;
  for (char ch : value) {
    switch (ch) {
    case '\\':
      escaped << "\\\\";
      break;
    case '"':
      escaped << "\\\"";
      break;
    case '\b':
      escaped << "\\b";
      break;
    case '\f':
      escaped << "\\f";
      break;
    case '\n':
      escaped << "\\n";
      break;
    case '\r':
      escaped << "\\r";
      break;
    case '\t':
      escaped << "\\t";
      break;
    default:
      escaped << ch;
      break;
    }
  }
  return escaped.str();
}

BenchmarkOutputConfig parse_benchmark_output_config(int argc, char *argv[]) {
  BenchmarkOutputConfig config;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--benchmark_out" && i + 1 < argc && argv[i + 1][0] != '-') {
      config.out_file = argv[++i];
      continue;
    }
    if (arg.rfind("--benchmark_out=", 0) == 0) {
      config.out_file = arg.substr(std::string("--benchmark_out=").size());
      continue;
    }
    if (arg == "--benchmark_out_format" && i + 1 < argc &&
        argv[i + 1][0] != '-') {
      config.out_format = argv[++i];
      continue;
    }
    if (arg.rfind("--benchmark_out_format=", 0) == 0) {
      config.out_format =
          arg.substr(std::string("--benchmark_out_format=").size());
      continue;
    }
  }

  if (config.out_format.empty())
    config.out_format = "json";

  return config;
}

void write_json_number(std::ostream &out, double value) { out << value; }
void write_json_number(std::ostream &out, float value) { out << value; }
void write_json_number(std::ostream &out, size_t value) { out << value; }
void write_json_number(std::ostream &out, int value) { out << value; }

void write_json_optional_number(std::ostream &out,
                                const std::optional<double> &value) {
  if (value.has_value()) {
    write_json_number(out, value.value());
  } else {
    out << "null";
  }
}

void write_benchmark_report_json(const BenchmarkOutputConfig &config,
                                 const BenchmarkReport &report) {
  if (config.out_file.empty())
    return;

  if (config.out_format != "json") {
    std::cerr << "Unsupported benchmark_out_format '" << config.out_format
              << "'; expected 'json'. Skipping JSON output.\n";
    return;
  }

  const std::filesystem::path out_path(config.out_file);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }

  std::ofstream out(config.out_file);
  if (!out) {
    std::cerr << "Failed to open benchmark output file '" << config.out_file
              << "'\n";
    return;
  }

  out << std::setprecision(17);
  out << "{\n";
  out << "  \"context\": {\n";
  out << "    \"world_size\": ";
  write_json_number(out, report.world_size);
  out << ",\n";
  out << "    \"warmup_iterations\": ";
  write_json_number(out, report.warmup_iterations);
  out << ",\n";
  out << "    \"bench_iterations\": ";
  write_json_number(out, report.bench_iterations);
  out << "\n";
  out << "  },\n";
  out << "  \"benchmarks\": [\n";
  out << "    {\n";
  out << "      \"name\": \"" << json_escape("MULTIPLE Allreduce (per matrix)")
      << "\",\n";
  out << "      \"avg_time_s\": ";
  write_json_number(out, report.time_multiple);
  out << ",\n";
  out << "      \"total_time_s\": ";
  write_json_number(out, report.time_multiple * report.bench_iterations);
  out << "\n";
  out << "    },\n";
  out << "    {\n";
  out << "      \"name\": \"" << json_escape("SINGLE Allreduce (flattened)")
      << "\",\n";
  out << "      \"avg_time_s\": ";
  write_json_number(out, report.time_single);
  out << ",\n";
  out << "      \"total_time_s\": ";
  write_json_number(out, report.time_single * report.bench_iterations);
  out << "\n";
  out << "    }\n";
  out << "  ],\n";
  out << "  \"analysis\": {\n";
  out << "    \"speedup\": ";
  write_json_optional_number(out, report.speedup);
  out << ",\n";
  out << "    \"overhead_pct\": ";
  write_json_number(out, report.overhead_pct);
  out << ",\n";
  out << "    \"parallel_efficiency\": ";
  write_json_optional_number(out, report.parallel_efficiency);
  out << ",\n";
  out << "    \"serial_fraction\": ";
  write_json_optional_number(out, report.serial_fraction);
  out << ",\n";
  out << "    \"correctness_passed\": "
      << (report.correctness_passed ? "true" : "false") << ",\n";
  out << "    \"max_abs_diff\": ";
  write_json_number(out, report.max_abs_diff);
  out << ",\n";
  out << "    \"max_rel_diff\": ";
  write_json_number(out, report.max_rel_diff);
  out << "\n";
  out << "  },\n";
  out << "  \"bandwidth\": {\n";
  out << "    \"total_params\": ";
  write_json_number(out, report.total_params);
  out << ",\n";
  out << "    \"msg_size_bytes\": ";
  write_json_number(out, report.msg_size_bytes);
  out << ",\n";
  out << "    \"total_traffic_mb\": ";
  write_json_number(out, report.total_traffic_mb);
  out << ",\n";
  out << "    \"bandwidth_mb_s\": ";
  write_json_number(out, report.bandwidth_mb_s);
  out << ",\n";
  out << "    \"bandwidth_gb_s\": ";
  write_json_number(out, report.bandwidth_gb_s);
  out << "\n";
  out << "  }\n";
  out << "}\n";
}

CorrectnessSummary
verify_correctness(AutoencoderModel &reference_model,
                   AutoencoderModel &candidate_model,
                   float abs_tolerance = kDefaultAbsTolerance,
                   float rel_tolerance = kDefaultRelTolerance, int rank = 0) {
  auto ref = reference_model.get_weights();
  auto cand = candidate_model.get_weights();

  int local_ok = 1;
  float local_max_abs = 0.0f;
  float local_max_rel = 0.0f;

  if (ref.size() != cand.size()) {
    local_ok = 0;
    if (rank == 0)
      std::cout << "FAIL: Different number of weight matrices\n";
  }

  for (const auto &kv : ref) {
    const auto &name = kv.first;
    const auto &rmat = kv.second;
    auto it = cand.find(name);
    if (it == cand.end()) {
      local_ok = 0;
      if (rank == 0)
        std::cout << "FAIL: Missing weight matrix: " << name << "\n";
      continue;
    }
    const auto &cmat = it->second;
    if (rmat.rows() != cmat.rows() || rmat.cols() != cmat.cols()) {
      local_ok = 0;
      if (rank == 0)
        std::cout << "FAIL: Dimension mismatch for " << name << "\n";
      continue;
    }

    Eigen::MatrixXf abs_diff = (rmat - cmat).cwiseAbs();
    const float mat_max_abs = abs_diff.maxCoeff();
    const float ref_scale = std::max(rmat.cwiseAbs().maxCoeff(),
                                     std::numeric_limits<float>::epsilon());
    const float mat_max_rel = mat_max_abs / ref_scale;

    local_max_abs = std::max(local_max_abs, mat_max_abs);
    local_max_rel = std::max(local_max_rel, mat_max_rel);

    Eigen::MatrixXf tol =
        abs_tolerance + rel_tolerance * rmat.cwiseAbs().array();
    if ((abs_diff.array() > tol.array()).any()) {
      local_ok = 0;
      if (rank == 0)
        std::cout << "FAIL: " << name << " exceeds tolerance\n";
    }
  }

  int global_ok = 0;
  float global_max_abs = 0.0f;
  float global_max_rel = 0.0f;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max_abs, &global_max_abs, 1, MPI_FLOAT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_max_rel, &global_max_rel, 1, MPI_FLOAT, MPI_MAX,
                MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Max abs diff: " << global_max_abs << "\n";
    std::cout << "Max rel diff: " << global_max_rel << "\n";
    std::cout << "Abs tol: " << abs_tolerance << " | Rel tol: " << rel_tolerance
              << "\n";
  }

  return {global_ok != 0, global_max_abs, global_max_rel};
}

void compute_bandwidth(AutoencoderModel &model, double avg_time, int world_size,
                       int rank) {
  size_t total_params = 0;
  auto weights = model.get_weights();
  for (auto &kv : weights)
    total_params += kv.second.rows() * kv.second.cols();

  size_t bytes_per_param = sizeof(float);
  size_t model_size_bytes = total_params * bytes_per_param;
  double total_traffic = model_size_bytes * 2.0 * (world_size - 1) / world_size;
  double total_traffic_mb = total_traffic / (1024.0 * 1024.0);
  double bandwidth_mb_s = total_traffic_mb / avg_time;
  double bandwidth_gb_s = bandwidth_mb_s / 1024.0;

  if (rank == 0) {
    std::cout << "\n=== Bandwidth Analysis ===\n";
    std::cout << "Model size: " << model_size_bytes / 1024.0 << " KB\n";
    std::cout << "Total parameters: " << total_params << "\n";
    std::cout << "Total traffic per iteration: " << total_traffic_mb << " MB\n";
    std::cout << "Effective bandwidth: " << std::fixed << std::setprecision(2)
              << bandwidth_mb_s << " MB/s (" << bandwidth_gb_s << " GB/s)\n";
    double theoretical_peak_gb_s = 12.5;
    double utilization = (bandwidth_gb_s / theoretical_peak_gb_s) * 100.0;
    std::cout << "Bandwidth utilization: " << utilization
              << "% of theoretical peak\n";
  }
}

} // namespace

// Strong scaling analysis
void analyze_strong_scaling(int world_size, double time_at_1_node,
                            double current_time, int rank) {
  if (rank == 0 && world_size > 1) {
    double speedup = time_at_1_node / current_time;
    double efficiency = speedup / world_size;

    std::cout << "\n=== Strong Scaling Analysis ===\n";
    std::cout << "Processes: " << world_size << "\n";
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup
              << "x\n";
    std::cout << "Parallel efficiency: " << efficiency * 100.0 << "%\n";

    // Karp-Flatt metric (serial fraction)
    double serial_fraction =
        (1.0 / speedup - 1.0 / world_size) / (1.0 - 1.0 / world_size);
    std::cout << "Estimated serial fraction: " << serial_fraction * 100.0
              << "%\n";

    if (efficiency < 0.7) {
      std::cout << "WARNING: Efficiency below 70% - scaling limited\n";
    } else if (efficiency < 0.9) {
      std::cout << "NOTE: Moderate scaling efficiency\n";
    } else {
      std::cout << "EXCELLENT: Near-linear scaling\n";
    }
  }
}

int main(int argc, char *argv[]) {
  const BenchmarkOutputConfig benchmark_output_config =
      parse_benchmark_output_config(argc, argv);

  // MPI Setup
  MPI_Init(&argc, &argv);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Configuration
  int warmup_iterations = 10;
  int bench_iterations = 50;
  float correctness_tolerance = 1e-5;

  if (rank == 0) {
    std::cout << "\n========================================\n";
    std::cout << "MPI Federated Averaging Benchmark\n";
    std::cout << "World size: " << world_size << " processes\n";
    std::cout << "Warmup iterations: " << warmup_iterations << "\n";
    std::cout << "Bench iterations: " << bench_iterations << "\n";
    std::cout << "========================================\n";
  }

  // Initialize a base model and create rank variants so averaging has work to
  // do
  AutoencoderModel base_model{256, 28 * 28, 256, 28 * 28};
  AutoencoderModel reference_model = make_rank_variant(base_model, rank);
  AutoencoderModel candidate_model = make_rank_variant(base_model, rank);

  // ===== BENCHMARK 1: Multiple Allreduce (slow per-matrix reference) =====
  double time_multiple = benchmark_weight_updates(
      "MULTIPLE Allreduce (per matrix)", reference_model, world_size,
      bench_iterations, warmup_iterations, rank, update_federated_weights);

  // ===== BENCHMARK 2: Single Allreduce (flattened candidate) =====
  double time_single =
      benchmark_weight_updates("SINGLE Allreduce (flattened)", candidate_model,
                               world_size, bench_iterations, warmup_iterations,
                               rank, update_federated_weights_single_call);

  // ===== PERFORMANCE COMPARISON =====
  if (rank == 0) {
    double overhead_pct = ((time_multiple - time_single) / time_single) * 100.0;

    std::cout << "\n========================================\n";
    std::cout << "PERFORMANCE COMPARISON\n";
    std::cout << "========================================\n";
    std::cout << "Multiple Allreduce (per matrix): " << time_multiple * 1000
              << " ms\n";
    std::cout << "Single Allreduce (flattened):   " << time_single * 1000
              << " ms\n";
    std::cout << "Speedup from flattening: " << std::fixed
              << std::setprecision(2) << (time_multiple / time_single) << "x\n";
    std::cout << "Overhead of multiple calls: " << overhead_pct << "%\n";

    if (overhead_pct > 50) {
      std::cout << "RECOMMENDATION: Use flattened Allreduce (significant "
                   "improvement)\n";
    } else if (overhead_pct > 10) {
      std::cout
          << "RECOMMENDATION: Flattened Allreduce provides moderate benefit\n";
    } else {
      std::cout << "NOTE: Overhead minimal - multiple calls acceptable\n";
    }
  }

  // ===== NUMERICAL CORRECTNESS =====
  if (rank == 0) {
    std::cout << "\n========================================\n";
    std::cout << "NUMERICAL CORRECTNESS\n";
    std::cout << "========================================\n";
  }

  // Recreate independent copies for a single verification step: apply the
  // slow averaging to the reference and the fast averaging to the candidate.
  AutoencoderModel ref_check = make_rank_variant(base_model, rank);
  AutoencoderModel cand_check = make_rank_variant(base_model, rank);
  update_federated_weights(ref_check, 0, world_size, false);
  update_federated_weights_single_call(cand_check, 0, world_size, false);

  const auto correctness =
      verify_correctness(ref_check, cand_check, correctness_tolerance,
                         correctness_tolerance, rank);

  if (rank == 0) {
    if (correctness.passed) {
      std::cout
          << "\n✓ VERIFICATION PASSED: Fast path matches slow reference\n";
    } else {
      std::cout
          << "\n✗ VERIFICATION FAILED: Fast path differs from slow reference\n";
      std::cout << "  Check weight ordering or floating-point accumulation\n";
    }
  }

  // ===== BANDWIDTH ANALYSIS (using candidate model timings) =====
  compute_bandwidth(candidate_model, time_single, world_size, rank);

  if (rank == 0) {
    BenchmarkReport report;
    report.world_size = world_size;
    report.warmup_iterations = warmup_iterations;
    report.bench_iterations = bench_iterations;
    report.time_multiple = time_multiple;
    report.time_single = time_single;
    report.overhead_pct = ((time_multiple - time_single) / time_single) * 100.0;
    report.speedup = time_single > 0.0
                         ? std::optional<double>(time_multiple / time_single)
                         : std::nullopt;

    report.correctness_passed = correctness.passed;
    report.max_abs_diff = correctness.max_abs_diff;
    report.max_rel_diff = correctness.max_rel_diff;

    auto weights = candidate_model.get_weights();
    size_t total_params = 0;
    for (auto &kv : weights) {
      total_params += kv.second.rows() * kv.second.cols();
    }

    report.total_params = total_params;
    report.msg_size_bytes = total_params * sizeof(float);
    report.total_traffic_mb = (static_cast<double>(report.msg_size_bytes) *
                               2.0 * (world_size - 1) / world_size) /
                              (1024.0 * 1024.0);
    report.bandwidth_mb_s = report.total_traffic_mb / time_single;
    report.bandwidth_gb_s = report.bandwidth_mb_s / 1024.0;

    if (world_size > 1 && report.speedup.has_value()) {
      report.parallel_efficiency = report.speedup.value() / world_size;
      report.serial_fraction =
          (1.0 / report.speedup.value() - 1.0 / world_size) /
          (1.0 - 1.0 / world_size);
    }

    write_benchmark_report_json(benchmark_output_config, report);
  }

  MPI_Finalize();
  return 0;
}