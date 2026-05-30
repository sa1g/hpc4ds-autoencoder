#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <memory>
#include <random>

#include <omp.h>

#include "mse.hh"

static std::unique_ptr<MSE> mse;
static Eigen::MatrixXf input;
static Eigen::MatrixXf target;
constexpr size_t DATA_DIM = 28 * 28;

static void DoSetup(const benchmark::State &state) {
  size_t max_batch_size = 1024;
  mse = std::make_unique<MSE>(max_batch_size, DATA_DIM);

  input = Eigen::MatrixXf(max_batch_size, DATA_DIM);
  target = Eigen::MatrixXf(max_batch_size, DATA_DIM);

  std::default_random_engine generator(42);
  std::uniform_real_distribution<float> distribution(-10.0, 10.0);

  for (int i = 0; i < input.rows(); ++i)
    for (int j = 0; j < input.cols(); ++j) {
      input(i, j) = distribution(generator);
      target(i, j) = distribution(generator);
    }
}

static void DoTeardown(const benchmark::State &state) {}

static void BM_MSE(benchmark::State &state) {
  size_t batch_size = state.range(0);
  int num_threads = state.range(1);
  omp_set_num_threads(num_threads);

  batch_size = std::min(batch_size, (size_t)input.rows());

  // Slice the input to requested batch size
  Eigen::MatrixXf in = input.topRows(batch_size);
  Eigen::MatrixXf tar = target.topRows(batch_size);

  for (auto _ : state) {
    auto out = mse->mse_loss(in, tar);
    benchmark::DoNotOptimize(out);
  }

  // state.counters["Threads"] = num_threads;
  state.counters["BatchSize"] = batch_size;
  state.counters["TotalElements"] = batch_size * DATA_DIM;
  state.counters["Threads"] = num_threads;

    const double bytes_processed =
      2.0 * batch_size * DATA_DIM * sizeof(float);
    state.counters["GB/s"] = benchmark::Counter(
      bytes_processed, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::OneK::kIs1024);
}

BENCHMARK(BM_MSE)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->ArgsProduct({
        benchmark::CreateRange(2, 1024, 2),
        benchmark::CreateRange(1, 16, 2) // number of threads
    });

BENCHMARK_MAIN();