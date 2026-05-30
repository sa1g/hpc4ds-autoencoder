#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <memory>
#include <random>

#include <omp.h>

#include <iostream>

#include "Eigen/src/Core/Matrix.h"
#include "linear.hh"

static std::unique_ptr<Linear> linear_layer;
static Eigen::MatrixXf input;
static Eigen::MatrixXf output;
constexpr size_t DATA_DIM = 28 * 28;

static void DoSetup(const benchmark::State &state) {
  size_t max_batch_size = 1024;
  linear_layer = std::make_unique<Linear>(max_batch_size, DATA_DIM, DATA_DIM);

  input = Eigen::MatrixXf(max_batch_size, DATA_DIM);
  output = Eigen::MatrixXf(max_batch_size, DATA_DIM);

  std::default_random_engine generator(42);
  std::uniform_real_distribution<float> distribution(-10.0, 10.0);
  std::uniform_real_distribution<float> distribution1(0.0, 1.0);

  for (int i = 0; i < input.rows(); ++i) {
    for (int j = 0; j < input.cols(); ++j) {
      input(i, j) = distribution(generator);
      output(i, j) = distribution1(generator);
    }
  }
}

static void DoTeardown(const benchmark::State &state) {}

static void BM_LinearForward(benchmark::State &state) {
  size_t batch_size = state.range(0);
  int num_threads = state.range(1);
  omp_set_num_threads(num_threads);

  batch_size = std::min(batch_size, (size_t)input.rows());

  // Slice
  Eigen::MatrixXf in = input.topRows(batch_size);

  for (auto _ : state) {
    auto out = linear_layer->forward(in);
    benchmark::DoNotOptimize(out);
  }

  state.counters["BatchSize"] = batch_size;
  state.counters["TotalElements"] = batch_size * DATA_DIM;
  state.counters["Threads"] = num_threads;

    const double bytes_processed =
      (2.0 * batch_size * DATA_DIM + DATA_DIM * DATA_DIM + DATA_DIM) *
      sizeof(float);
    state.counters["GB/s"] = benchmark::Counter(
      bytes_processed, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::OneK::kIs1024);
}

static void BM_LinearBackward(benchmark::State &state) {
  size_t batch_size = state.range(0);
  int num_threads = state.range(1);
  omp_set_num_threads(num_threads);

  batch_size = std::min(batch_size, (size_t)input.rows());

  // Slice
  Eigen::MatrixXf in = input.topRows(batch_size);
  Eigen::MatrixXf ou = output.topRows(batch_size);

  for (auto _ : state) {
    auto out = linear_layer->backward(in, ou);
    benchmark::DoNotOptimize(out);
  }

  state.counters["BatchSize"] = batch_size;
  state.counters["TotalElements"] = batch_size * DATA_DIM;
  state.counters["Threads"] = num_threads;

    const double bytes_processed =
      (3.0 * batch_size * DATA_DIM + 2.0 * DATA_DIM * DATA_DIM + DATA_DIM) *
      sizeof(float);
    state.counters["GB/s"] = benchmark::Counter(
      bytes_processed, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::OneK::kIs1024);
}

BENCHMARK(BM_LinearForward)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({
        benchmark::CreateRange(2, 1024, 2), // batch size
        benchmark::CreateRange(1, 16, 2)    // number of threads
    });

BENCHMARK(BM_LinearBackward)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->ArgsProduct({
        benchmark::CreateRange(2, 1024, 2),
        benchmark::CreateRange(1, 16, 2) // number of threads
    });

BENCHMARK_MAIN();
