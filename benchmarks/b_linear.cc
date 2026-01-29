#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <random>
#include <memory>

#include <omp.h>

#include <iostream>

#include "Eigen/src/Core/Matrix.h"
#include "linear.hh"

static std::unique_ptr<Linear> linear_layer;
static Eigen::MatrixXf input;
static Eigen::MatrixXf output;
constexpr size_t DATA_DIM = 28 * 28;

static void DoSetup(const benchmark::State &state)
{
    size_t max_batch_size = 256;
    linear_layer = std::make_unique<Linear>(max_batch_size, DATA_DIM, DATA_DIM);

    input = Eigen::MatrixXf(max_batch_size, DATA_DIM);
    output = Eigen::MatrixXf(max_batch_size, DATA_DIM);

    std::default_random_engine generator(42);
    std::uniform_real_distribution<float> distribution(-10.0, 10.0);
    std::uniform_real_distribution<float> distribution1(0.0, 1.0);

    for (int i = 0; i < input.rows(); ++i)
    {
        for (int j = 0; j < input.cols(); ++j)
        {
            input(i, j) = distribution(generator);
            output(i, j) = distribution1(generator);
        }
    }
}

static void DoTeardown(const benchmark::State &state) {}

static void BM_LinearForward(benchmark::State &state)
{
    size_t batch_size = state.range(0);

    batch_size = std::min(batch_size, (size_t)input.rows());

    // Slice
    Eigen::MatrixXf in = input.topRows(batch_size);

    for (auto _ : state)
    {
        auto out = linear_layer->forward(in);
        benchmark::DoNotOptimize(out);
    }

    state.counters["BatchSize"] = batch_size;
    state.counters["TotalElements"] = batch_size * DATA_DIM;

    state.counters["Throughput"] = benchmark::Counter(
        batch_size * DATA_DIM,
        benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_LinearBackward(benchmark::State &state)
{
    size_t batch_size = state.range(0);

    batch_size = std::min(batch_size, (size_t)input.rows());

    // Slice
    Eigen::MatrixXf in = input.topRows(batch_size);
    Eigen::MatrixXf ou = output.topRows(batch_size);

    for (auto _ : state)
    {
        auto out = linear_layer->backward(in, ou);
        benchmark::DoNotOptimize(out);
    }

    state.counters["BatchSize"] = batch_size;
    state.counters["TotalElements"] = batch_size * DATA_DIM;

    state.counters["Throughput"] = benchmark::Counter(
        batch_size * DATA_DIM * 2,
        benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_LinearForward)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->ArgsProduct({
        benchmark::CreateRange(2, 256, 2),
    })
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(16);

BENCHMARK(BM_LinearBackward)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->ArgsProduct({
        benchmark::CreateRange(2, 256, 2),
    })
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(16);
BENCHMARK_MAIN();
