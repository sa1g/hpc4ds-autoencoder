#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <random>
#include <memory>

#include <omp.h>

#include <iostream>

#include "relu.hh"

static std::unique_ptr<ReLU> relu;
static Eigen::MatrixXf input;
static Eigen::MatrixXf grad_output;
constexpr size_t DATA_DIM = 28 * 28;

static void DoSetup(const benchmark::State &state)
{
    size_t max_batch_size = 256;
    relu = std::make_unique<ReLU>(max_batch_size, DATA_DIM);

    input = Eigen::MatrixXf(max_batch_size, DATA_DIM);
    grad_output = Eigen::MatrixXf(max_batch_size, DATA_DIM);

    std::default_random_engine generator(42);
    std::uniform_real_distribution<float> distribution(-10.0, 10.0);
    std::uniform_real_distribution<float> smaller_distribution(0.0, 1.0);

    for (int i = 0; i < input.rows(); ++i)
        for (int j = 0; j < input.cols(); ++j)
        {
            input(i, j) = distribution(generator);
            grad_output(i, j) = smaller_distribution(generator);
        }
}

static void DoTeardown(const benchmark::State &state)
{
}

static void BM_ReLUForward(benchmark::State &state)
{
    size_t batch_size = state.range(0);

    batch_size = std::min(batch_size, (size_t)input.rows());

    // Slice the input to requested batch size
    Eigen::MatrixXf in = input.topRows(batch_size);

    for (auto _ : state)
    {
        auto out = relu->forward(in);
        benchmark::DoNotOptimize(out);
    }

    state.counters["BatchSize"] = batch_size;
    state.counters["TotalElements"] = batch_size * DATA_DIM;

    state.counters["Throughput"] = benchmark::Counter(
        batch_size * DATA_DIM,
        benchmark::Counter::kIsIterationInvariantRate);
    // benchmark::Counter::OneK::kIs1000
}

static void BM_ReluBackward(benchmark::State &state)
{
    size_t batch_size = state.range(0);

    batch_size = std::min(batch_size, (size_t)input.rows());

    // Slice the input and grad_output to requested batch size
    Eigen::MatrixXf in = input.topRows(batch_size);
    Eigen::MatrixXf grad_out = grad_output.topRows(batch_size);

    for (auto _ : state)
    {
        auto grad_in = relu->backward(in, grad_out);
        benchmark::DoNotOptimize(grad_in);
    }

    state.counters["BatchSize"] = batch_size;
    state.counters["TotalElements"] = batch_size * DATA_DIM;

    state.counters["Throughput"] = benchmark::Counter(
        batch_size * DATA_DIM,
        benchmark::Counter::kIsIterationInvariantRate);
    // benchmark::Counter::OneK::kIs1000);
}

BENCHMARK(BM_ReLUForward)
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

BENCHMARK(BM_ReluBackward)
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
