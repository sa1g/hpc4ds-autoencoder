#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <random>
#include <memory>

#include <omp.h>

#include <iostream>

// This benchmark/test is a self-contained version of the
// sgd.hh functionality. It's pure trash.

struct SimpleLayer
{
    Eigen::MatrixXf weights;
    Eigen::MatrixXf bias;
    Eigen::MatrixXf grad_weights;
    Eigen::MatrixXf grad_bias;

    SimpleLayer() = default; // Add default constructor

    SimpleLayer(size_t input_dim, size_t output_dim)
    {
        weights.resize(output_dim, input_dim);
        bias.resize(output_dim, 1);
        grad_weights.resize(output_dim, input_dim);
        grad_bias.resize(output_dim, 1);

        weights.setRandom();
        bias.setRandom();
        grad_weights.setRandom();
        grad_bias.setRandom();
    }
};

static SimpleLayer *layer1 = nullptr; // Use pointers instead
static SimpleLayer *layer2 = nullptr;
static size_t input_dim = 28 * 28;
static size_t hidden_dim = 256;
static size_t output_dim = 28 * 28;

template <typename Layer>
void sgd(Layer &layer, float learning_rate)
{
    layer.weights -= learning_rate * layer.grad_weights;
    layer.bias -= learning_rate * layer.grad_bias;
    layer.grad_weights.setZero();
    layer.grad_bias.setZero();
}

template <typename... Layers>
void sgd(float learning_rate, Layers &...layers)
{
    (sgd(layers, learning_rate), ...);
}

static void DoSetup(const benchmark::State &state)
{
    // Allocate new layers
    layer1 = new SimpleLayer(input_dim, hidden_dim);
    layer2 = new SimpleLayer(hidden_dim, output_dim);
}

static void DoTeardown(const benchmark::State &state)
{
    // Clean up allocated memory
    delete layer1;
    delete layer2;
    layer1 = nullptr;
    layer2 = nullptr;
}

static void BM_SGDUpdate(benchmark::State &state)
{
    float learning_rate = 0.1f;

    for (auto _ : state)
    {
        sgd(learning_rate, *layer1, *layer2);
    }

    int total_params = input_dim * output_dim + output_dim;
    state.counters["Throughput"] = benchmark::Counter(
        total_params,
        benchmark::Counter::kIsIterationInvariantRate);
    // benchmark::Counter::OneK::kIs1000);
}

BENCHMARK(BM_SGDUpdate)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(16);

BENCHMARK_MAIN();