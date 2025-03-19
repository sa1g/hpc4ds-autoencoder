#include <benchmark/benchmark.h>
#include "relu.hh" 

static void BM_ExampleFunction(benchmark::State& state) {
    constexpr size_t max_batch_size = 10;
    constexpr size_t data_dim = 3;

    ReLU<max_batch_size, data_dim> relu;
    Eigen::MatrixXf input(2, 3);
    input << -1, 2, -3, 4, -5, 6;
    
    for (auto _ : state) {
        // Call your function here
        Eigen::MatrixXf output = relu.forward(input);
    }
}

// Register the benchmark
BENCHMARK(BM_ExampleFunction);

// Run the benchmark
BENCHMARK_MAIN();