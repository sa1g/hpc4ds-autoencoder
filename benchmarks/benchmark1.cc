#include <benchmark/benchmark.h>
#include <vector>
#include <array>

// First implementation using std::vector
static void BM_Vector(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<int> v(state.range(0), 1);
        benchmark::DoNotOptimize(v.data());  // Prevents compiler optimizations
    }
}

// Second implementation using std::array
static void BM_Array(benchmark::State& state) {
    for (auto _ : state) {
        std::array<int, 1000> arr{};
        arr.fill(1);
        benchmark::DoNotOptimize(arr.data());
    }
}

// Register both benchmarks with different sizes
BENCHMARK(BM_Vector)->Arg(1000)->Arg(10000)->Arg(100000);
BENCHMARK(BM_Array);

// Run the benchmark
BENCHMARK_MAIN();