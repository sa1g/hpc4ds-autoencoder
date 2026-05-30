#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <memory>
#include <random>

#include <omp.h>

#include "dataset.hh"
#include "stb_image.h"
#include "stb_image_write.h"
#include <filesystem>

static std::unique_ptr<Dataloader> dataloader;
constexpr size_t DATA_DIM = 28 * 28;
constexpr size_t NUM_IMAGES = 1024 * 8;

static void DoSetup(const benchmark::State &state) {
  size_t batch_size = state.range(0);

  // Create mock filenames
  std::vector<std::string> filenames;
  for (size_t i = 0; i < NUM_IMAGES; ++i) {
    filenames.push_back("image_" + std::to_string(i));
  }

  // Create mock images and save them to disk
  std::string path = "/tmp/dataloader_benchmark";
  std::filesystem::create_directories(path);
  for (const auto &filename : filenames) {
    std::string filepath = path + "/" + filename + ".png";
    std::vector<unsigned char> image(DATA_DIM);
    for (size_t i = 0; i < DATA_DIM; ++i) {
      image[i] = static_cast<unsigned char>(i % 256);
    }
    stbi_write_png(filepath.c_str(), 28, 28, 1, image.data(), 0);
  }

  dataloader = std::make_unique<Dataloader>(path, filenames, 28, 28, NUM_IMAGES,
                                            batch_size, false);
}

static void DoTeardown(const benchmark::State &state) {
  // Clean up mock images
  std::string path = "/tmp/dataloader_benchmark";
  for (const auto &entry : std::filesystem::directory_iterator(path)) {
    std::filesystem::remove(entry.path());
  }
  std::filesystem::remove(path);
}

static void BM_Dataloader_GetBatch(benchmark::State &state) {
  size_t batch_size = state.range(0);
  int num_threads = state.range(1);
  omp_set_num_threads(num_threads);

  batch_size = std::min(batch_size, (size_t)dataloader->get_num_images());

  for (auto _ : state) {
    Eigen::MatrixXf &batch = dataloader->get_batch();
    benchmark::DoNotOptimize(batch);
  }

  state.counters["BatchSize"] = batch_size;
  state.counters["TotalElements"] = batch_size * DATA_DIM;
  state.counters["Threads"] = num_threads;
}

BENCHMARK(BM_Dataloader_GetBatch)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->ArgsProduct({
        benchmark::CreateRange(2, 1024, 2), // batch size
        benchmark::CreateRange(1, 16, 2)    // number of threads
    });

BENCHMARK_MAIN();
