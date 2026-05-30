#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <filesystem>
#include <memory>
#include <random>

#include <omp.h>

#include "common.hh"
#include "dataset.hh"
#include "loops.hh"
#include "model.hh"
#include "mse.hh"

#include "stb_image.h"
#include "stb_image_write.h"

static std::unique_ptr<Dataloader> dataloader;
static std::unique_ptr<AutoencoderModel> model;
static std::unique_ptr<MSE> mse;
static std::unique_ptr<experiment_config> config;
constexpr size_t DATA_DIM = 28 * 28;
constexpr size_t NUM_IMAGES = 4096;
// constexpr size_t BATCH_SIZE = 256;

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

  //
  model =
      std::make_unique<AutoencoderModel>(batch_size, DATA_DIM, 256, DATA_DIM);
  mse = std::make_unique<MSE>(batch_size, DATA_DIM);

  // Experiment config
  config = std::make_unique<experiment_config>(
      experiment_config{.train_path = path,
                        .test_path = path,
                        .batch_size = batch_size,
                        .input_dim = DATA_DIM,
                        .hidden_dim = 256,
                        .output_dim = DATA_DIM,
                        .lr = 0.01f,
                        .epoch = 1});
}

static void DoTeardown(const benchmark::State &state) {
  // Clean up mock images
  std::string path = "/tmp/dataloader_benchmark";
  for (const auto &entry : std::filesystem::directory_iterator(path)) {
    std::filesystem::remove(entry.path());
  }
  std::filesystem::remove(path);
}

static void BM_TrainStep(benchmark::State &state) {
  size_t batch_size = state.range(0);
  int num_threads = state.range(1);
  omp_set_num_threads(num_threads);

  for (auto _ : state) {
    auto loss = train("Train: ", *config, *dataloader, *model, *mse);
    benchmark::DoNotOptimize(loss);
  }

  state.counters["BatchSize"] = batch_size;
  state.counters["TotalElements"] = batch_size * DATA_DIM;
  state.counters["Threads"] = num_threads;
  state.counters["ImagesPerSec"] = benchmark::Counter(
      NUM_IMAGES, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_TrainStep)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->ArgsProduct({
        benchmark::CreateRange(2, 1024, 2),
        benchmark::CreateRange(1, 16, 2) // number of threads
    });

BENCHMARK_MAIN();
