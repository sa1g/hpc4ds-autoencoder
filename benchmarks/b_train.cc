#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <random>
#include <memory>
#include <filesystem>

#include <omp.h>

#include "dataset.hh"
#include "model.hh"
#include "mse.hh"
#include "loops.hh"
#include "common.hh"

#include "stb_image.h"
#include "stb_image_write.h"

static std::unique_ptr<Dataloader> dataloader;
static std::unique_ptr<AutoencoderModel> model;
static std::unique_ptr<MSE> mse;
static std::unique_ptr<experiment_config> config;
constexpr size_t DATA_DIM = 28 * 28;
constexpr size_t NUM_IMAGES = 4096;
constexpr size_t BATCH_SIZE = 256;

static void DoSetup(const benchmark::State &state)
{
    omp_set_num_threads(state.threads());

    // Create mock filenames
    std::vector<std::string> filenames;
    for (size_t i = 0; i < NUM_IMAGES; ++i)
    {
        filenames.push_back("image_" + std::to_string(i));
    }

    // Create mock images and save them to disk
    std::string path = "/tmp/dataloader_benchmark";
    std::filesystem::create_directories(path);
    for (const auto &filename : filenames)
    {
        std::string filepath = path + "/" + filename + ".png";
        std::vector<unsigned char> image(DATA_DIM);
        for (size_t i = 0; i < DATA_DIM; ++i)
        {
            image[i] = static_cast<unsigned char>(i % 256);
        }
        stbi_write_png(filepath.c_str(), 28, 28, 1, image.data(), 0);
    }

    dataloader = std::make_unique<Dataloader>(path, filenames, 28, 28,
                                              NUM_IMAGES, BATCH_SIZE, false);

    //
    model = std::make_unique<AutoencoderModel>(BATCH_SIZE, DATA_DIM, 256, DATA_DIM);
    mse = std::make_unique<MSE>(BATCH_SIZE, DATA_DIM);

    // Experiment config
    config = std::make_unique<experiment_config>(
        experiment_config{
            .train_path = path,
            .test_path = path,
            .batch_size = BATCH_SIZE,
            .input_dim = DATA_DIM,
            .hidden_dim = 256,
            .output_dim = DATA_DIM,
            .lr = 0.01f,
            .epoch = 1});
}

static void DoTeardown(const benchmark::State &state)
{
    // Clean up mock images
    std::string path = "/tmp/dataloader_benchmark";
    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        std::filesystem::remove(entry.path());
    }
    std::filesystem::remove(path);
}

static void BM_TrainStep(benchmark::State &state)
{
    for (auto _ : state)
    {
        auto loss = train("Train: ", *config, *dataloader, *model, *mse);
        benchmark::DoNotOptimize(loss);
    }
}

BENCHMARK(BM_TrainStep)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(16);

BENCHMARK_MAIN();
