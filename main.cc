#include <iostream>
#include <string_view>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifndef DATASET_NAME
#define DATASET_NAME "mnist"
#endif

#include "common.hh"
#include "worker.hh"

int main(int argc, char *argv[]) {

  int my_rank = 0;
  int comm_sz = 1;
  
  #ifdef USE_MPI

  // MPI initialization blocks

  MPI_Init(NULL, NULL);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  #endif

  // Autoencoder code
  const experiment_config config = {
      .train_path = std::string("data/") + DATASET_NAME + "/train",
      .test_path = std::string("data/") + DATASET_NAME + "/test",
      .batch_size = 256,
      .input_dim = 28 * 28,
      .hidden_dim = 256,
      .output_dim = 28 * 28,
      .lr = 0.01f,
      .epoch = 20};

  ///////////////////////////////////////////////////////

  std::vector<std::string> filenames = get_filenames(config.train_path);

  // For workloads shared on different nodes, it's important the seed used here
  // stays the same, otherwise the data used in testing might end up in the
  // training data used by some other node
  auto [train_filenames, eval_filenames] =
      random_split_filenames(filenames, 20, 42);
  std::vector<std::string> test_filenames = get_filenames(config.test_path);

  // Separating my portion of the training data to use
  std::vector<std::string> my_train =
      split_data(train_filenames, my_rank, comm_sz);

  // And since we apply weight averaging at the end, the eval and test data
  // should be the same throughout training for each node
  std::vector<std::string> my_eval = eval_filenames;
  std::vector<std::string> my_test = test_filenames;

  // Printing the portion of data we got
  if (train_filenames.size() > 0) {
    std::cout << "[Rank " << my_rank << "/" << comm_sz << "] Assigned "
              << my_train.size() << " training files ("
              << (int)(1000.0 * my_train.size() / train_filenames.size()) / 10.0
              << "% of total)" << std::endl;
  }

  // ---- Total time measurement (wall-clock)
  #ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  #else
    #ifdef _OPENMP
    double t0 = omp_get_wtime();
    #else
    auto t0 = std::chrono::steady_clock::now();
    #endif
  #endif

  auto_worker(config, my_train, my_eval, my_test, "prove", my_rank, comm_sz,
              get_timestamp_string_with_full_micros());

  #ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();
  double local = t1 - t0;

  double max_t = 0.0, avg_t = 0.0;
  MPI_Reduce(&local, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    avg_t /= comm_sz;
    std::cout << "Total wall time (avg/max): " << avg_t
              << " / " << max_t << " s\n";
  }
  #else
    #ifdef _OPENMP
    double t1 = omp_get_wtime();
    std::cout << "Total wall time: " << (t1 - t0) << " s\n";
    #else
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "Total wall time: " << dt.count() << " s\n";
    #endif
  #endif

  std::cout << "Done!" << std::endl;


  #ifdef USE_MPI

  // MPI teardown
  MPI_Finalize();

  #endif

  return 0;
}
