#include <iostream>
#include <string_view>

// TODO: remove this one
#ifdef USE_MPI
#include <mpi.h>
#endif

#ifndef DATASET_NAME
#define DATASET_NAME "mnist"
#endif

#include "common.hh"
#include "worker.hh"

int main(int argc, char *argv[]) {

  // MPI initialization blocks

  int my_rank;
  int comm_sz;

  MPI_Init(NULL, NULL);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  // Autoencoder code
  const std::string dataset_name = DATASET_NAME;
  int n_channels = 0;

  if (dataset_name == "mnist") {
    n_channels = 1;
  } else if (dataset_name == "svhn") {
    n_channels = 3;
  } else {
    // Handle error at runtime
    throw std::runtime_error("Unknown DATASET_NAME: " +
                             std::string(dataset_name));
  }

  const experiment_config config = {
      .train_path = "data/" + dataset_name + "/train",
      .test_path = "data/" + dataset_name + "/test",
      .n_channels = n_channels,
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

  auto_worker(config, my_train, my_eval, my_test, "prove", my_rank, comm_sz,
              get_timestamp_string_with_full_micros());

  std::cout << "Done!" << std::endl;

  // MPI teardown
  MPI_Finalize();

  return 0;
}
