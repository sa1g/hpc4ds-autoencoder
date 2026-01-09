#include <iostream>

// TODO: remove this one
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "common.hh"
#include "worker.hh"

int main(int argc, char *argv[])
{
  const experiment_config config = {.train_path = "data/mnist/train",
                                    .test_path = "data/mnist/test",
                                    .batch_size = 256,
                                    .input_dim = 28 * 28,
                                    .hidden_dim = 256,
                                    .output_dim = 28 * 28,
                                    .lr = 0.01f,
                                    .epoch = 20};

  ///////////////////////////////////////////////////////

  std::vector<std::string> filenames = get_filenames(config.train_path);
  auto [train_filenames, eval_filenames] =
      random_split_filenames(filenames, 20, 42);

  std::vector<std::string> test_filenames = get_filenames(config.test_path);

  auto_worker(config, train_filenames, eval_filenames, test_filenames, "prove",
              0, get_timestamp_string_with_full_micros());

  std::cout << "Done!" << std::endl;

  return 0;
}
