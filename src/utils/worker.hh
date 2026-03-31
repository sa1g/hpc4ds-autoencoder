#ifndef __AUTOENCODER_WORKER_HH__
#define __AUTOENCODER_WORKER_HH__

#include "common.hh"

/**
    * @brief Main worker function that performs the training and evaluation of the autoencoder model.
    * 
    * This function sets up the dataloaders, model, criterion, and logger, and then runs the training loop for the specified number of epochs. After training, if MPI is used and there are multiple processes, it performs weight averaging across the processes.
 */
void auto_worker(const experiment_config &config,
                 std::vector<std::string> &train_filenames,
                 std::vector<std::string> &eval_filenames,
                 std::vector<std::string> &test_filenames,
                 std::string experiment_name, int worker_id, int world_size,
                 std::string timestamp);

#endif // __AUTOENCODER_WORKER_HH__