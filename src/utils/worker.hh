#ifndef __AUTOENCODER_WORKER_HH__
#define __AUTOENCODER_WORKER_HH__

#include "common.hh"

void auto_worker(const experiment_config &config,
                 std::vector<std::string> &train_filenames,
                 std::vector<std::string> &eval_filenames,
                 std::vector<std::string> &test_filenames,
                 std::string experiment_name, int worker_id, int world_size,
                 std::string timestamp);

// void auto_worker(const experiment_config &config);

#endif // __AUTOENCODER_WORKER_HH__