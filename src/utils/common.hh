#include <vector>
#include <string>

// TODO: remove this one
#ifdef USE_MPI
#include <mpi.h>
#endif

#ifndef __AUTOENCODER_COMMON_HH__
#define __AUTOENCODER_COMMON_HH__

/**
 * Get all PNG filenames from the dataset directory.
 * Filenames are formatted as "subdir/filename" where subdir is the digit directory (0-9) and filename is the name of the image without extension.
 *
 * @param root_dir The root directory containing subdirectories named 0, 1, ..., 9.
 * @return A vector of formatted filenames.
 */
std::vector<std::string> get_filenames(const std::string &path);

/**
 * Randomly splits the filenames into training and testing sets based on the given percentage.
 *
 * @param filenames Vector of filenames to split.
 * @param percentage_test Percentage of filenames to include in the test set (default is 20%).
 * @return A tuple containing two vectors: training filenames and testing filenames.
 */
std::tuple<std::vector<std::string>, std::vector<std::string>> random_split_filenames(const std::vector<std::string> &filenames, const int percentage_test, const int seed);

/**
 * Distributes a list of filenames across MPI processes (Data Sharding)
 * 
 * @param filenames The complete vector of all available filenames.
 * @param rank The rank of the current MPI process (0 to size-1).
 * @param size The total number of MPI processes in the communicator.
 * @return The vector containing the subset of filenames assigned to this rank.
 */
std::vector<std::string> split_data(const std::vector<std::string>& filenames, 
                                         int rank, int size);

struct experiment_config
{
    std::string train_path;
    std::string test_path;
    int n_channels;
    size_t batch_size;
    size_t input_dim;
    size_t hidden_dim;
    size_t output_dim;
    float lr;
    int epoch;
};

bool create_directory_if_not_exists(const std::string &path);

std::string get_timestamp_string_with_full_micros();

#endif
// __AUTOENCODER_COMMON_HH__