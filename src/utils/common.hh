#include <vector>
#include <string>
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

struct experiment_config {
    std::string train_path;
    std::string test_path;
    size_t batch_size;
    size_t input_dim;
    size_t hidden_dim;
    size_t output_dim;
    float lr;
    int epoch;
};

#endif
// __AUTOENCODER_COMMON_HH__