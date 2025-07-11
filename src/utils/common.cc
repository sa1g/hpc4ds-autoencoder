#include "common.hh"
#include <algorithm>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

/**
 * Get all PNG filenames from the dataset directory.
 * Filenames are formatted as "subdir/filename" where subdir is the digit directory (0-9) and filename is the name of the image without extension.
 *
 * @param root_dir The root directory containing subdirectories named 0, 1, ..., 9.
 * @return A vector of formatted filenames.
 */
std::vector<std::string> get_filenames(const std::string &root_dir)
{
    std::vector<std::string> filenames;

    // Traverse through the directories
    for (const auto &entry : fs::directory_iterator(root_dir))
    {
        // Get the subdirectory name (0, 1, ..., 9)
        std::string subdir_name = entry.path().filename().string();

        // Traverse the files inside this subdirectory
        for (const auto &file : fs::directory_iterator(entry))
        {
            // Check if it's PNG
            if (fs::is_regular_file(file) && file.path().extension() == ".png")
            {
                // Get filename w/o extension
                std::string filename = file.path().stem().string();

                // Format and store in the vector
                filenames.push_back(subdir_name + "/" + filename);
            }
        }
    }

    return filenames;
}

/**
 * Randomly splits the filenames into training and testing sets based on the given percentage.
 *
 * @param filenames Vector of filenames to split.
 * @param percentage_test Percentage of filenames to include in the test set (default is 20%).
 * @return A tuple containing two vectors: training filenames and testing filenames.
 */
std::tuple<std::vector<std::string>, std::vector<std::string>> random_split_filenames(const std::vector<std::string> &filenames, const int percentage_test, const int seed)
{
    std::vector<std::string> train_filenames;
    std::vector<std::string> test_filenames;

    // Shuffle the filenames
    std::vector<std::string> shuffled_filenames = filenames;

    std::mt19937 rg(seed);

    std::shuffle(shuffled_filenames.begin(), shuffled_filenames.end(), rg);

    // Calculate the split index
    size_t split_index = (shuffled_filenames.size() * (100 - percentage_test)) / 100;

    // Split into train and test sets
    train_filenames.assign(shuffled_filenames.begin(), shuffled_filenames.begin() + split_index);
    test_filenames.assign(shuffled_filenames.begin() + split_index, shuffled_filenames.end());

    return {train_filenames, test_filenames};
}