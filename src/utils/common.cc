#include "common.hh"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>
#include <format>

namespace fs = std::filesystem;

std::vector<std::string> get_filenames(const std::string &root_dir) {
  std::vector<std::string> filenames;

  // Traverse through the directories
  for (const auto &entry : fs::directory_iterator(root_dir)) {
    // Get the subdirectory name (0, 1, ..., 9)
    std::string subdir_name = entry.path().filename().string();

    // Traverse the files inside this subdirectory
    for (const auto &file : fs::directory_iterator(entry)) {
      // Check if it's PNG
      if (fs::is_regular_file(file) && file.path().extension() == ".png") {
        // Get filename w/o extension
        std::string filename = file.path().stem().string();

        // Format and store in the vector
        filenames.push_back(subdir_name + "/" + filename);
      }
    }
  }

  return filenames;
}

std::tuple<std::vector<std::string>, std::vector<std::string>>
random_split_filenames(const std::vector<std::string> &filenames,
                       const int percentage_test, const int seed) {
  std::vector<std::string> train_filenames;
  std::vector<std::string> test_filenames;

  // Shuffle the filenames
  std::vector<std::string> shuffled_filenames = filenames;

  std::mt19937 rg(seed);

  std::shuffle(shuffled_filenames.begin(), shuffled_filenames.end(), rg);

  // Calculate the split index
  size_t split_index =
      (shuffled_filenames.size() * (100 - percentage_test)) / 100;

  // Split into train and test sets
  train_filenames.assign(shuffled_filenames.begin(),
                         shuffled_filenames.begin() + split_index);
  test_filenames.assign(shuffled_filenames.begin() + split_index,
                        shuffled_filenames.end());

  return {train_filenames, test_filenames};
}

std::vector<std::string> split_data(const std::vector<std::string>& filenames, 
                                         int rank, int size) {
  size_t total_files = filenames.size();
  
  // Calculate basic files per rank and the remainder
  size_t files_per_rank = total_files / size;
  size_t remainder = total_files % size;

  // Calculate the start index for this rank.
  // Ranks lower than 'remainder' get an extra file, so their start 
  // offset increases by 1 for each preceding rank.
  size_t start = rank * files_per_rank + std::min((size_t)rank, remainder);
    
  // Calculate the number of files for this specific rank.
  size_t count = files_per_rank + (static_cast<size_t>(rank) < remainder ? 1 : 0);

  // Compute the final list to return
  std::vector<std::string> sharded_list;
  if (start < total_files) {
      auto begin_it = filenames.begin() + start;
      auto end_it = filenames.begin() + std::min(start + count, total_files);
      sharded_list.assign(begin_it, end_it);
  }

  return sharded_list;
}

bool create_directory_if_not_exists(const std::string &path) {
  try {
    if (fs::exists(path)) {
      if (fs::is_directory(path)) {
        return true;
      } else {
        return false;
      }
    }

    // Create the directory (including parent directories if needed)
    return fs::create_directories(path);
  } catch (const fs::filesystem_error &e) {
    std::cerr << "Error creating directory: " << e.what() << std::endl;
    return false;
  }
}

std::string get_timestamp_string_with_full_micros() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_us = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto us = now_us.time_since_epoch().count() % 1000000;
    
    std::tm tm = *std::localtime(&now_time_t);
    
    return std::format("{:02d}{:02d}{:04d}_{:02d}{:02d}{:02d}{:06d}",
                      tm.tm_mday,
                      tm.tm_mon + 1,
                      tm.tm_year + 1900,
                      tm.tm_hour,
                      tm.tm_min,
                      tm.tm_sec,
                      us);
}
