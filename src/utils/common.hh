#include <vector>
#include <string>
#ifndef __AUTOENCODER_COMMON_HH__
#define __AUTOENCODER_COMMON_HH__

std::vector<std::string> get_filenames(const std::string &path);

std::tuple<std::vector<std::string>, std::vector<std::string>> random_split_filenames(const std::vector<std::string> &filenames, const int percentage_test, const int seed);

#endif
// __AUTOENCODER_COMMON_HH__