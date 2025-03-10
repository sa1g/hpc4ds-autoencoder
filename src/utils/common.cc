#include "common.hh"
#include <filesystem>

namespace fs = std::filesystem;

std::vector<std::string> get_filenames(const std::string &root_dir){
    std::vector<std::string> filenames;

    // Traverse through the directories
    for (const auto& entry : fs::directory_iterator(root_dir)){
        // Get the subdirectory name (0, 1, ..., 9)
        std::string subdir_name = entry.path().filename().string();

        // Traverse the files inside this subdirectory
        for (const auto& file : fs::directory_iterator(entry)){
            // Check if it's PNG
            if (fs::is_regular_file(file) && file.path().extension() == ".png"){
                // Get filename w/o extension
                std::string filename = file.path().stem().string();

                // Format and store in the vector
                filenames.push_back(subdir_name + "/" + filename);
            }
        }
    }

    return filenames;
}
