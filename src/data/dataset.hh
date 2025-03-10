#ifndef __AUTOENCODER_DATASET_HH__
#define __AUTOENCODER_DATASET_HH__

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::MatrixXd loadImageToMatrix(const std::string &filename, bool &is_rgb, int &width, int &height);

class Dataloader
{
private:
    std::vector<std::string> _filenames;
    int _width;
    int _height;
    const int _num_images;
    const std::string _path;
    const int _batch_size;
    const std::string _extension{".png"};
    const bool _shuffle;
    int _n_channels{1};
    int _batch_start_index{0};

    int _current_index{0};
    // Eigen::MatrixXd get_image(int index); // Modify it to save the image in the pointer of the tensor
    // Eigen::MatrixXd get_next();
    // Add a Tensor to store the current batch

public:
    Dataloader(const std::string &path, const std::vector<std::string> filenames, const int width, const int height, const int num_images, const int batch_size, const bool shuffle);

    const int num_batches;
    Eigen::Tensor<float, 3> get_batch(); // Modify it to return a pointer
};

#endif // __AUTOENCODER_DATASET_HH__