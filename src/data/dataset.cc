#include "dataset.hh"
#include "rand.hh"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif // STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <algorithm>

Eigen::MatrixXd loadImageToMatrix(const std::string &filename, bool &is_rgb, int &width, int &height)
{

    // Load the image using stb_image
    int n_channels;
    unsigned char *data = stbi_load(filename.c_str(), &width, &height, &n_channels, 0);
    if (!data)
    {
        std::cerr << "Error loading image: " << filename << std::endl;
        exit(1);
    }

    // Set is_rbg to true if the image has 3 channels (RGB), false if 1 channel (grayscale)
    is_rgb = (n_channels == 3);

    // Create the Eigen matrix
    if (is_rgb)
    {
        // If the image is RGB, create a matrix with 3 rows for each pixel (R, G, B)
        Eigen::MatrixXd rgb_matrix(height, width * 3);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; y < width; ++x)
            {
                int idx = (y * width + x) * 3;
                rgb_matrix(y, x * 3) = data[idx] / 255.0;
                rgb_matrix(y, x * 3 + 1) = data[idx + 1] / 255.0;
                rgb_matrix(y, x * 3 + 2) = data[idx + 2] / 255.0;
            }
        }

        // Cleanup and return RGB matrix
        stbi_image_free(data);
        return rgb_matrix;
    }
    else
    {
        Eigen::MatrixXd matrix(height, width);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = (y * width + x);
                matrix(y, x) = data[idx] / 255.0;
            }
        }

        // Cleanup and return matrix
        stbi_image_free(data);
        return matrix;
    }
}

Dataloader::Dataloader(const std::string &path, const std::vector<std::string> filenames, const int width, const int height, const int num_images, const int batch_size, const bool shuffle) : _path(path), _filenames(filenames), _width(width), _height(height), _num_images(num_images), _batch_size(batch_size), _shuffle(shuffle), num_batches(_num_images % _batch_size)
{
    if (_num_images > _filenames.size())
    {
        std::cerr << "Number of images is greater than the number of filenames" << std::endl;
        exit(1);
    }

    if (_shuffle)
    {
        std::shuffle(std::begin(_filenames), std::end(_filenames),
                     autoencoder_random_generator);
    }
}

// Eigen::MatrixXd Dataloader::get_image(int index)
// {

//     unsigned char *data = stbi_load(_filenames[index].c_str(), &_width, &_height, &_n_channels, 0);

//     if (!data)
//     {
//         std::cerr << "Error loading image: " << _filenames[index] << std::endl;
//         exit(1);
//     }

//     Eigen::MatrixXd matrix(_height, _width);

//     for (int y = 0; y < _height; ++y)
//     {
//         for (int x = 0; x < _width; ++x)
//         {
//             int idx = (y * _width + x);
//             matrix(y, x) = data[idx] / 255.0;
//         }
//     }

//     // Cleanup and return matrix
//     stbi_image_free(data);
//     return matrix;
// }

Eigen::Tensor<float, 3> Dataloader::get_batch()
{
    // Shape [BATCH_SIZE, HEIGHT, WIDTH]
    Eigen::Tensor<float, 3> batch(_batch_size, _height, _width);

    for (int b = 0; b < _batch_size; ++b)
    {
        int index = _batch_start_index + b; // Get the correct image index for the batch

        // Load the image using stb_image
        std::string filename = _path + "/" + _filenames[index] + _extension;
        std::cout << filename << std::endl;
        unsigned char *data = stbi_load(filename.c_str(), &_width, &_height, &_n_channels, 0);

        if (!data)
        {
            std::cerr << "Error loading image: " << _filenames[index] << "Line: " << __LINE__ << std::endl;
            exit(1);
        }

        for (int y = 0; y < _height; ++y)
        {
            for (int x = 0; x < _width; ++x)
            {
                int idx = (y * _width + x);
                batch(b, y, x) = data[idx] / 255.0f; // Normalize the pixel value
            }
        }

        // Cleanup
        stbi_image_free(data);
    }

    // Update the batch start index, if it's greater than the number of images, reset it to 0
    _batch_start_index += 1;
    if (_batch_start_index > num_batches)
    {
        _batch_start_index = 0;
    }

    return batch;
}
