#include "dataset.hh"
#include "rand.hh"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif // STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <algorithm>

Dataloader::Dataloader(const std::string &path, const std::vector<std::string> filenames, const int width, const int height, const int num_images, const int batch_size, const bool shuffle) : _path(path), _filenames(filenames), _width(width), _height(height), _num_images(num_images), _batch_size(batch_size), _shuffle(shuffle), num_batches((_num_images + _batch_size - 1) / _batch_size), _current_batch_data(batch_size, height, width) // Ceiling division to avoid missing the last batch
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

// Eigen::Tensor<float, 3> Dataloader::get_batch()
Eigen::Tensor<float, 3>& Dataloader::get_batch()
{
    // Shape [BATCH_SIZE, HEIGHT, WIDTH]
    // Eigen::Tensor<float, 3> batch(_batch_size, _height, _width);
    // _current_batch_data.setZero(); // Reset or populate with real data

    for (int b = 0; b < _batch_size; ++b)
    {
        int index = _batch_start_index + b; // Get the correct image index for the batch

        if (index >= _filenames.size())
        {
            std::cerr << "Batch index out of range: " << index << std::endl;
            exit(1);
        }

        // Load the image using stb_image
        std::string filename = _path + "/" + _filenames[index] + _extension;

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
                _current_batch_data(b, y, x) = data[idx] / 255.0f; // Normalize the pixel value
            }
        }

        // Cleanup
        stbi_image_free(data);
    }

    return _current_batch_data;
}
