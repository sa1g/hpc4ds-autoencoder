#include "dataset.hh"
#include "rand.hh"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif // STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <algorithm>

Dataloader::Dataloader(
    const std::string &path,
    const std::vector<std::string> filenames,
    const int width,
    const int height,
    const int num_images,
    const int batch_size,
    const bool shuffle) : _path(path),
                          _filenames(filenames),
                          _width(width),
                          _height(height),
                          _num_images(num_images),
                          _batch_size(batch_size),
                          _shuffle(shuffle),
                          num_batches((_num_images + _batch_size - 1) / _batch_size),
                          _current_batch_data(batch_size, height * width) // Ceiling division to avoid missing the last batch
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

Eigen::MatrixXf &Dataloader::get_batch()
{
    // Ensure _current_batch_data is properly sized
    // _current_batch_data.resize(_batch_size, _height * _width);
    // _current_batch_data.setZero(); // Reset or populate with real data

    for (int b = 0; b < _batch_size; ++b)
    {
        int index = _batch_start_index + b; // Get the correct image index for the batch

        if (index >= _filenames.size())
        {
            // std::cerr << "Batch index out of range: " << index << std::endl;
            // exit(1);
            break;
        }

        // Load the image using stb_image
        std::string filename = _path + "/" + _filenames[index] + _extension;

        unsigned char *data = stbi_load(filename.c_str(), &_width, &_height, &_n_channels, 0);

        if (!data)
        {
            std::cerr << "Error loading image: " << _filenames[index] << "Line: " << __LINE__ << std::endl;
            exit(1);
        }

        // Copy the flattened image data directly into _current_batch_data(b)
        for (int i = 0; i < _height * _width; ++i)
        {
            _current_batch_data(b, i) = static_cast<float>(data[i]) / 255.0f; // Normalize the pixel value
        }

        // Cleanup
        stbi_image_free(data);
    }

    return _current_batch_data;
}