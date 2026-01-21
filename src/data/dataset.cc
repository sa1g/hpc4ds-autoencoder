#include "dataset.hh"
#include "rand.hh"
#include <cstdlib>

#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <iostream>

Dataloader::Dataloader(const std::string &path,
                       const std::vector<std::string> filenames,
                       const int width, const int height, const int n_channels,
                       const int num_images, const int batch_size,
                       const bool shuffle): _path(path), _filenames(filenames), _width(width), _height(height),
      _n_channels(n_channels), _num_images(num_images), _batch_size(batch_size),
      _shuffle(shuffle),
      num_batches((_num_images + _batch_size - 1) / _batch_size),
      _current_batch_data(
          batch_size,
          height * width) // Ceiling division to avoid missing the last batch
{

  if (_num_images > _filenames.size()) {
    std::cerr << "Number of images is greater than the number of filenames"
              << std::endl;
    exit(1);
  }

  if (_shuffle) {
    std::shuffle(std::begin(_filenames), std::end(_filenames),
                 autoencoder_random_generator);
  }

  _full_paths.reserve(_filenames.size());
  for (auto &fname : _filenames) {
    _full_paths.push_back(_path + "/" + fname + _extension);
  }
}

Eigen::MatrixXf &Dataloader::get_batch() {

#pragma omp parallel for
  for (int b = 0; b < _batch_size; ++b) {
    int index =
        _batch_start_index + b; // Get the correct image index for the batch

    if (index >= _filenames.size()) {
      continue;
    }

    const std::string &filename = _full_paths[index];

    unsigned char *data =
        stbi_load(filename.c_str(), &_width, &_height, &_n_channels, 0);

    if (!data) {
#pragma omp critical
      std::cerr << "Error loading image: " << _filenames[index]
                << "Line: " << __LINE__ << std::endl;
      continue;
    }

    // Copy the flattened image data directly into _current_batch_data(b)
    for (int i = 0; i < _height * _width; ++i) {
      _current_batch_data(b, i) =
          static_cast<float>(data[i]) / 255.0f; // Normalize the pixel value
    }

    // Cleanup
    stbi_image_free(data);
  }

  return _current_batch_data;
}

bool Dataloader::save_batch_image(const Eigen::MatrixXf &batch, int batch_index,
                                  const std::string &output_path) {
  if (batch_index < 0 || batch_index >= _batch_size) {
    std::cerr << "Error: batch_index " << batch_index
              << " out of bounds. Batch size: " << _batch_size << std::endl;
    return false;
  }

  if (batch.rows() != _batch_size || batch.cols() != _width * _height) {
    std::cerr << "Error: Batch dimensions mismatch. Expected (" << _batch_size
              << ", " << _width * _height << "), got (" << batch.rows() << ", "
              << batch.cols() << ")" << std::endl;
    return false;
  }

  // Extract the image data from the batch
  std::vector<unsigned char> image_data(_width * _height);

  for (int i = 0; i < _width * _height; ++i) {
    float pixel_value = batch(batch_index, i);
    // De-normalize (assuming original normalization was /255.0f)
    pixel_value = std::clamp(pixel_value, 0.0f, 1.0f);
    image_data[i] = static_cast<unsigned char>(pixel_value * 255.0f);
  }

  // Save using stb_image_write
  int result = stbi_write_png(output_path.c_str(), _width, _height,
                              1, // grayscale
                              image_data.data(),
                              _width); // stride

  if (result == 0) {
    std::cerr << "Error saving image to: " << output_path << std::endl;
    return false;
  }

  std::cout << "Saved image to: " << output_path << std::endl;
  return true;
}