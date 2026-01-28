#include "dataset.hh"
#include "rand.hh"
#include <cstdlib>

#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <iostream>

Dataloader::Dataloader(const std::string &path,
                       const std::vector<std::string> filenames,
                       const int width, const int height, const int num_images,
                       const int batch_size, const bool shuffle)
    : _path(path), _filenames(filenames), _width(width), _height(height),
      _num_images(num_images), _batch_size(batch_size), _shuffle(shuffle),
      num_batches((_num_images + _batch_size - 1) / _batch_size),
      _current_batch_data(
          batch_size,
          height * width) // Ceiling division to avoid missing the last batch
{

  if (_num_images > _filenames.size())
  {
    std::cerr << "Number of images is greater than the number of filenames"
              << std::endl;
    exit(1);
  }

  if (_shuffle)
  {
    std::shuffle(std::begin(_filenames), std::end(_filenames),
                 autoencoder_random_generator);
  }

  _full_paths.reserve(_filenames.size());
  for (auto &fname : _filenames)
  {
    _full_paths.push_back(_path + "/" + fname + _extension);
  }
}

Eigen::MatrixXf &Dataloader::get_batch()
{
#pragma omp parallel for
  for (int b = 0; b < _batch_size; ++b)
  {
    int index =
        _batch_start_index + b; // Get the correct image index for the batch

    if (index >= _filenames.size())
    {
      continue;
    }

    const std::string &filename = _full_paths[index];

    int w{_width}, h{_height}, n{_n_channels};
    unsigned char *data =
        stbi_load(filename.c_str(), &w, &h, &n, 0);

    if (!data)
    {
#pragma omp critical
      std::cerr << "Error loading image: " << _filenames[index]
                << "Line: " << __LINE__ << std::endl;
      continue;
    }

    // Copy the flattened image data directly into _current_batch_data(b)
    for (int i = 0; i < _height * _width; ++i)
    {
      _current_batch_data(b, i) =
          static_cast<float>(data[i]) / 255.0f; // Normalize the pixel value
    }

    // Cleanup
    stbi_image_free(data);
  }

  return _current_batch_data;
}

bool Dataloader::save_batch_image(const Eigen::MatrixXf &batch, int batch_index,
                                  const std::string &output_path)
{
  if (batch_index < 0 || batch_index >= batch.rows())
  {
    std::cerr << "Error: batch_index out of bounds\n";
    return false;
  }

  int expected_cols = _width * _height * _n_channels;
  if (batch.cols() != expected_cols)
  {
    std::cerr << "Error: Batch dimensions mismatch. Expected cols = "
              << expected_cols << " got " << batch.cols() << "\n";
    return false;
  }

  std::vector<unsigned char> image_data(expected_cols);

  for (int i = 0; i < expected_cols; ++i)
  {
    float v = std::clamp(batch(batch_index, i), 0.0f, 1.0f);
    image_data[i] = static_cast<unsigned char>(v * 255.0f);
  }

  int stride = _width * _n_channels;

  int result = stbi_write_png(output_path.c_str(), _width, _height, _n_channels,
                              image_data.data(), stride);

  if (!result)
  {
    std::cerr << "Error saving image to: " << output_path << "\n";
    return false;
  }

  std::cout << "Saved image to: " << output_path << "\n";
  return true;
}