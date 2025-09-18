#include "dataset.hh"
#include "rand.hh"
#include <cstdlib>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif // STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <iostream>

#include <omp.h>

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

  // Allocate prefetch buffer same shape as current batch
    _prefetch_buffer.resize(batch_size, height * width);

    // Launch background worker
    _worker = std::thread([this]{ prefetch_loop(); });

}

void Dataloader::prefetch_loop()
{
    while (!_stop)
    {
        // Wait until consumer takes the last batch
        std::unique_lock<std::mutex> lock(_mtx);
        _cv.wait(lock, [this]{ return !_ready || _stop; });
        if (_stop) break;
        lock.unlock();

        // Fill prefetch buffer (same code as get_batch, but into _prefetch_buffer)
        #pragma omp parallel for
        for (int b = 0; b < _batch_size; ++b)
        {
            int index = _batch_start_index + b;
            if (index >= _filenames.size()) continue;

            const std::string &filename = _full_paths[index];
            int width, height, n_channels;
            unsigned char *data = stbi_load(filename.c_str(), &width, &height, &n_channels, 0);

            if (!data) {
                #pragma omp critical
                std::cerr << "Error loading image: " << filename << " Line: " << __LINE__ << std::endl;
                continue;
            }

            Eigen::Map<Eigen::VectorXf>(_prefetch_buffer.row(b).data(), width * height) =
                Eigen::Map<Eigen::Array<unsigned char, Eigen::Dynamic, 1>>(data, width * height)
                    .cast<float>() / 255.0f;

            stbi_image_free(data);
        }

        // Mark ready
        lock.lock();
        _ready = true;
        lock.unlock();
        _cv.notify_one();
    }
}

Eigen::MatrixXf &Dataloader::get_batch()
{
    std::unique_lock<std::mutex> lock(_mtx);
    _cv.wait(lock, [this]{ return _ready || _stop; });

    // Swap prefetch buffer into current batch
    _current_batch_data.swap(_prefetch_buffer);
    _ready = false;
    lock.unlock();
    _cv.notify_one();

    return _current_batch_data;
}

Dataloader::~Dataloader()
{
    _stop = true;
    _cv.notify_all();
    if (_worker.joinable())
        _worker.join();
}



// Eigen::MatrixXf &Dataloader::get_batch() {
//   // Ensure _current_batch_data is properly sized
//   // _current_batch_data.resize(_batch_size, _height * _width);
//   // _current_batch_data.setZero(); // Reset or populate with real data

// #pragma omp parallel for
//   for (int b = 0; b < _batch_size; ++b) {
//     int index =
//         _batch_start_index + b; // Get the correct image index for the batch

//     if (index >= _filenames.size()) {
//       continue;
//     }

//     // const std::string filename = _path + "/" + _filenames[index] +
//     // _extension;
//     const std::string &filename = _full_paths[index];

//     unsigned char *data =
//         stbi_load(filename.c_str(), &_width, &_height, &_n_channels, 0);

//     if (!data) {
// #pragma omp critical
//       std::cerr << "Error loading image: " << _filenames[index]
//                 << "Line: " << __LINE__ << std::endl;
//       continue;
//     }

//     // // Copy the flattened image data directly into _current_batch_data(b)
//     // for (int i = 0; i < _height * _width; ++i) {
//     //   _current_batch_data(b, i) =
//     //       static_cast<float>(data[i]) / 255.0f; // Normalize the pixel value
//     // }

//     Eigen::Map<Eigen::VectorXf>(_current_batch_data.row(b).data(), _width * _height) = Eigen::Map<Eigen::Array<unsigned char, Eigen::Dynamic, 1>>(data, _width * _height).cast<float>() / 255.0f;

//     // Cleanup
//     stbi_image_free(data);
//   }

//   return _current_batch_data;
// }