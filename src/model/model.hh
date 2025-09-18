#ifndef __AUTOENCODER_MODEL_HH__
#define __AUTOENCODER_MODEL_HH__

#include "linear.hh"
#include "relu.hh"
#include <Eigen/Dense>
#include <unordered_map>

#include <fstream>
#include <unordered_map>

/**
 * Simple autoencoder model with linear input and output.
 *
 * Consider that this works without any special management both for
 * training and inference. The only thing you need to care about is
 * making sure that there's no concurrency (e.g. training and testing
 * at the same time), otherwise the temp. values will explode (overwritten).
 *
 * Have fun! - E
 */
class AutoencoderModel {
private:
  size_t max_batch_size;
  size_t input_dim;
  size_t hidden_dim;
  size_t output_dim;

public:
  Linear encoder{max_batch_size, input_dim, hidden_dim};
  ReLU encoder_activation{max_batch_size, hidden_dim};
  Linear decoder{max_batch_size, hidden_dim, output_dim};
  ReLU decoder_activation{max_batch_size, output_dim};

  // // Cached activations from the forward pass
  Eigen::MatrixXf encoded;
  Eigen::MatrixXf activated_encoded;
  Eigen::MatrixXf decoded;
  Eigen::MatrixXf activated_decoded;

  AutoencoderModel(size_t max_batch_size, size_t input_dim, size_t hidden_dim,
                   size_t output_dim)
      : max_batch_size(max_batch_size), input_dim(input_dim),
        hidden_dim(hidden_dim), output_dim(output_dim),
        encoder(max_batch_size, input_dim, hidden_dim),
        encoder_activation(max_batch_size, hidden_dim),
        decoder(max_batch_size, hidden_dim, output_dim),
        decoder_activation(max_batch_size, output_dim) {}

  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) {
    encoded = encoder.forward(input);
    activated_encoded = encoder_activation.forward(encoded);
    decoded = decoder.forward(activated_encoded);
    activated_decoded = decoder_activation.forward(decoded);
    return activated_decoded;
  }

  Eigen::MatrixXf backward(const Eigen::MatrixXf &input,
                           const Eigen::MatrixXf &grad_output) {
    Eigen::MatrixXf grad_decoded =
        decoder_activation.backward(decoded, grad_output);
    Eigen::MatrixXf grad_hidden =
        decoder.backward(activated_encoded, grad_decoded);
    Eigen::MatrixXf grad_activated_hidden =
        encoder_activation.backward(encoded, grad_hidden);
    Eigen::MatrixXf grad_input = encoder.backward(input, grad_activated_hidden);

    return grad_input;
  }

  std::unordered_map<std::string, Eigen::MatrixXf> get_weights() {
    // We have an encoder and a decoder. We need to store both weights and bias
    // std::unordered_map<std::string, Eigen::MatrixXf> weights

    return {
        {"encoder_w", encoder.weights},
        {"encoder_b", encoder.bias},
        {"decoder_w", decoder.weights},
        {"decoder_b", decoder.bias},
    };
  }

  void set_weights(std::unordered_map<std::string, Eigen::MatrixXf> weights) {
    encoder.weights = weights["encoder_w"];
    encoder.bias = weights["encoder_b"];
    encoder.grad_weights.setZero();
    encoder.grad_bias.setZero();
    encoder.grad_input.setZero();

    decoder.weights = weights["decoder_w"];
    decoder.bias = weights["decoder_b"];
    encoder.grad_weights.setZero();
    encoder.grad_bias.setZero();
    encoder.grad_input.setZero();
  }

  void save_weights(std::string path) {
    auto weights = get_weights();
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // Write each matrix to the file
    for (const auto &[name, matrix] : weights) {
      // Write the name (as a null-terminated string)
      out.write(name.c_str(), name.size() + 1);

      // Write the dimensions
      int rows = matrix.rows();
      int cols = matrix.cols();
      out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
      out.write(reinterpret_cast<const char *>(&cols), sizeof(int));

      // Write the matrix data
      out.write(reinterpret_cast<const char *>(matrix.data()),
                rows * cols * sizeof(float));
    }
  }

  std::unordered_map<std::string, Eigen::MatrixXf>
  load_weights(std::string path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("Failed to open file for reading: " + path);
    }

    std::unordered_map<std::string, Eigen::MatrixXf> weights;

    while (in.peek() != EOF) {
      // Read the name (null-terminated string)
      std::string name;
      std::getline(in, name, '\0');

      // Read the dimensions
      int rows, cols;
      in.read(reinterpret_cast<char *>(&rows), sizeof(int));
      in.read(reinterpret_cast<char *>(&cols), sizeof(int));

      // Read the matrix data
      Eigen::MatrixXf matrix(rows, cols);
      in.read(reinterpret_cast<char *>(matrix.data()),
              rows * cols * sizeof(float));

      weights[name] = matrix;
    }

    return weights;
  }
};

#endif // __AUTOENCODER_MODEL_HH__
