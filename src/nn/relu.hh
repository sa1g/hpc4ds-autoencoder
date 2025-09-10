#ifndef __AUTOENCODER_RELU_HH__
#define __AUTOENCODER_RELU_HH__

// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

#include <Eigen/Dense>

/**
 * @brief ReLU activation function class
 *
 * This class implements the ReLU activation function, which is defined as:
 * f(x) = max(0, x)
 */
class ReLU {
public:
  size_t max_batch_size;
  size_t data_dim;

  /**
   * @brief Constructor for ReLU activation function
   * @param max_batch_size Maximum batch size supported
   * @param data_dim Dimension of input data
   */
  ReLU(size_t max_batch_size, size_t data_dim)
      : max_batch_size(max_batch_size), data_dim(data_dim) {
    // No heap allocation needed for ReLU since it's stateless
    // The parameters are just for validation
  }

  /**
   * @brief Forward pass of the ReLU activation function
   * @param input Input matrix of shape [batch_size, data_dim]
   * @return Output matrix of the same shape, with ReLU applied
   */
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) const {
#ifdef DEBUG
    assert(input.rows() <= max_batch_size &&
           "Batch size exceeds maximum batch size");
    assert(input.cols() == data_dim && "Input dimension mismatch");
#endif
    return input.cwiseMax(0);
  }

  /**
   * @brief Backward pass of the ReLU activation function
   * @param input Input matrix of shape [batch_size, data_dim]
   * @param grad_output Gradient of the loss with respect to the output
   * @return Gradient of the loss with respect to the input, of the same shape
   */
  Eigen::MatrixXf backward(const Eigen::MatrixXf &input,
                           const Eigen::MatrixXf &grad_output) const {
    // return (input.array() > 0).template cast<float>() * grad_output.array();
    return (input.array()>0).select(grad_output, 0);
  }
};

#endif // __AUTOENCODER_RELU_HH__