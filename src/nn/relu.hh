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
template <size_t max_batch_size, size_t data_dim>
class ReLU
{
public:
    /**
     * @brief Forward pass of the ReLU activation function
     * @param input Input matrix of shape [batch_size, data_dim]
     * @return Output matrix of the same shape, with ReLU applied
     */
    Eigen::Matrix<float, Eigen::Dynamic, data_dim> forward(
        const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &input) const
    {
        return input.cwiseMax(0);
    }

    /**
     * @brief Backward pass of the ReLU activation function
     * @param input Input matrix of shape [batch_size, data_dim]
     * @param grad_output Gradient of the loss with respect to the output
     * @return Gradient of the loss with respect to the input, of the same shape
     */
    Eigen::Matrix<float, Eigen::Dynamic, data_dim> backward(
        const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &input,
        const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &grad_output) const
    {
        return (input.array() > 0).template cast<float>() * grad_output.array();
    }
};

#endif // __AUTOENCODER_RELU_HH__