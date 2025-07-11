#ifndef __AUTOENCODER_LINEAR_HH__
#define __AUTOENCODER_LINEAR_HH__

#include <Eigen/Dense>
#include "common.hh"

/**
 * @brief Linear layer class
 *
 * This class implements a linear layer with weights and bias, and provides methods for forward and backward passes.
 */
template <size_t max_batch_size, size_t input_dim, size_t output_dim>
class Linear
{
public:
    Eigen::Matrix<float, output_dim, input_dim> weights;
    Eigen::Matrix<float, output_dim, 1> bias;

    Eigen::Matrix<float, output_dim, input_dim> grad_weights;
    Eigen::Matrix<float, output_dim, 1> grad_bias;
    Eigen::Matrix<float, Eigen::Dynamic, input_dim> grad_input;

    Linear()
    {
        weights.setRandom();
        bias.setRandom();
        grad_weights.setZero();
        grad_bias.setZero();
        grad_input.setZero();
    }

    /**
     * @brief Forward pass of the linear layer
     * @param input Input matrix of shape [batch_size, input_dim]
     * @return Output matrix of shape [batch_size, output_dim]
     */
    Eigen::Matrix<float, Eigen::Dynamic, output_dim> forward(
        const Eigen::Matrix<float, Eigen::Dynamic, input_dim> &input) const
    {
        assert(input.rows() <= max_batch_size && "Batch size exceeds maximum batch size");

        Eigen::Matrix<float, Eigen::Dynamic, output_dim> output;
        output = (input * weights.transpose()) + bias.replicate(1, input.rows()).transpose();
        return output;
    }

    /**
     * @brief Backward pass of the linear layer
     * @param input Input matrix of shape [batch_size, input_dim]
     * @param grad_output Gradient of the loss with respect to the output
     * @return Gradient of the loss with respect to the input, of shape [batch_size, input_dim]
     */
    Eigen::Matrix<float, Eigen::Dynamic, input_dim> backward(
        const Eigen::Matrix<float, Eigen::Dynamic, input_dim> &input,
        const Eigen::Matrix<float, Eigen::Dynamic, output_dim> &grad_output)
    {
        grad_weights = grad_output.transpose() * input;
        grad_bias = grad_output.colwise().sum();
        grad_input = grad_output * weights;

        return grad_input;
    }
};

#endif // __AUTOENCODER_LINEAR_HH__