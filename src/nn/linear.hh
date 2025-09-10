#ifndef __AUTOENCODER_LINEAR_HH__
#define __AUTOENCODER_LINEAR_HH__

#include <Eigen/Dense>
// #include "Eigen/src/Core/Matrix.h"
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
    Eigen::MatrixXf weights;
    Eigen::VectorXf bias;
    Eigen::MatrixXf grad_weights;
    Eigen::VectorXf grad_bias;
    Eigen::MatrixXf grad_input;
    
    Linear() :
        weights(output_dim, input_dim),
        bias(output_dim),
        grad_weights(output_dim, input_dim),
        grad_bias(output_dim),
        grad_input(max_batch_size, input_dim)
    
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
    Eigen::MatrixXf forward(
        const Eigen::MatrixXf &input) const
    {
        #ifdef DEBUG
        assert(input.rows() <= max_batch_size && "Batch size exceeds maximum batch size");
        assert(input.cols() == input_dim && "Input dimension mismatch");
        #endif
        Eigen::MatrixXf output(input.rows(), output_dim);
        // output.noalias() = input * weights.transpose(); // TODO: add docs
        // output.rowwise() += bias.transpose();
        output = (input * weights.transpose()) + bias.replicate(1, input.rows()).transpose();
        return output;
    }

    /**
     * @brief Backward pass of the linear layer
     * @param input Input matrix of shape [batch_size, input_dim]
     * @param grad_output Gradient of the loss with respect to the output
     * @return Gradient of the loss with respect to the input, of shape [batch_size, input_dim]
     */
    Eigen::MatrixXf backward(
        const Eigen::MatrixXf &input,
        const Eigen::MatrixXf &grad_output)
    {
        grad_weights = grad_output.transpose() * input;
        grad_bias = grad_output.colwise().sum();
        grad_input = grad_output * weights;

        return grad_input;
    }
};

#endif // __AUTOENCODER_LINEAR_HH__