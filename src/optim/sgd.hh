#ifndef __AUTOENCODER_OPTIM_HH__
#define __AUTOENCODER_OPTIM_HH__

#include <concepts>

#include "linear.hh"

template <typename T>
concept IsLinear = requires(T layer) {
    requires std::same_as<decltype(layer.weights), Eigen::MatrixXf>;
    requires std::same_as<decltype(layer.bias), Eigen::VectorXf>;
    requires std::same_as<decltype(layer.grad_weights), Eigen::MatrixXf>;
    requires std::same_as<decltype(layer.grad_bias), Eigen::VectorXf>;
    requires std::same_as<decltype(layer.grad_input), Eigen::MatrixXf>;
};

/**
 * @brief Stochastic Gradient Descent optimizer for linear layers
 * 
 * @param layer Linear layer to optimize
 * @param learning_rate Learning rate for the update
 * This function updates the weights and biases of the layer using the gradients computed during the backward pass, and then resets the gradients to zero for the next iteration.   
 * 
 * Note: This function assumes that the gradients have already been computed and stored in the layer's grad_weights and grad_bias members. It also assumes that the weights and biases are stored in the layer's weights and bias members, respectively.
 */
template <IsLinear Layer>
void sgd(Layer &layer, float learning_rate)
{
    layer.weights -= learning_rate * layer.grad_weights;
    layer.bias -= learning_rate * layer.grad_bias;
    layer.grad_weights.setZero();
    layer.grad_bias.setZero();
}

template <IsLinear... Layers>
void sgd(float learning_rate, Layers &...layers)
{
    (sgd(layers, learning_rate), ...);
}

#endif // __AUTOENCODER_OPTIM_HH__