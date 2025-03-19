#ifndef __AUTOENCODER_OPTIM_HH__
#define __AUTOENCODER_OPTIM_HH__

#include <concepts>

#include "linear.hh"

template <typename T>
concept IsLinear = requires {
    typename T::weights;
    typename T::bias;
    typename T::grad_weights;
    typename T::grad_bias;
    typename T::grad_input;
};

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

#endif __AUTOENCODER_OPTIM_HH__