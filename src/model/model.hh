#ifndef __AUTOENCODER_MODEL_HH__
#define __AUTOENCODER_MODEL_HH__

#include <Eigen/Dense>
#include "relu.hh"
#include "linear.hh"

template <size_t max_batch_size, size_t input_dim, size_t hidden_dim, size_t output_dim>
class AutoencoderModel
{
public:
    Linear<max_batch_size, input_dim, hidden_dim> encoder;
    // ReLU<max_batch_size, hidden_dim> encoder_activation;
    // Linear<max_batch_size, hidden_dim, output_dim> decoder;
    // ReLU<max_batch_size, output_dim> decoder_activation;

    // // Cached activations from the forward pass
    // Eigen::Matrix encoded;
    // Eigen::Matrix activated_encoded;
    // Eigen::Matrix decoded;
    // Eigen::Matrix activated_decoded;

    // AutoencoderModel() {}

    // Eigen::Matrix forward(const Eigen::Matrix &input)
    // {
    //     encoded = encoder.forward(input);
    //     activated_encoded = encoder_activation.forward(encoded);
    //     decoded = decoder.forward(activated_encoded);
    //     activated_decoded = decoder_activation.forward(decoded);
    //     return activated_decoded;
    // }

    // Eigen::Matrix backward(const Eigen::Matrix &input, const Eigen::Matrix &grad_output)
    // {
    //     Eigen::Matrix grad_decoded = decoder_activation.backward(grad_output);
    //     Eigen::Matrix grad_hidden = decoder.backward(activated_encoded, grad_decoded);
    //     Eigen::Matrix grad_activated_hidden = encoder_activation.backward(grad_hidden);
    //     Eigen::Matrix grad_input = encoder.backward(input, grad_activated_hidden);
    //     return grad_input;
    // }
};

#endif // __AUTOENCODER_MODEL_HH__
