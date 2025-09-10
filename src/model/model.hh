#ifndef __AUTOENCODER_MODEL_HH__
#define __AUTOENCODER_MODEL_HH__

#include <Eigen/Dense>
#include "Eigen/src/Core/Matrix.h"
#include "relu.hh"
#include "linear.hh"

class AutoencoderModel
{
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

    AutoencoderModel(
        size_t max_batch_size, 
        size_t input_dim, 
        size_t hidden_dim, 
        size_t output_dim) : 
        max_batch_size(max_batch_size), 
        input_dim(input_dim), 
        hidden_dim(hidden_dim), 
        output_dim(output_dim),
        encoder(max_batch_size, input_dim, hidden_dim),
        encoder_activation(max_batch_size, hidden_dim),
        decoder(max_batch_size, hidden_dim, output_dim),
        decoder_activation(max_batch_size, output_dim)    
    {}

    Eigen::MatrixXf forward(const Eigen::MatrixXf &input)
    {
        encoded = encoder.forward(input);
        activated_encoded = encoder_activation.forward(encoded);
        decoded = decoder.forward(activated_encoded);
        activated_decoded = decoder_activation.forward(decoded);
        return activated_decoded;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf &input, const Eigen::MatrixXf &grad_output)
    {
        Eigen::MatrixXf grad_decoded = decoder_activation.backward(decoded, grad_output);
        Eigen::MatrixXf grad_hidden = decoder.backward(activated_encoded, grad_decoded);
        Eigen::MatrixXf grad_activated_hidden = encoder_activation.backward(encoded, grad_hidden);
        Eigen::MatrixXf grad_input = encoder.backward(input, grad_activated_hidden);
        
        return grad_input;
    }
};

#endif // __AUTOENCODER_MODEL_HH__
