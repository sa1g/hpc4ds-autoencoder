#ifndef __AUTOENCODER_RELU_HH__
#define __AUTOENCODER_RELU_HH__

// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

#include <Eigen/Dense>

template <size_t max_batch_size, size_t data_dim>
class ReLU
{
public:
    Eigen::Matrix<float, Eigen::Dynamic, data_dim> forward(
        const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &input) const
    {
        return input.cwiseMax(0);
    }

    Eigen::Matrix<float, Eigen::Dynamic, data_dim> backward(
        const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &input,
        const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &grad_output) const
    {
        return (input.array() > 0).template cast<float>() * grad_output.array();
    }
};

#endif // __AUTOENCODER_RELU_HH__