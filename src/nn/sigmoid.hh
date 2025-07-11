#ifndef __AUTOENCODER_SIGMOID_HH__
#define __AUTOENCODER_SIGMOID_HH__

// https://en.wikipedia.org/wiki/Sigmoid_function

#include <Eigen/Dense>

template <size_t max_batch_size, size_t data_dim>
class Sigmoid
{
    public:
        Eigen::Matrix<float, Eigen::Dynamic, data_dim> forward(
            const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &input) const
        {
            return ((input.array() * -1).exp() + 1).inverse().matrix();
        }
    
        Eigen::Matrix<float, Eigen::Dynamic, data_dim> backward(
            const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &input,
            const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &grad_output) const
        {
            return ((input.array() * -1).exp() + 1).inverse() * (1 - ((input.array() * -1).exp() + 1).inverse()) * grad_output.array();
        }
    };

#endif // __AUTOENCODER_SIGMOID_HH__
