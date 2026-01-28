#ifndef __AUTOENCODER_SIGMOID_HH__
#define __AUTOENCODER_SIGMOID_HH__

// https://en.wikipedia.org/wiki/Sigmoid_function

#include <Eigen/Dense>

class Sigmoid
{
public:
    size_t max_batch_size;
    size_t data_dim;

    Sigmoid(size_t max_batch_size, size_t data_dim) : max_batch_size(max_batch_size), data_dim(data_dim) {}

    Eigen::MatrixXf forward(
        const Eigen::MatrixXf &input) const
    {
#ifdef DEBUG
        assert(input.rows() <= max_batch_size && "Batch size exceeds maximum batch size");
        assert(input.cols() == data_dim && "Input dimension mismatch");
#endif
        return ((input.array() * -1).exp() + 1).inverse().matrix();
    }

    Eigen::MatrixXf backward(
        const Eigen::MatrixXf &input,
        const Eigen::MatrixXf &grad_output) const
    {
        return ((input.array() * -1).exp() + 1).inverse() * (1 - ((input.array() * -1).exp() + 1).inverse()) * grad_output.array();
    }
};

#endif // __AUTOENCODER_SIGMOID_HH__
