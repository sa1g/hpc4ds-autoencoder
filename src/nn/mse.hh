#ifndef __AUTOENCODER_MSE_HH__
#define __AUTOENCODER_MSE_HH__

#include <Eigen/Dense>

template <size_t max_batch_size, size_t data_dim>
class MSE
{
public:
    // Each row of input/output is a different picture of size data_dim
    Eigen::Matrix<float, Eigen::Dynamic, 1> mse_loss(
        const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &input,
        const Eigen::Matrix<float, Eigen::Dynamic, data_dim> &output)
    {
        Eigen::Matrix<float, Eigen::Dynamic, data_dim> diff = input - output;
        Eigen::Matrix<float, Eigen::Dynamic, 1> losses = diff.rowwise().squaredNorm() / data_dim;
        return losses;
    }
};

#endif // __AUTOENCODER_MSE_HH__
