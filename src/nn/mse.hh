#ifndef __AUTOENCODER_MSE_HH__
#define __AUTOENCODER_MSE_HH__

#include <Eigen/Dense>

template <size_t max_batch_size, size_t data_dim>
class MSE
{
public:
    // Each row of input/output is a different picture of size data_dim
    Eigen::VectorXf mse_loss(
        const Eigen::MatrixXf &input,
        const Eigen::MatrixXf &output)
    {
        #ifdef DEBUG
        assert(input.rows() <= max_batch_size && "Input rows exceed max batch size");
        assert(input.cols() == data_dim && "Input cols dimension mismatch");        
        assert(output.rows() <= max_batch_size && "Output rows exceed max batch size");
        assert(output.cols() == data_dim && "Output cols dimension mismatch");        
        #endif
        Eigen::MatrixXf diff = input - output;
        
        Eigen::VectorXf losses = diff.rowwise().squaredNorm() / data_dim;
        // Eigen::Matrix<float, Eigen::Dynamic, 1> losses = 
        
        return losses;
    }
};

#endif // __AUTOENCODER_MSE_HH__
