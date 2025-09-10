#ifndef __AUTOENCODER_MSE_HH__
#define __AUTOENCODER_MSE_HH__

#include <Eigen/Dense>

class MSE
{
private:
    size_t max_batch_size;
    size_t data_dim;    

public:
    MSE(size_t max_batch_size, size_t data_dim) : max_batch_size(max_batch_size), data_dim(data_dim) {}

    // Each row of input/output is a different picture of size data_dim
    float mse_loss(
        const Eigen::MatrixXf &input,
        const Eigen::MatrixXf &output)
    {
        #ifdef DEBUG
        assert(input.rows() <= max_batch_size && "Input rows exceed max batch size");
        assert(input.cols() == data_dim && "Input cols dimension mismatch");        
        assert(output.rows() <= max_batch_size && "Output rows exceed max batch size");
        assert(output.cols() == data_dim && "Output cols dimension mismatch");        
        #endif
        
        auto losses = (input - output).squaredNorm() / (input.rows() * input.cols());
        
        return losses;
    }

    Eigen::MatrixXf mse_gradient(const Eigen::MatrixXf &target, const Eigen::MatrixXf &prediction){
        return 2.0f * (prediction - target) / (target.rows() * target.cols());
    }
};

#endif // __AUTOENCODER_MSE_HH__
