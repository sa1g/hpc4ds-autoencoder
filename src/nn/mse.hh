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
    
    /**
     * @brief Compute the mean squared error loss between input and output
     * @param input Input matrix of shape [batch_size, data_dim]
     * @param output Output matrix of shape [batch_size, data_dim]
     * @return Mean squared error loss
     * 
     * Note: this function assumes that the input and output matrices have the same shape, and that the number of rows (batch size) does not exceed max_batch_size. The number of columns (data_dim) should also match. These checks can be enabled in debug mode with assertions.
     */
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

    /**
     * @brief Compute the gradient of the MSE loss with respect to the output
     * @param target Target matrix of shape [batch_size, data_dim]
     * @param prediction Predicted matrix of shape [batch_size, data_dim]
     * @return Gradient of the loss with respect to the prediction, of shape [batch_size, data_dim]
     */
    Eigen::MatrixXf mse_gradient(const Eigen::MatrixXf &target, const Eigen::MatrixXf &prediction){
        return 2.0f * (prediction - target) / (target.rows() * target.cols());
    }
};

#endif // __AUTOENCODER_MSE_HH__
