#ifndef __AUTOENCODER_LINEAR_HH__
#define __AUTOENCODER_LINEAR_HH__

#include <Eigen/Dense>

class Linear
{
public:
    Eigen::MatrixXf weights;
    Eigen::MatrixXf bias;

    Eigen::MatrixXf grad_weights;
    Eigen::VectorXf grad_bias;
    Eigen::MatrixXf grad_input;

    Linear(int input_dim, int output_dim);
    Eigen::MatrixXf forward(Eigen::MatrixXf &input);
    Eigen::MatrixXf backward(Eigen::MatrixXf &input, Eigen::MatrixXf &grad_output);
};

#endif // __AUTOENCODER_LINEAR_HH__