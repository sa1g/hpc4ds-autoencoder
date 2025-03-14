#ifndef __AUTOENCODER_RELU_HH__
#define __AUTOENCODER_RELU_HH__

// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

#include <Eigen/Dense>

class ReLU
{
public:
    Eigen::MatrixXf forward(Eigen::MatrixXf &input);
    Eigen::MatrixXf backward(Eigen::MatrixXf &input, Eigen::MatrixXf &grand_output);
};

#endif // __AUTOENCODER_RELU_HH__