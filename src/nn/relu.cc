#include "relu.hh"

Eigen::MatrixXf ReLU::forward(Eigen::MatrixXf &input)
{
    return input.cwiseMax(0);
}

Eigen::MatrixXf ReLU::backward(Eigen::MatrixXf &input, Eigen::MatrixXf &grad_output)
{
    // Derivative of ReLU is 1 for x>0, 0 otherwise
    return (input.array() > 0).cast<float>() * grad_output.array();
}