#include "linear.hh"

#include <iostream>

Linear::Linear(int input_dim, int output_dim)
{
    weights = Eigen::MatrixXf::Random(output_dim, input_dim) * sqrt(2.0 / input_dim);
    bias = Eigen::VectorXf::Zero(output_dim);
}

Eigen::MatrixXf Linear::forward(Eigen::MatrixXf &input)
{
    return ((weights * input.transpose()) + bias.replicate(1, input.rows())).transpose();
}