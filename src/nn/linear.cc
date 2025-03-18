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

Eigen::MatrixXf Linear::backward(Eigen::MatrixXf &input, Eigen::MatrixXf &grad_output)
{
    // Gradient wrt weights: grad_output * inputT
    grad_weights = grad_output * input;

    // Gradient wrt bias: sum of grad_output along the batch dimension
    grad_bias = grad_output.rowwise().sum();

    // Gradient wrt input: weightsT * grad_output
    grad_input = weights.transpose() * grad_output;

    return grad_input;
}
