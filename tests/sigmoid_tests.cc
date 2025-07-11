#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "sigmoid.hh"

// Test that Sigmoid is correctly applied
TEST(SigmoidLayerTest, ForwardPass)
{
    constexpr size_t max_batch_size = 10;
    constexpr size_t data_dim = 3;

    Sigmoid<max_batch_size, data_dim> sigmoid;
    Eigen::MatrixXf input(2, 3);
    input << -1, 2, -3, 4, -5, 6;

    Eigen::MatrixXf output = sigmoid.forward(input);

    Eigen::MatrixXf expected_output(2, 3);
    expected_output <<  0.26894142137, 0.88079707797788, 0.047425873177567, 0.98201379003791, 0.0066928509242849, 0.99752737684337;

    EXPECT_TRUE(output.isApprox(expected_output));
}

// Test gradient computation in backward pass
TEST(SigmoidLayerTest, BackwardPass)
{
    constexpr size_t max_batch_size = 10;
    constexpr size_t data_dim = 3;

    Sigmoid<max_batch_size, data_dim> sigmoid;
    Eigen::MatrixXf input(2, 3);
    input << 1, -2, 3, -4, 5, -6;

    Eigen::MatrixXf grad_output(2, 3);
    grad_output << 1, 1, 1, 1, 1, 1;

    Eigen::MatrixXf grad_input = sigmoid.backward(input, grad_output);

    // Expected computation: sigmoid(input) * (1 - sigmoid(input)) * grad_output;
    Eigen::MatrixXf expected_grad(2, 3);
    expected_grad << 0.19661193324148185, 0.1049935854035065, 0.045176659730912, 0.017662706213291118, 0.006648056670790033, 0.002466509291360048;

    EXPECT_TRUE(grad_input.isApprox(expected_grad));
}