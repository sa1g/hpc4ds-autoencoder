#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "relu.hh"

// Test that ReLU corretly applied max(0,x)
TEST(ReLULayerTest, ForwardPass)
{
    constexpr size_t max_batch_size = 10;
    constexpr size_t data_dim = 3;

    ReLU relu{max_batch_size, data_dim};
    Eigen::MatrixXf input(2, 3);
    input << -1, 2, -3, 4, -5, 6;

    Eigen::MatrixXf output = relu.forward(input);

    Eigen::MatrixXf expected_output(2, 3);
    expected_output << 0, 2, 0, 4, 0, 6;

    EXPECT_TRUE(output.isApprox(expected_output));
}

// Test gradient computation in backward pass
TEST(ReLULayerTest, BackwardPass)
{
    constexpr size_t max_batch_size = 10;
    constexpr size_t data_dim = 3;

    ReLU relu{max_batch_size, data_dim};
    Eigen::MatrixXf input(2, 3);
    input << 1, -2, 3, -4, 5, -6;

    Eigen::MatrixXf grad_output(2, 3);
    grad_output << 1, 1, 1, 1, 1, 1;

    Eigen::MatrixXf grad_input = relu.backward(input, grad_output);

    // Expected derivative: 1 if input > 0, 0 otherwise
    Eigen::MatrixXf expected_grad(2, 3);
    expected_grad << 1, 0, 1, 0, 1, 0;

    EXPECT_TRUE(grad_input.isApprox(expected_grad));
}