#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "mse.hh"

// Test the loss function on different types of values
TEST(MSETest, DiverseLossCases)
{
    constexpr size_t max_batch_size = 10;
    constexpr size_t data_dim = 3;

    MSE<max_batch_size, data_dim> mse;

    Eigen::MatrixXf input(3, 3);
    Eigen::MatrixXf output(3, 3);

    // MSE should be 0
    input.row(0) << 1, -2, 3;
    output.row(0) << 1, -2, 3;

    // MSE should be 1, the average error is 1
    input.row(1) << 1, 2, 3;
    output.row(1) << 2, 3, 4;

    // Random values
    input.row(2) << 0, -1, 2;
    output.row(2) << 1, -2, 0;

    Eigen::Matrix<float, Eigen::Dynamic, 1> losses = mse.mse_loss(input, output);

    // Ground truth
    Eigen::VectorXf expected(3);
    expected(0) = 0.0f;
    expected(1) = (1 + 1 + 1) / 3.0f;
    expected(2) = (1 + 1 + 4) / 3.0f;

    // Check with a small tolerance
    EXPECT_TRUE(losses.isApprox(expected, 1e-4));
}
