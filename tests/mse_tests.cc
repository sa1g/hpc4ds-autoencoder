#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "mse.hh"

TEST(MSETest, DiverseLossCases)
{
    constexpr size_t max_batch_size = 3; // Match actual batch size
    constexpr size_t data_dim = 3;

    MSE mse{max_batch_size, data_dim};

    // Test case 1: Perfect match (MSE = 0)
    {
        Eigen::MatrixXf input(1, 3);
        Eigen::MatrixXf target(1, 3);
        input << 1, -2, 3;
        target << 1, -2, 3;

        float loss = mse.mse_loss(input, target);
        EXPECT_NEAR(loss, 0.0f, 1e-4f);
    }

    // Test case 2: Constant offset (MSE = 1)
    {
        Eigen::MatrixXf input(1, 3);
        Eigen::MatrixXf target(1, 3);
        input << 1, 2, 3;
        target << 2, 3, 4;

        float loss = mse.mse_loss(input, target);
        float expected = (1 * 1 + 1 * 1 + 1 * 1) / 3.0f; // (1+1+1)/3 = 1.0
        EXPECT_NEAR(loss, expected, 1e-4f);
    }

    // Test case 3: Mixed errors (MSE = (1+1+4)/3 = 2.0)
    {
        Eigen::MatrixXf input(1, 3);
        Eigen::MatrixXf target(1, 3);
        input << 0, -1, 2;
        target << 1, -2, 0;

        float loss = mse.mse_loss(input, target);
        float expected = (1 * 1 + 1 * 1 + 2 * 2) / 3.0f; // (1+1+4)/3 = 2.0
        EXPECT_NEAR(loss, expected, 1e-4f);
    }
}
