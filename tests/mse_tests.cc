#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "mse.hh"

TEST(MSETest, DiverseLossCases)
{
    constexpr size_t max_batch_size = 3;  // Match actual batch size
    constexpr size_t data_dim = 3;

    MSE<max_batch_size, data_dim> mse;

    // Test case 1: Perfect match (MSE = 0)
    {
        Eigen::MatrixXf input(1, 3);
        Eigen::MatrixXf target(1, 3);
        input << 1, -2, 3;
        target << 1, -2, 3;
        
        Eigen::VectorXf loss = mse.mse_loss(input, target);
        EXPECT_NEAR(loss(0), 0.0f, 1e-4f);
    }

    // Test case 2: Constant offset (MSE = 1)
    {
        Eigen::MatrixXf input(1, 3);
        Eigen::MatrixXf target(1, 3);
        input << 1, 2, 3;
        target << 2, 3, 4;
        
        Eigen::VectorXf loss = mse.mse_loss(input, target);
        float expected = (1*1 + 1*1 + 1*1) / 3.0f; // (1+1+1)/3 = 1.0
        EXPECT_NEAR(loss(0), expected, 1e-4f);
    }

    // Test case 3: Mixed errors (MSE = (1+1+4)/3 = 2.0)
    {
        Eigen::MatrixXf input(1, 3);
        Eigen::MatrixXf target(1, 3);
        input << 0, -1, 2;
        target << 1, -2, 0;
        
        Eigen::VectorXf loss = mse.mse_loss(input, target);
        float expected = (1*1 + 1*1 + 2*2) / 3.0f; // (1+1+4)/3 = 2.0
        EXPECT_NEAR(loss(0), expected, 1e-4f);
    }

    // Test case 4: Batch processing (multiple samples at once)
    {
        Eigen::MatrixXf input(2, 3);
        Eigen::MatrixXf target(2, 3);
        // Sample 1: Perfect match
        input.row(0) << 1, 1, 1;
        target.row(0) << 1, 1, 1;
        // Sample 2: All errors of 1
        input.row(1) << 2, 2, 2;
        target.row(1) << 3, 3, 3;
        
        Eigen::VectorXf losses = mse.mse_loss(input, target);
        EXPECT_NEAR(losses(0), 0.0f, 1e-4f);  // First sample: MSE = 0
        EXPECT_NEAR(losses(1), 1.0f, 1e-4f);   // Second sample: MSE = (1+1+1)/3 = 1.0
    }
}
