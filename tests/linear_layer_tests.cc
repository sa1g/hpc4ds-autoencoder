#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "linear.hh"

TEST(LinearLayerTest, OutputShape)
{
    int batch_size{2};
    int input_dim{10};
    int output_dim{5};

    Linear layer(input_dim, output_dim);
    Eigen::MatrixXf input(batch_size, input_dim);

    input.setRandom();

    Eigen::MatrixXf output = layer.forward(input);

    EXPECT_EQ(output.rows(), batch_size);
    EXPECT_EQ(output.cols(), output_dim);
}

TEST(LinearLayerTest, BiasApplication)
{
    int batch_size{1};
    int input_dim{3};
    int output_dim{2};

    Linear layer(input_dim, output_dim);
    layer.weights << 1, 2, 3, 4, 5, 6;
    layer.bias << 1, 2;

    Eigen::MatrixXf input(1, 3);
    input << 1, 1, 1;

    Eigen::MatrixXf output = layer.forward(input);

    EXPECT_FLOAT_EQ(output(0, 0), 6 + 1);
    EXPECT_FLOAT_EQ(output(0, 1), 15 + 2);
}

TEST(LinearLayerTest, NoBias)
{
    int batch_size{1};
    int input_dim{3};
    int output_dim{2};

    Linear layer(input_dim, output_dim);
    layer.bias.setZero();

    Eigen::MatrixXf input(1, 3);
    input.setOnes();

    Eigen::MatrixXf output = layer.forward(input);

    Eigen::MatrixXf expected_output = layer.weights * input.transpose();

    for (int i{0}; i < output.size(); ++i)
    {
        EXPECT_FLOAT_EQ(output(i), expected_output(i));
    }
}

TEST(LinearLayerTest, ZeroInput)
{
    int batch_size{3};
    int input_dim{4};
    int output_dim{2};

    Linear layer(input_dim, output_dim);
    Eigen::MatrixXf input = Eigen::MatrixXf::Zero(batch_size, input_dim);

    Eigen::MatrixXf output = layer.forward(input);

    for (int i = 0; i < batch_size; ++i)
    {
        EXPECT_TRUE((output.row(i).transpose().array() == layer.bias.array()).all());
    }
}

TEST(LinearLayerTest, LargeValues)
{
    int batch_size{2};
    int input_dim{3};
    int output_dim{2};

    Linear layer(input_dim, output_dim);

    Eigen::MatrixXf input(batch_size, input_dim);
    input.setConstant(1e6);

    Eigen::MatrixXf output = layer.forward(input);

    // Ensure that the output is not NaN or Inf
    for (int i{0}; i < output.rows(); ++i)
    {
        for (int j = 0; j < output.cols(); ++j)
        {
            EXPECT_TRUE(std::isfinite(output(i, j)));
        }
    }
}

TEST(LinearLayerTest, BackwardPassWeights)
{
    int input_dim = 3;
    int output_dim = 2;
    Linear layer(input_dim, output_dim);

    Eigen::MatrixXf input(3, 3);
    input << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;

    Eigen::MatrixXf grad_output(2, 3); // Gradient of loss w.r.t. output
    grad_output << 1, 0, 0,
        0, 1, 1;

    Eigen::MatrixXf grad_input = layer.backward(input, grad_output);

    // Calculate expected gradients for weights
    Eigen::MatrixXf expected_grad_weights = grad_output * input;

    // Check that the gradient matches
    EXPECT_TRUE(layer.grad_weights.isApprox(expected_grad_weights));
}

TEST(LinearLayerTest, BackwardPassInput)
{
    int input_dim{3};
    int output_dim{2};
    Linear layer(input_dim, output_dim);

    Eigen::MatrixXf input(3, 3);
    input << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    Eigen::MatrixXf grad_output(2, 3);
    grad_output << 1, 0, 0, 0, 1, 1;

    Eigen::MatrixXf grad_input = layer.backward(input, grad_output);

    Eigen::MatrixXf expected_grad_input = layer.weights.transpose() * grad_output;

    EXPECT_TRUE(grad_input.isApprox(expected_grad_input));
}