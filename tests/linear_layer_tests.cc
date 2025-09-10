#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "Eigen/src/Core/Matrix.h"
#include "linear.hh"

TEST(LinearLayerTest, OutputShape)
{
    constexpr size_t max_batch_size = 10;
    constexpr size_t input_dim = 10;
    constexpr size_t output_dim = 5;

    Linear layer{max_batch_size, input_dim, output_dim};
    int batch_size = 2; // Actual batch size used in the test

    // Eigen::Matrix<float, Eigen::Dynamic, input_dim> input(batch_size, input_dim);
    Eigen::MatrixXf input(batch_size, input_dim);
    input.setRandom();

    Eigen::MatrixXf output = layer.forward(input);

    EXPECT_EQ(output.rows(), batch_size);
    EXPECT_EQ(output.cols(), output_dim);
}

TEST(LinearLayerTest, BiasApplication)
{
    constexpr size_t max_batch_size = 10;
    constexpr size_t input_dim = 3;
    constexpr size_t output_dim = 2;

    Linear layer{max_batch_size, input_dim, output_dim};
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
    constexpr size_t max_batch_size = 10;
    constexpr size_t input_dim = 3;
    constexpr size_t output_dim = 2;

    Linear layer{max_batch_size, input_dim, output_dim};
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
    constexpr size_t max_batch_size = 10;
    constexpr size_t input_dim = 4;
    constexpr size_t output_dim = 2;

    Linear layer{max_batch_size, input_dim, output_dim};

    int batch_size{3};
    Eigen::MatrixXf input = Eigen::MatrixXf::Zero(batch_size, input_dim);

    Eigen::MatrixXf output = layer.forward(input);

    for (int i = 0; i < batch_size; ++i)
    {
        EXPECT_TRUE((output.row(i).transpose().array() == layer.bias.array()).all());
    }
}

TEST(LinearLayerTest, LargeValues)
{
    constexpr size_t max_batch_size = 10;
    constexpr size_t input_dim = 3;
    constexpr size_t output_dim = 2;

    Linear layer{max_batch_size, input_dim, output_dim};

    int batch_size{2};

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

constexpr size_t max_batch_size = 4;
constexpr size_t input_dim = 3;
constexpr size_t output_dim = 2;

class LinearTest : public ::testing::Test
{
protected:
    Linear linear_layer{max_batch_size, input_dim, output_dim};

    void SetUp() override
    {
        // Set deterministic weights and biases for reproducibility
        linear_layer.weights << 0.1, 0.2, 0.3,
            0.4, 0.5, 0.6;
        linear_layer.bias << 0.1, 0.2;
    }
};

TEST_F(LinearTest, BackwardComputesCorrectGradients)
{
    Eigen::MatrixXf input(2, input_dim);
    input << 1.0, 2.0, 3.0, 
        4.0, 5.0, 6.0;

    Eigen::MatrixXf grad_output(2, output_dim);
    grad_output << 0.5, -0.5,
        1.0, -1.0;

    Eigen::MatrixXf grad_input = linear_layer.backward(input, grad_output);

    // Expected gradients
    Eigen::MatrixXf expected_grad_weights(output_dim, input_dim);
    expected_grad_weights << (0.5 * 1.0 + 1.0 * 4.0), (0.5 * 2.0 + 1.0 * 5.0), (0.5 * 3.0 + 1.0 * 6.0),
        (-0.5 * 1.0 + -1.0 * 4.0), (-0.5 * 2.0 + -1.0 * 5.0), (-0.5 * 3.0 + -1.0 * 6.0);

    Eigen::VectorXf expected_grad_bias(output_dim);
    expected_grad_bias << (0.5 + 1.0), (-0.5 + -1.0);

    Eigen::MatrixXf expected_grad_input(2, input_dim);
    expected_grad_input = grad_output * linear_layer.weights;

    // Validate computed gradients
    EXPECT_TRUE(linear_layer.grad_weights.isApprox(expected_grad_weights, 1e-5))
        << "Gradient weights do not match expected values.";
    EXPECT_TRUE(linear_layer.grad_bias.isApprox(expected_grad_bias, 1e-5))
        << "Gradient bias does not match expected values.";
    EXPECT_TRUE(grad_input.isApprox(expected_grad_input, 1e-5))
        << "Gradient input does not match expected values.";
}