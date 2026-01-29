#include <gtest/gtest.h>
#include <Eigen/Dense>

// This benchmark/test is a self-contained version of the
// sgd.hh functionality. It's pure trash.

// Define a simple layer structure for testing
template <size_t input_dim, size_t output_dim>
struct SimpleLayer
{
    Eigen::Matrix<float, output_dim, input_dim> weights;
    Eigen::Matrix<float, output_dim, 1> bias;
    Eigen::Matrix<float, output_dim, input_dim> grad_weights;
    Eigen::Matrix<float, output_dim, 1> grad_bias;

    SimpleLayer()
    {
        weights.setRandom();
        bias.setRandom();
        grad_weights.setRandom();
        grad_bias.setRandom();
    }
};

// SGD implementation for a single layer
template <typename Layer>
void sgd(Layer &layer, float learning_rate)
{
    layer.weights -= learning_rate * layer.grad_weights;
    layer.bias -= learning_rate * layer.grad_bias;
    layer.grad_weights.setZero();
    layer.grad_bias.setZero();
}

// SGD implementation for multiple layers
template <typename... Layers>
void sgd(float learning_rate, Layers &...layers)
{
    (sgd(layers, learning_rate), ...);
}

// Test fixture for SGD
class SGDTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize layers with random weights, biases, and gradients
        layer1.weights.setRandom();
        layer1.bias.setRandom();
        layer1.grad_weights.setRandom();
        layer1.grad_bias.setRandom();

        layer2.weights.setRandom();
        layer2.bias.setRandom();
        layer2.grad_weights.setRandom();
        layer2.grad_bias.setRandom();
    }

    SimpleLayer<3, 2> layer1; // Layer with input_dim = 3, output_dim = 2
    SimpleLayer<2, 4> layer2; // Layer with input_dim = 2, output_dim = 4
};

// Test single layer SGD update
TEST_F(SGDTest, SingleLayerUpdate)
{
    float learning_rate = 0.1f;

    // Save original weights and biases
    auto original_weights = layer1.weights;
    auto original_bias = layer1.bias;

    // Save gradients
    auto grad_weights = layer1.grad_weights;
    auto grad_bias = layer1.grad_bias;

    // Apply SGD
    sgd(layer1, learning_rate);

    // Verify weights and biases are updated correctly
    for (int i = 0; i < layer1.weights.rows(); ++i)
    {
        for (int j = 0; j < layer1.weights.cols(); ++j)
        {
            EXPECT_NEAR(layer1.weights(i, j),
                        original_weights(i, j) - learning_rate * grad_weights(i, j),
                        1e-5);
        }
    }
    for (int i = 0; i < layer1.bias.rows(); ++i)
    {
        EXPECT_NEAR(layer1.bias(i),
                    original_bias(i) - learning_rate * grad_bias(i),
                    1e-5);
    }

    // Verify gradients are reset to zero
    EXPECT_TRUE(layer1.grad_weights.isZero());
    EXPECT_TRUE(layer1.grad_bias.isZero());
}

// Test multiple layers SGD update
TEST_F(SGDTest, MultipleLayersUpdate)
{
    float learning_rate = 0.1f;

    // Save original weights and biases for both layers
    auto original_weights1 = layer1.weights;
    auto original_bias1 = layer1.bias;
    auto original_weights2 = layer2.weights;
    auto original_bias2 = layer2.bias;

    // Save gradients for both layers
    auto grad_weights1 = layer1.grad_weights;
    auto grad_bias1 = layer1.grad_bias;
    auto grad_weights2 = layer2.grad_weights;
    auto grad_bias2 = layer2.grad_bias;

    // Apply SGD to multiple layers
    sgd(learning_rate, layer1, layer2);

    // Verify weights and biases are updated correctly for layer1
    for (int i = 0; i < layer1.weights.rows(); ++i)
    {
        for (int j = 0; j < layer1.weights.cols(); ++j)
        {
            EXPECT_NEAR(layer1.weights(i, j),
                        original_weights1(i, j) - learning_rate * grad_weights1(i, j),
                        1e-5);
        }
    }
    for (int i = 0; i < layer1.bias.rows(); ++i)
    {
        EXPECT_NEAR(layer1.bias(i),
                    original_bias1(i) - learning_rate * grad_bias1(i),
                    1e-5);
    }

    // Verify weights and biases are updated correctly for layer2
    for (int i = 0; i < layer2.weights.rows(); ++i)
    {
        for (int j = 0; j < layer2.weights.cols(); ++j)
        {
            EXPECT_NEAR(layer2.weights(i, j),
                        original_weights2(i, j) - learning_rate * grad_weights2(i, j),
                        1e-5);
        }
    }
    for (int i = 0; i < layer2.bias.rows(); ++i)
    {
        EXPECT_NEAR(layer2.bias(i),
                    original_bias2(i) - learning_rate * grad_bias2(i),
                    1e-5);
    }

    // Verify gradients are reset to zero for both layers
    EXPECT_TRUE(layer1.grad_weights.isZero());
    EXPECT_TRUE(layer1.grad_bias.isZero());
    EXPECT_TRUE(layer2.grad_weights.isZero());
    EXPECT_TRUE(layer2.grad_bias.isZero());
}