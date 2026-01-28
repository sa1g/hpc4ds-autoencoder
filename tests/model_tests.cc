#include <gtest/gtest.h>

#include "model.hh"

TEST(ModelTest, GetWeights)
{
    AutoencoderModel model(10, 784, 64, 784); // batch_size, input_dim, hidden_dim, output_dim
    auto weights = model.get_weights();

    // Check that all expected keys exist
    EXPECT_TRUE(weights.find("encoder_w") != weights.end());
    EXPECT_TRUE(weights.find("encoder_b") != weights.end());
    EXPECT_TRUE(weights.find("decoder_w") != weights.end());
    EXPECT_TRUE(weights.find("decoder_b") != weights.end());

    // Check dimensions
    EXPECT_EQ(weights["encoder_w"].rows(), 64);
    EXPECT_EQ(weights["encoder_w"].cols(), 784);
    EXPECT_EQ(weights["decoder_w"].rows(), 784);
    EXPECT_EQ(weights["decoder_w"].cols(), 64);
}

TEST(ModelTest, SetWeights)
{
    AutoencoderModel model(10, 784, 64, 784);
    auto original_weights = model.get_weights();

    // Modify weights
    original_weights["encoder_w"].setRandom();
    original_weights["encoder_b"].setRandom();
    original_weights["decoder_w"].setRandom();
    original_weights["decoder_b"].setRandom();

    // Set and verify
    model.set_weights(original_weights);
    auto new_weights = model.get_weights();
    EXPECT_TRUE(original_weights["encoder_w"].isApprox(new_weights["encoder_w"]));
    EXPECT_TRUE(original_weights["encoder_b"].isApprox(new_weights["encoder_b"]));
    EXPECT_TRUE(original_weights["decoder_w"].isApprox(new_weights["decoder_w"]));
    EXPECT_TRUE(original_weights["decoder_b"].isApprox(new_weights["decoder_b"]));
}

TEST(ModelTest, SaveLoadWeights)
{
    AutoencoderModel model(10, 784, 64, 784);
    auto original_weights = model.get_weights();
    original_weights["encoder_w"].setRandom();
    original_weights["encoder_b"].setRandom();
    original_weights["decoder_w"].setRandom();
    original_weights["decoder_b"].setRandom();
    model.set_weights(original_weights);

    // Save and load
    std::string path = "test_weights.bin";
    model.save_weights(path);
    auto loaded_weights = model.load_weights(path);

    // Verify
    EXPECT_TRUE(original_weights["encoder_w"].isApprox(loaded_weights["encoder_w"]));
    EXPECT_TRUE(original_weights["encoder_b"].isApprox(loaded_weights["encoder_b"]));
    EXPECT_TRUE(original_weights["decoder_w"].isApprox(loaded_weights["decoder_w"]));
    EXPECT_TRUE(original_weights["decoder_b"].isApprox(loaded_weights["decoder_b"]));

    // Clean up
    std::remove(path.c_str());
}

TEST(ModelTest, LoadNonexistentFile)
{
    AutoencoderModel model(10, 784, 64, 784);
    EXPECT_THROW(model.load_weights("nonexistent_file.bin"), std::runtime_error);
}

TEST(ModelTest, WeightManagementIntegration)
{
    AutoencoderModel model1(10, 784, 64, 784);
    AutoencoderModel model2(10, 784, 64, 784);

    // Set random weights in model1
    auto weights = model1.get_weights();
    weights["encoder_w"].setRandom();
    weights["encoder_b"].setRandom();
    weights["decoder_w"].setRandom();
    weights["decoder_b"].setRandom();
    model1.set_weights(weights);

    // Save from model1, load into model2
    std::string path = "integration_test_weights.bin";
    model1.save_weights(path);
    auto loaded_weights = model2.load_weights(path);
    model2.set_weights(loaded_weights);

    // Verify model2 has the same weights as model1
    auto model1_weights = model1.get_weights();
    auto model2_weights = model2.get_weights();
    EXPECT_TRUE(model1_weights["encoder_w"].isApprox(model2_weights["encoder_w"]));
    EXPECT_TRUE(model1_weights["encoder_b"].isApprox(model2_weights["encoder_b"]));
    EXPECT_TRUE(model1_weights["decoder_w"].isApprox(model2_weights["decoder_w"]));
    EXPECT_TRUE(model1_weights["decoder_b"].isApprox(model2_weights["decoder_b"]));

    // Clean up
    std::remove(path.c_str());
}
