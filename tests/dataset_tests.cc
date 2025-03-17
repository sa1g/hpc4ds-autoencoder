#include <gtest/gtest.h>
#include <Eigen/Dense>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "dataset.hh"

// Mock image data for testing
std::vector<unsigned char> create_mock_image(int width, int height)
{
    std::vector<unsigned char> image(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        image[i] = static_cast<unsigned char>(i % 256); // Fill wit hdummy pixel values
    }

    return image;
}

// Test fixture for Dataloader
class DataloaderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create mock filenames
        filenames = {"image1", "image2", "image3", "image4", "image5"};
        width = 2;
        height = 2;
        num_images = filenames.size();
        batch_size = 2;
        shuffle = false;

        // Create mock images and save them to disk
        for (const auto &filename : filenames)
        {
            std::string filepath = path + "/" + filename + ".png";
            std::vector<unsigned char> image = create_mock_image(width, height);
            stbi_write_png(filepath.c_str(), width, height, 1, image.data(), 0);
        }
    }

    void TearDown() override
    {
        // Clean up mock images
        for (const auto &filename : filenames)
        {
            std::string filepath = path + "/" + filename + ".png";
            std::remove(filepath.c_str());
        }
    }

    // Note that I don't want to manage the creation of test
    // directories/similar, so we just put everything in /tmp.
    std::string path = "/tmp";
    std::vector<std::string> filenames;
    int width, height, num_images, batch_size;
    bool shuffle;
};

// test constructor
TEST_F(DataloaderTest, ConstructorTest)
{
    Dataloader dataloader(path, filenames, width, height, num_images, batch_size, shuffle);

    // Check if the #batches is calculated correctly
    EXPECT_EQ(dataloader.get_num_batches(), (num_images + batch_size - 1) / batch_size);

    // Check if shuffling is disabled
    EXPECT_FALSE(dataloader.is_shuffled());

    // Check if the #images is valid
    EXPECT_EQ(dataloader.get_num_images(), num_images);
}

// test get_batch
TEST_F(DataloaderTest, GetBatchTest)
{
    Dataloader dataloader(path, filenames, width, height, num_images, batch_size, shuffle);

    // Load the first batch
    Eigen::MatrixXf batch = dataloader.get_batch();

    // Check the shape of the batch matrix
    EXPECT_EQ(batch.rows(), batch_size);
    EXPECT_EQ(batch.cols(), width * height);

    // Check the values of the first image in the batch
    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_FLOAT_EQ(batch(0, i), static_cast<float>(i % 256) / 255.0f);
    }

    // Check the values of the 2nd image in the batch
    for (int i = 0; i < width * height; ++i)
    {
        EXPECT_FLOAT_EQ(batch(1, i), static_cast<float>(i % 256) / 255.0f);
    }
}

// test iterator
TEST_F(DataloaderTest, IteratorTest)
{
    Dataloader dataloader(path, filenames, width, height, num_images, batch_size, shuffle);

    std::cerr << dataloader.get_num_batches() << std::endl;

    int batch_count = 0;
    for (auto it = dataloader.begin(); it != dataloader.end(); ++it)
    {
        Eigen::MatrixXf &batch = *it;
        EXPECT_EQ(batch.rows(), batch_size);
        EXPECT_EQ(batch.cols(), width * height);
        batch_count++;
    }

    EXPECT_EQ(batch_count, dataloader.get_num_batches());
}

// test shuffling
TEST_F(DataloaderTest, ShuffleTest){
    bool shuffle = true;
    Dataloader dataloader(path, filenames, width, height, num_images, batch_size, shuffle);

    // Check if filenames are shuffled
    bool is_shuffled = false;
    for(size_t i =0; i<filenames.size(); ++i){
        if(dataloader.get_filenames()[i] != filenames[i]){
            is_shuffled=true;
            break;
        }
    }
    EXPECT_TRUE(is_shuffled);
}

// TODO: Add tests for edge cases as this is not a complete test suite - rn we are covering just the basics