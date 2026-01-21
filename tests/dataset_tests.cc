#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <filesystem>

#include "dataset.hh"

#include "stb_image.h"
#include "stb_image_write.h"

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
    Dataloader dataloader(path, filenames, width, height, 1, num_images, batch_size, shuffle);

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
    Dataloader dataloader(path, filenames, width, height, 1, num_images, batch_size, shuffle);

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
    Dataloader dataloader(path, filenames, width, height, 1, num_images, batch_size, shuffle);

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
    Dataloader dataloader(path, filenames, width, height, 1, num_images, batch_size, shuffle);

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

TEST_F(DataloaderTest, SaveImageTest)
{   
    Dataloader dataloader(path, filenames, width, height, 1, num_images, batch_size, shuffle);
    
    // Load the first batch
    Eigen::MatrixXf batch = dataloader.get_batch();
    
    // Check the shape of the batch matrix
    EXPECT_EQ(batch.rows(), batch_size);
    EXPECT_EQ(batch.cols(), width * height);
    
    // Save the image
    std::string test_image_path = "test_img.png";
    dataloader.save_batch_image(batch, 0, test_image_path);
    
    // --- VERIFICATION SECTION ---
    
    // 1. Check if file was created
    EXPECT_TRUE(std::filesystem::exists(test_image_path)) 
        << "Image file was not created: " << test_image_path;
    
    // 2. Load the saved image back using stb_image
    int img_width, img_height, img_channels;
    unsigned char* img_data = stbi_load(test_image_path.c_str(), 
                                        &img_width, &img_height, 
                                        &img_channels, 0);
    
    // Verify image loaded successfully
    ASSERT_NE(img_data, nullptr) 
        << "Failed to load saved image. stbi_failure_reason: " 
        << (stbi_failure_reason() ? stbi_failure_reason() : "Unknown error");
    
    // 3. Check dimensions match
    EXPECT_EQ(img_width, width) << "Saved image width doesn't match";
    EXPECT_EQ(img_height, height) << "Saved image height doesn't match";
    EXPECT_EQ(img_channels, 1) << "Expected grayscale image (1 channel)";
    
    // 4. Verify pixel values (normalized comparison)
    // Convert Eigen batch row to float array for comparison
    Eigen::VectorXf original_pixels = batch.row(0);  // First image in batch
    
    float max_diff = 0.0f;
    float tolerance = 1.0f / 255.0f;  // Allow for 1 pixel value difference due to quantization
    
    for (int i = 0; i < width * height; i++) {
        // Convert saved pixel from [0, 255] to [0, 1] for comparison
        float saved_pixel = img_data[i] / 255.0f;
        float original_pixel = original_pixels[i];
        
        float diff = std::abs(saved_pixel - original_pixel);
        max_diff = std::max(max_diff, diff);
        
        // Optional: Check each pixel individually (more verbose)
        // EXPECT_NEAR(saved_pixel, original_pixel, tolerance) 
        //     << "Pixel mismatch at position " << i;
    }
    
    // Check overall similarity
    EXPECT_LT(max_diff, tolerance) 
        << "Saved image differs from original. Max diff: " << max_diff;
    
    // 5. Optional: Compare histogram statistics
    float original_mean = original_pixels.mean();
    float saved_mean = 0.0f;
    for (int i = 0; i < width * height; i++) {
        saved_mean += img_data[i] / 255.0f;
    }
    saved_mean /= (width * height);
    
    EXPECT_NEAR(original_mean, saved_mean, 0.01f) 
        << "Mean pixel values differ significantly";
    
    // 6. Cleanup
    stbi_image_free(img_data);
    
    // Optional: Clean up test file
    std::filesystem::remove(test_image_path);
    
    // 7. Additional verification: Test multiple images in batch
    for (int i = 0; i < std::min(3, batch_size); i++) {
        std::string multi_img_path = "test_img_" + std::to_string(i) + ".png";
        dataloader.save_batch_image(batch, i, multi_img_path);
        
        // Quick check - just verify file exists and has correct size
        EXPECT_TRUE(std::filesystem::exists(multi_img_path));
        EXPECT_GT(std::filesystem::file_size(multi_img_path), 0);
        
        // Clean up
        std::filesystem::remove(multi_img_path);
    }
}

// TEST_F(DataloaderTest, SaveImageTest)
// {   
//     Dataloader dataloader(path, filenames, width, height, 1, num_images, batch_size, shuffle);

//     // Load the first batch
//     // const 
//     Eigen::MatrixXf batch = dataloader.get_batch();

//     // Check the shape of the batch matrix
//     EXPECT_EQ(batch.rows(), batch_size);
//     EXPECT_EQ(batch.cols(), width * height);

//     dataloader.save_batch_image(batch, 0, "test_img.png");



// }

// TODO: Add tests for edge cases as this is not a complete test suite - rn we are covering just the basics