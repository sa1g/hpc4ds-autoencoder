#include "common.hh"

#include <gtest/gtest.h>

TEST(RandomSplitTest, SplitFileNames)
{
    std::vector<std::string> filenames = {"0/image1", "0/image2", "1/image1", "1/image2", "2/image1", "2/image2"};

    auto [train_filenames, test_filenames] = random_split_filenames(filenames, 20, 42);

    // Check that the total number of filenames is preserved
    EXPECT_EQ(train_filenames.size() + test_filenames.size(), filenames.size());

    // Check that the test set has approximately 20% of the original filenames
    EXPECT_GE(test_filenames.size(), filenames.size() * 0.2);

    // Check that the train set has the remaining filenames
    EXPECT_EQ(train_filenames.size(), filenames.size() - test_filenames.size());
}
