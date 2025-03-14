#include <gtest/gtest.h>
#include "common.hh"

TEST(ExampleTests, DemonstrateGTestMacros)
{
    EXPECT_TRUE(true);
    std::string path = "../data/mnist/test";
    std::vector<std::string> filenames = get_filenames(path);
}