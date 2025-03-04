#include "dataset.hh"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif // STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

Eigen::MatrixXd loadImageToMatrix(const std::string &filename, bool &is_rgb, int &width, int &height)
{

    // Load the image using stb_image
    int n_channels;
    unsigned char *data = stbi_load(filename.c_str(), &width, &height, &n_channels, 0);
    if (!data)
    {
        std::cerr << "Error loading image: " << filename << std::endl;
        exit(1);
    }

    // Set is_rbg to true if the image has 3 channels (RGB), false if 1 channel (grayscale)
    is_rgb = (n_channels == 3);

    // Create the Eigen matrix
    if (is_rgb)
    {
        // If the image is RGB, create a matrix with 3 rows for each pixel (R, G, B)
        Eigen::MatrixXd rgb_matrix(height, width * 3);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; y < width; ++x)
            {
                int idx = (y * width + x) * 3;
                rgb_matrix(y, x * 3) = data[idx] / 255.0;
                rgb_matrix(y, x * 3 + 1) = data[idx + 1] / 255.0;
                rgb_matrix(y, x * 3 + 2) = data[idx + 2] / 255.0;
            }
        }

        // Cleanup and return RGB matrix
        stbi_image_free(data);
        return rgb_matrix;
    }
    else
    {
        Eigen::MatrixXd matrix(height, width);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = (y * width + x);
                matrix(y, x) = data[idx] / 255.0;
            }
        }

        // Cleanup and return matrix
        stbi_image_free(data);
        return matrix;
    }
}