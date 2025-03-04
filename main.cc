// #include <stdio.h>
#include <iostream>

#include "dataset.hh"

int main(int argc, char *argv[])
{
    std::string filename = "../data/mnist/test/0/3.png";
    int width{28}, height{28};
    bool is_rgb{false};

    Eigen::MatrixXd matrix = loadImageToMatrix(filename, is_rgb, width, height);

    std::cout << "Image loaded" << width << "x" << height << " " << (is_rgb ? "RGB" : "grayscale") << std::endl;

    // Print the first 10 elements of the matrix
    std::cout << matrix.block(0,0,10,10) << std::endl;

    return 0;
}
