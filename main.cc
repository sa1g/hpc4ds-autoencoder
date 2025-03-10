// #include <stdio.h>
#include <iostream>

#include "dataset.hh"
#include "common.hh"

int main(int argc, char *argv[])
{
    std::string path = "../data/mnist/test";
    std::vector<std::string> filenames = get_filenames(path);

    Dataloader dataloader(path, filenames, 28, 28, filenames.size(), 2000, true);

    std::cout << "Siamo qui" << std::endl;

    // for (auto batch = dataloader.begin(); batch != dataloader.end(); ++batch) 
    for (auto &batch : dataloader)
    {
        const auto &d = batch.dimensions();
        std::cout << "Dim size: " << ", dim 0: " << d[0]
                  << ", dim 1: " << d[1] << ", dim 2: " << d[2] << "  || \t " << batch.sum() << std::endl;

        
    }

    std::cout << "Done!" << std::endl;

    return 0;
}
