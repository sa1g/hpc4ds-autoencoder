// #include <stdio.h>
#include <iostream>

#include "dataset.hh"
#include "common.hh"
#include "model.hh"

int main(int argc, char *argv[])
{
    std::string train_path = "../data/mnist/train";
    std::vector<std::string> filenames = get_filenames(train_path);
    auto [train_filenames, eval_filenames] = random_split_filenames(filenames, 20, 42);

    std::string test_path = "../data/mnist/test";
    std::vector<std::string> test_filenames = get_filenames(test_path);

    const int batch_size = 64;

    Dataloader train_dataloader(train_path, train_filenames, 28, 28, train_filenames.size(), batch_size, true);
    Dataloader eval_dataloader(train_path, eval_filenames, 28, 28, eval_filenames.size(), batch_size, false);
    Dataloader test_dataloader(test_path, test_filenames, 28, 28, test_filenames.size(), batch_size, false);

    std::cout << "Got dataloaders" << std::endl;

    Linear<10, 28*28, 2> encoder;


    // AutoencoderModel<2000, 28 * 28, 256, 28 * 28> model;
    // std::cout << "Created model" << std::endl;

    // return 1;

    // // for (auto batch = dataloader.begin(); batch != dataloader.end(); ++batch)
    // int counter = 0;
    // for (auto &batch : train_dataloader)
    // {
    //     // const auto &d = batch.N();
    //     // std::cout << "Dim size: " << ", dim 0: " << d[0]
    //     //           << ", dim 1: " << d[1] << ", dim 2: " << d[2] << "  || \t " << batch.sum() << std::endl;
    //     // std::cout << counter << std::endl;
    //     // ++counter;

    //     std::cout << "Batch shape: " << batch.rows() << "x" << batch.cols() << std::endl;
    // }

    std::cout << "Done!" << std::endl;

    return 0;
}
