// #include <stdio.h>
#include <cstdio>
#include <iostream>

#include "common.hh"
#include "dataset.hh"
#include "model.hh"
#include "mse.hh"
// #include "sgd.hh"
#include "loops.hh"
// #include "linear.hh"
#include "worker.hh"

int main(int argc, char *argv[]) {
  ////////////////////////////////////////////////////////
  const experiment_config config = {.train_path = "../data/mnist/train",
                                    .test_path = "../data/mnist/test",
                                    .batch_size = 256,
                                    .input_dim = 28 * 28,
                                    .hidden_dim = 256,
                                    .output_dim = 28 * 28,
                                    .lr = 0.01f,
                                    .epoch = 100};

  // const int max_batch_size{254};
  // const int input_dim{28 * 28};
  // const int hidden_dim{256};
  // const int output_dim{input_dim};
  // const float lr{0.01};

  ///////////////////////////////////////////////////////

  // std::string train_path = "../data/mnist/train";
  std::vector<std::string> filenames = get_filenames(config.train_path);
  auto [train_filenames, eval_filenames] =
      random_split_filenames(filenames, 20, 42);

  // std::string test_path = "../data/mnist/test";
  std::vector<std::string> test_filenames = get_filenames(config.test_path);

  // Dataloader train_dataloader(config.train_path, train_filenames, 28, 28,
  //                             train_filenames.size(), config.batch_size,
  //                             true);
  // Dataloader eval_dataloader(config.train_path, eval_filenames, 28, 28,
  //                            eval_filenames.size(), config.batch_size,
  //                            false);
  // Dataloader test_dataloader(config.test_path, test_filenames, 28, 28,
  //                            test_filenames.size(), config.batch_size,
  //                            false);

  // std::cout << "Got dataloaders" << std::endl;

  // //////////////////////////////////////////////////////////

  // AutoencoderModel model{config.batch_size, config.input_dim,
  // config.hidden_dim,
  //                        config.output_dim};
  // MSE criterion{config.batch_size, config.input_dim};

  // /////////////////////////////////////////////////////////

  auto_worker(config, train_filenames, eval_filenames, test_filenames, "prove", 0);
  // auto_worker(config);
  // for (int epoch = 0; epoch < config.epoch; ++epoch) {
  //   float train_loss = train(config, train_dataloader, model, criterion);

  //   float test_loss = test("Eval:", eval_dataloader, model, criterion);

  //   std::cout << "train_loss: " << train_loss << "\n";

  //   break;
  // }

  std::cout << "Done!" << std::endl;

  return 0;
}
