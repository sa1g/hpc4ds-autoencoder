// #include <stdio.h>
#include <cstdio>
#include <iostream>

#include "common.hh"
#include "dataset.hh"
#include "model.hh"
#include "mse.hh"
#include "sgd.hh"
// #include "linear.hh"

int main(int argc, char *argv[]) {
  ////////////////////////////////////////////////////////
  experiment_config config = {
    .train_path = "../data/mnist/train", 
    .test_path = "../data/mnist/test",
    .batch_size = 254,
    .input_dim=28*28,
    .hidden_dim=256,
    .output_dim=28*28,
    .lr=0.01f,
    .epoch=100 
  };


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

  std::string test_path = "../data/mnist/test";
  std::vector<std::string> test_filenames = get_filenames(test_path);

  Dataloader train_dataloader(config.train_path, train_filenames, 28, 28,
                              train_filenames.size(), config.batch_size, true);
  Dataloader eval_dataloader(config.train_path, eval_filenames, 28, 28,
                             eval_filenames.size(), config.batch_size, false);
  Dataloader test_dataloader(test_path, test_filenames, 28, 28,
                             test_filenames.size(), config.batch_size, false);

  std::cout << "Got dataloaders" << std::endl;

  //////////////////////////////////////////////////////////

  AutoencoderModel model{config.batch_size, config.input_dim, config.hidden_dim, config.output_dim};
  // std::cout << "Created model" << std::endl;

  // return 1;

  // // for (auto batch = dataloader.begin(); batch != dataloader.end();
  // ++batch) int counter = 0;

  MSE criterion{config.batch_size, config.input_dim};

  /////////////////////////////////////////////////////////


  for (int epoch = 0; epoch < 100; ++epoch) {
    // TODO: collect metrics, etc

    float epoch_loss{0};
    int num_batches{0};

    for (auto &batch : train_dataloader) {
      auto prediction = model.forward(batch);
      auto loss = criterion.mse_loss(batch, prediction);
      auto grad = criterion.mse_gradient(batch, prediction);
      model.backward(batch, grad);
      sgd(config.lr, model.encoder, model.decoder);

      num_batches++;
      epoch_loss += loss;

      printf("Batch %i/%i | Loss: %.3f\n", num_batches,
             train_dataloader.get_num_batches(), loss);
    }

    // float average_epoch_loss = epoch_loss / num_batches;
    
    break;
  }
  std::cout << "Done!" << std::endl;

  return 0;
}
