#include "worker.hh"
#include "dataset.hh"
#include "loops.hh"
#include "model.hh"
#include "mse.hh"

#include "tensorboard_logger.h"

#include <iostream>

void auto_worker(const experiment_config &config,
                 std::vector<std::string> &train_filenames,
                 std::vector<std::string> &eval_filenames,
                 std::vector<std::string> &test_filenames,
                 std::string experiment_name, int worker_id) {
  // -- DATALOADERS SETUP
  Dataloader train_dataloader(config.train_path, train_filenames, 28, 28,
                              train_filenames.size(), config.batch_size, true);
  Dataloader eval_dataloader(config.train_path, eval_filenames, 28, 28,
                             eval_filenames.size(), config.batch_size, false);
  Dataloader test_dataloader(config.test_path, test_filenames, 28, 28,
                             test_filenames.size(), config.batch_size, false);

  // -- MODEL, CRITERION SETUP

  AutoencoderModel model{config.batch_size, config.input_dim, config.hidden_dim,
                         config.output_dim};
  MSE criterion{config.batch_size, config.input_dim};

//   // -- TensorBoardLogger
  std::string logger_path = "../experiments/" + experiment_name + "/tfevents.pb";
  TensorBoardLogger logger(logger_path);

  // -- TRAINING
  for (int epoch = 0; epoch < config.epoch; ++epoch) {
    float train_loss =
        train("Train: ", config, train_dataloader, model, criterion);

    float eval_loss = test("Eval:", eval_dataloader, model, criterion);

    std::cout << "train_loss: " << train_loss << "\n";

    logger.add_scalar("train_loss", epoch, train_loss);
    logger.add_scalar("eval_loss", epoch, eval_loss);
    break;
  }

  float test_loss = test("Test: ", test_dataloader, model, criterion);
  logger.add_scalar("test_loss", config.epoch, test_loss);
}