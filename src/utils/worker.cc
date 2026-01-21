#include "worker.hh"
#include "dataset.hh"
#include "loops.hh"
#include "model.hh"
#include "mse.hh"

#include "tensorboard_logger.h"

#include <iostream>
#include <string>

// TODO: remove this one
#ifdef USE_MPI
#include <mpi.h>
#endif

void auto_worker(const experiment_config &config,
                 std::vector<std::string> &train_filenames,
                 std::vector<std::string> &eval_filenames,
                 std::vector<std::string> &test_filenames,
                 std::string experiment_name, int worker_id, int world_size,
                 std::string timestamp) {
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

  // -- TensorBoardLogger
  create_directory_if_not_exists("../runs/");
  
  std::string logger_path = "../runs/" + experiment_name + "_" + std::to_string(worker_id) + "_" + timestamp;//"/tfevents.pb";
  create_directory_if_not_exists(logger_path);
  
  TensorBoardLogger logger(logger_path + "/tfevents.pb");

  // -- TRAINING
  for (int epoch = 0; epoch < config.epoch; ++epoch) {
    float train_loss =
        train("Train: ", config, train_dataloader, model, criterion);

    float eval_loss = test("Eval:", eval_dataloader, model, criterion);

    std::cout << "train_loss: " << train_loss << "\n";

    logger.add_scalar("train_loss", epoch, train_loss);
    logger.add_scalar("eval_loss", epoch, eval_loss);
    // break;
  }

  // Averaging weights before testing

  // 1. get local weights
  auto local_weights = model.get_weights();
  auto averaged_weights = local_weights; // create same structure

  for (auto &kv : local_weights) {
      auto &name = kv.first;
      auto &mat = kv.second;

      // flatten matrix
      int count = mat.rows() * mat.cols();
      std::vector<float> sendbuf(mat.data(), mat.data() + count);
      std::vector<float> recvbuf(count);

      // 2. sum all workers' weights
      MPI_Allreduce(sendbuf.data(), recvbuf.data(), count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

      // 3. convert to Eigen and average
      Eigen::Map<Eigen::MatrixXf> averaged_mat(recvbuf.data(), mat.rows(), mat.cols());
      averaged_mat /= static_cast<float>(world_size);

      averaged_weights[name] = averaged_mat;
  }

  // 4. set averaged weights into model
  model.set_weights(averaged_weights);

  if (worker_id == 0) {
      std::cout << "[MPI] Averaged weights across " << world_size << " workers.\n";
  }

  // TESTING
  float test_loss = test("Test: ", test_dataloader, model, criterion);
  logger.add_scalar("test_loss", config.epoch, test_loss);
}