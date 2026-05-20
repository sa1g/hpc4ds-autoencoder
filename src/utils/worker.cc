#include "worker.hh"
#include "dataset.hh"
#include "loops.hh"
#include "model.hh"
#include "mse.hh"

#include "tensorboard_logger.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

void auto_worker(const experiment_config &config,
                 std::vector<std::string> &train_filenames,
                 std::vector<std::string> &eval_filenames,
                 std::vector<std::string> &test_filenames,
                 std::string experiment_name, int worker_id, int world_size,
                 std::string timestamp) {
  const bool should_print =
#ifdef USE_MPI
      worker_id == 0;
#else
      true;
#endif
  const bool should_log_metrics = should_print;

  auto average_metric = [&](float local_value) {
#ifdef USE_MPI
    if (world_size > 1) {
      float global_sum = 0.0f;
      MPI_Allreduce(&local_value, &global_sum, 1, MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      return global_sum / static_cast<float>(world_size);
    }
#endif
    return local_value;
  };

  auto max_metric = [&](double local_value) {
#ifdef USE_MPI
    if (world_size > 1) {
      double global_max = 0.0;
      MPI_Allreduce(&local_value, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      return global_max;
    }
#endif
    return local_value;
  };

  auto sum_metric = [&](double local_value) {
#ifdef USE_MPI
    if (world_size > 1) {
      double global_sum = 0.0;
      MPI_Allreduce(&local_value, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      return global_sum;
    }
#endif
    return local_value;
  };

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
  std::unique_ptr<TensorBoardLogger> logger;
  const std::string logger_path = "runs/" + experiment_name + "_" + timestamp;
  if (should_log_metrics) {
    create_directory_if_not_exists("runs/");
    create_directory_if_not_exists(logger_path);
    logger = std::make_unique<TensorBoardLogger>(logger_path + "/tfevents.pb");
  }

  // -- TRAINING
  for (int epoch = 0; epoch < config.epoch; ++epoch) {
    const auto epoch_start = std::chrono::steady_clock::now();

    const float train_loss =
        train("Train: ", config, train_dataloader, model, criterion);

    const float eval_loss = test("Eval:", eval_dataloader, model, criterion);

    const auto epoch_end = std::chrono::steady_clock::now();
    const double local_epoch_time_sec =
        std::chrono::duration<double>(epoch_end - epoch_start).count();
    const double epoch_time_sec = max_metric(local_epoch_time_sec);
    const double total_train_samples =
        sum_metric(static_cast<double>(train_filenames.size()));
    const double samples_per_sec =
        epoch_time_sec > 0.0 ? total_train_samples / epoch_time_sec : 0.0;
    const float global_train_loss = average_metric(train_loss);
    const float global_eval_loss = average_metric(eval_loss);

    // std::cout << "train_loss: " << train_loss << "\n";

    if (should_print) {
      std::cout << "Epoch " << (epoch + 1) << "/" << config.epoch
                << " | train_loss=" << global_train_loss
                << " | eval_loss=" << global_eval_loss
                << " | epoch_time_sec=" << epoch_time_sec
                << " | samples_per_sec=" << samples_per_sec << "\n";
    }

    if (logger) {
      logger->add_scalar("train_loss", epoch, global_train_loss);
      logger->add_scalar("eval_loss", epoch, global_eval_loss);
      logger->add_scalar("epoch_time_sec", epoch, epoch_time_sec);
      logger->add_scalar("samples_per_sec", epoch, samples_per_sec);

      // For per-rank TensorBoard metrics instead of globally averaged metrics:
      // logger->add_scalar("train_loss/rank_" + std::to_string(worker_id),
      // epoch, train_loss); logger->add_scalar("eval_loss/rank_" +
      // std::to_string(worker_id), epoch, eval_loss);

      // If histogram summaries are available in the logger API, weights can be
      // logged here: logger->add_histogram("encoder/weights", epoch,
      // model.encoder.weights); logger->add_histogram("decoder/weights", epoch,
      // model.decoder.weights);

      // Gradient monitoring examples:
      // logger->add_scalar("encoder_grad_norm", epoch,
      // model.encoder.grad_weights.norm());
      // logger->add_scalar("decoder_grad_norm", epoch,
      // model.decoder.grad_weights.norm());
    }
  }

  // -- TESTING (always done locally)
  const float test_loss = test("Test: ", test_dataloader, model, criterion);
  const float global_test_loss = average_metric(test_loss);
  if (logger) {
    logger->add_scalar("test_loss", config.epoch, global_test_loss);
  }
}
