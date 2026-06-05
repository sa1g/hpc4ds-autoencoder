#include "worker.hh"
#include "dataset.hh"
#include "loops.hh"
#include "model.hh"
#include "mse.hh"

#include "tensorboard_logger.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

void update_federated_weights_single_call(AutoencoderModel &model,
                                          int worker_id, int world_size,
                                          bool should_print) {
  (void)worker_id;
  (void)should_print;

  if (world_size <= 1)
    return;

  const size_t total_params =
      model.encoder.weights.size() + model.encoder.bias.size() +
      model.decoder.weights.size() + model.decoder.bias.size();

  thread_local std::vector<float> flat_weights;
  flat_weights.resize(total_params);

  size_t offset = 0;
  std::memcpy(flat_weights.data() + offset, model.encoder.weights.data(),
              model.encoder.weights.size() * sizeof(float));
  offset += model.encoder.weights.size();
  std::memcpy(flat_weights.data() + offset, model.encoder.bias.data(),
              model.encoder.bias.size() * sizeof(float));
  offset += model.encoder.bias.size();
  std::memcpy(flat_weights.data() + offset, model.decoder.weights.data(),
              model.decoder.weights.size() * sizeof(float));
  offset += model.decoder.weights.size();
  std::memcpy(flat_weights.data() + offset, model.decoder.bias.data(),
              model.decoder.bias.size() * sizeof(float));

  if (total_params > 0) {
    MPI_Allreduce(MPI_IN_PLACE, flat_weights.data(),
                  static_cast<int>(flat_weights.size()), MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);
  }

  float inv_world = 1.0f / world_size;
  for (float &value : flat_weights) {
    value *= inv_world;
  }

  offset = 0;
  std::memcpy(model.encoder.weights.data(), flat_weights.data() + offset,
              model.encoder.weights.size() * sizeof(float));
  offset += model.encoder.weights.size();
  std::memcpy(model.encoder.bias.data(), flat_weights.data() + offset,
              model.encoder.bias.size() * sizeof(float));
  offset += model.encoder.bias.size();
  std::memcpy(model.decoder.weights.data(), flat_weights.data() + offset,
              model.decoder.weights.size() * sizeof(float));
  offset += model.decoder.weights.size();
  std::memcpy(model.decoder.bias.data(), flat_weights.data() + offset,
              model.decoder.bias.size() * sizeof(float));
}

void update_federated_weights(AutoencoderModel &model, int worker_id,
                              int world_size, bool should_print) {
  // ---- WEIGHT AVERAGING (only if MPI + more than 1 process)
  if (world_size > 1) {
    auto average_matrix = [world_size](auto &matrix) {
      thread_local std::vector<float> buffer;
      const size_t count = matrix.size();
      buffer.resize(count);

      std::memcpy(buffer.data(), matrix.data(), count * sizeof(float));
      MPI_Allreduce(MPI_IN_PLACE, buffer.data(), static_cast<int>(count),
                    MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

      const float inv_world = 1.0f / static_cast<float>(world_size);
      for (float &value : buffer) {
        value *= inv_world;
      }

      std::memcpy(matrix.data(), buffer.data(), count * sizeof(float));
    };

    average_matrix(model.encoder.weights);
    average_matrix(model.encoder.bias);
    average_matrix(model.decoder.weights);
    average_matrix(model.decoder.bias);

    if (should_print) {
      std::cout << "[MPI] Averaged weights across " << world_size
                << " workers.\n";
    }
  }
}

// void update_federated_weights(AutoencoderModel &model, int worker_id,
//                               int world_size, bool should_print) {
//   // ---- WEIGHT AVERAGING (only if MPI + more than 1 process)
//   if (world_size > 1) {
//     auto local_weights = model.get_weights();
//     auto averaged_weights = local_weights;

//     for (auto &kv : local_weights) {
//       auto &name = kv.first;
//       auto &mat = kv.second;

//       int count = mat.rows() * mat.cols();
//       std::vector<float> sendbuf(mat.data(), mat.data() + count);
//       std::vector<float> recvbuf(count);

//       MPI_Allreduce(sendbuf.data(), recvbuf.data(), count, MPI_FLOAT,
//       MPI_SUM,
//                     MPI_COMM_WORLD);

//       Eigen::Map<Eigen::MatrixXf> averaged_mat(recvbuf.data(), mat.rows(),
//                                                mat.cols());

//       averaged_mat /= static_cast<float>(world_size);
//       averaged_weights[name] = averaged_mat;
//     }

//     model.set_weights(averaged_weights);

//     if (should_print) {
//       std::cout << "[MPI] Averaged weights across " << world_size
//                 << " workers.\n";
//     }
//   }
// }

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

  auto start_wall_clock = [&]() {
#ifdef USE_MPI
    if (world_size > 1) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
    return std::chrono::steady_clock::now();
  };

  auto end_wall_clock = [&]() {
#ifdef USE_MPI
    if (world_size > 1) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
    return std::chrono::steady_clock::now();
  };

  const auto worker_start = start_wall_clock();

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

#ifdef USE_MPI
    update_federated_weights(model, worker_id, world_size, should_print);
#endif // USE_MPI

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

  const auto worker_end = end_wall_clock();
  const double local_total_time_sec =
      std::chrono::duration<double>(worker_end - worker_start).count();
  const double total_time_sec = max_metric(local_total_time_sec);
  const double total_train_samples =
      sum_metric(static_cast<double>(train_filenames.size()));
  const double total_samples_per_sec =
      total_time_sec > 0.0 ? total_train_samples / total_time_sec : 0.0;

  if (logger) {
    logger->add_scalar("test_loss", config.epoch, global_test_loss);
    logger->add_scalar("total_time_sec", config.epoch, total_time_sec);
    logger->add_scalar("total_samples_per_sec", config.epoch,
                       total_samples_per_sec);
  }

  if (should_print) {
    std::cout << "Total worker time=" << total_time_sec
              << " | total_samples_per_sec=" << total_samples_per_sec << "\n";
  }
}
