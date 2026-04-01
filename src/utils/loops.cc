#include "loops.hh"
#include "sgd.hh"

#include <cmath>
#include <iostream>
#include <limits>

float train(std::string text, const experiment_config &config,
            Dataloader &dataloader, AutoencoderModel &model, MSE &criterion)
{

  float epoch_loss{0};
  int num_batches{0};

  for (auto &batch : dataloader)
  {
    auto prediction = model.forward(batch);
    auto loss = criterion.mse_loss(batch, prediction);

    if (!std::isfinite(loss))
    {
      std::cerr << "Non-finite loss detected in train() at batch "
                << (num_batches + 1) << "/" << dataloader.get_num_batches()
                << std::endl;
      return std::numeric_limits<float>::quiet_NaN();
    }

    auto grad = criterion.mse_gradient(batch, prediction);
    model.backward(batch, grad);
    sgd(config.lr, model.encoder, model.decoder);

    num_batches++;
    epoch_loss += loss;

    // printf("%s batch %i/%i | Loss: %.3f\n", text.c_str(), num_batches,
    //  dataloader.get_num_batches(), loss);
  }

  float average_epoch_loss = epoch_loss / dataloader.get_num_batches();

  return average_epoch_loss;
}

float test(std::string text, Dataloader &dataloader, AutoencoderModel &model,
           MSE &criterion)
{
  float epoch_loss{0};
  int num_batches{0};

  for (auto &batch : dataloader)
  {
    auto prediction = model.forward(batch);
    auto loss = criterion.mse_loss(batch, prediction);

    if (!std::isfinite(loss))
    {
      std::cerr << "Non-finite loss detected in test() at batch "
                << (num_batches + 1) << "/" << dataloader.get_num_batches()
                << std::endl;
      return std::numeric_limits<float>::quiet_NaN();
    }

    num_batches++;
    epoch_loss += loss;

    printf("%s batch %i/%i | Loss: %.3f\n", text.c_str(), num_batches,
           dataloader.get_num_batches(), loss);
  }

  return epoch_loss / dataloader.get_num_batches();
}
