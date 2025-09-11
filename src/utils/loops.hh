#ifndef __AUTOENCODER_LOOPS_HH__

#include "model.hh"
#include "mse.hh"
#include "dataset.hh"
#include "common.hh"

/**
 * Train the model for 1 epoch on the specified dataset.
 * 
 * @return Average loss for this epoch.
 */
float train(std::string text, const experiment_config &config, Dataloader &dataloader, AutoencoderModel &model, MSE &criterion);

float test(std::string text, Dataloader &dataloader, AutoencoderModel &model, MSE &criterion);

#endif // __AUTOENCODER_LOOPS_HH