set(SOURCE_FILES
  linear.hh
  relu.hh
  sigmoid.hh
  mse.hh
)

target_sources(autoencoder_lib PRIVATE ${SOURCE_FILES})