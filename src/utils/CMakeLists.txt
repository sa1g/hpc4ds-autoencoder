set(SOURCE_FILES
  rand.cc
  common.cc
  loops.cc
  worker.cc
)

target_sources(autoencoder_lib PRIVATE ${SOURCE_FILES})