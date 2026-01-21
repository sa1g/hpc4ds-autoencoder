set(SOURCE_FILES
  rand.cc
  common.cc
  loops.cc
  worker.cc
  stb_impl.cc 
)

target_sources(autoencoder_lib PRIVATE ${SOURCE_FILES})