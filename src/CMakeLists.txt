# Define the library and list the source files
set(SOURCE_FILES
    # Add core source files here (if any)
)

# Define the library before using it
add_library(autoencoder_lib STATIC ${SOURCE_FILES})

# Set include directories for the library
target_include_directories(autoencoder_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/nn
    ${CMAKE_CURRENT_SOURCE_DIR}/optim
    ${CMAKE_CURRENT_SOURCE_DIR}/model
)

# Add subdirectories
add_subdirectory(data)
add_subdirectory(utils)
add_subdirectory(nn)
add_subdirectory(optim)
add_subdirectory(model)

# Ensure the main executable (autoencoder) can access the library
target_link_libraries(autoencoder_lib PUBLIC Eigen3::Eigen)
target_link_libraries(autoencoder_lib PRIVATE
    tensorboard_logger
    protobuf::libprotobuf
)
if(WITH_OPENMP)
    target_link_libraries(autoencoder_lib PRIVATE OpenMP::OpenMP_CXX)
endif()
