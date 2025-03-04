cmake_minimum_required(VERSION 3.31)

project(autoencoder)

# Set standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set build type to RelWithDebInfo if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message("Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable downloading dependencies for Google Benchmark
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)

# Include the stb directory for stb_image.h
include_directories(external/stb)

# Add external libraries
add_subdirectory(external/eigen)
add_subdirectory(external/benchmark)

# Add executable
add_executable(
  ${PROJECT_NAME} 
  main.cc
)
  
# Link Eigen and Google Benchmark
target_link_libraries(
  ${PROJECT_NAME} 
  Eigen3::Eigen 
  benchmark::benchmark
)
    
#
target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

# Set include directories
add_subdirectory(${CMAKE_SOURCE_DIR}/src)
# add_subdirectory(src)
