cmake_minimum_required(VERSION 3.31)

project(autoencoder)

# Set standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_PARALLEL_LEVEL 6)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set build type to RelWithDebInfo if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message("Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable FetchContent module
include(FetchContent)

# -------------------------------
# Fetch stb (Single-Header Library)
# -------------------------------
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG master  # Use master branch (stb is header-only and stable)
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(stb)

# Include stb headers globally
include_directories(${stb_SOURCE_DIR})
# option(BENCHMARK_DOWNLOAD_DEPENDENCIES "Enable downloading of benchmark dependencies" ON)

# ----------------------
# Fetch Eigen
# ----------------------
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(Eigen)

# ----------------------
# Fetch Google Test
# ----------------------
set(INSTALL_GTEST OFF CACHE BOOL "Disable GoogleTest installation" FORCE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.16.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

# ----------------------
# Fetch Benchmark
# ----------------------

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark installation" FORCE)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(benchmark)



# ----------------------
# Add source directory for `autoencoder_lib`
# ----------------------
add_subdirectory(${CMAKE_SOURCE_DIR}/src)

# ----------------------
# Add executable
# ----------------------
add_executable(${PROJECT_NAME} main.cc)

# Link Libraries
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
  Eigen3::Eigen
  benchmark::benchmark
  autoencoder_lib
)

# ----------------------
# Enable Testing
# ----------------------
enable_testing()
add_subdirectory(tests)

# ----------------------
# Enable Benchmarks
# ----------------------
add_subdirectory(benchmarks)

