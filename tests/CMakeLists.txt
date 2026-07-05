
include(GoogleTest)

set(AUTOENCODER_TEST_DISCOVERY_TIMEOUT 60 CACHE STRING
    "Seconds allowed for GoogleTest test discovery")

function(discover_autoencoder_tests target_name)
    gtest_discover_tests(${target_name}
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT ${AUTOENCODER_TEST_DISCOVERY_TIMEOUT}
        PROPERTIES LABELS "autoencoder_tests"
    )
endfunction()

# Test for autoencoder
add_executable(autoencoder_tests autoencoder_tests.cc)
target_link_libraries(autoencoder_tests PRIVATE gtest_main autoencoder_lib)
discover_autoencoder_tests(autoencoder_tests)

# Test for linear layers
add_executable(linear_layer_tests linear_layer_tests.cc)
target_link_libraries(linear_layer_tests PRIVATE gtest_main autoencoder_lib gtest)
discover_autoencoder_tests(linear_layer_tests)

# Test for relu
add_executable(relu_tests relu_tests.cc)
target_link_libraries(relu_tests PRIVATE gtest_main autoencoder_lib gtest)
discover_autoencoder_tests(relu_tests)

# Test for dataset
add_executable(dataset_tests dataset_tests.cc)
target_include_directories(dataset_tests PRIVATE ${stb_SOURCE_DIR})
target_link_libraries(dataset_tests PRIVATE gtest_main autoencoder_lib gtest)
discover_autoencoder_tests(dataset_tests)

# Test for sgd
add_executable(sgd_tests sgd_tests.cc)
target_link_libraries(sgd_tests PRIVATE gtest_main autoencoder_lib gtest)
discover_autoencoder_tests(sgd_tests)

# Test for common/random_split
add_executable(random_split_tests random_split_tests.cc)
target_link_libraries(random_split_tests PRIVATE gtest_main autoencoder_lib gtest)
discover_autoencoder_tests(random_split_tests)


# Test for sigmoid
add_executable(sigmoid_tests sigmoid_tests.cc)
target_link_libraries(sigmoid_tests PRIVATE gtest_main autoencoder_lib gtest)
discover_autoencoder_tests(sigmoid_tests)

# Test for mean squared error
add_executable(mse_tests mse_tests.cc)
target_link_libraries(mse_tests PRIVATE gtest_main autoencoder_lib gtest)
discover_autoencoder_tests(mse_tests)

# Test for model
add_executable(model_tests model_tests.cc)
target_link_libraries(model_tests PRIVATE gtest_main autoencoder_lib gtest)
discover_autoencoder_tests(model_tests)

