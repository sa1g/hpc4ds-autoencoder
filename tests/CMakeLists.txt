include(GoogleTest)

# Test for autoencoder
add_executable(autoencoder_tests autoencoder_tests.cc)
target_link_libraries(autoencoder_tests PRIVATE gtest_main autoencoder_lib)
gtest_discover_tests(autoencoder_tests
    # TEST_PREFIX autoencoder_  # Add a prefix to test names
    PROPERTIES LABELS "autoencoder_tests"  # Assign a label to these tests
)

# Test for linear layers
add_executable(linear_layer_tests linear_layer_tests.cc)
target_link_libraries(linear_layer_tests PRIVATE gtest_main autoencoder_lib gtest)
gtest_discover_tests(linear_layer_tests
    PROPERTIES LABELS "autoencoder_tests"
)

# Test for relu
add_executable(relu_tests relu_tests.cc)
target_link_libraries(relu_tests PRIVATE gtest_main autoencoder_lib gtest)
gtest_discover_tests(relu_tests
    PROPERTIES LABELS "autoencoder_tests"
)

# Test for dataset
add_executable(dataset_tests dataset_tests.cc)
target_include_directories(dataset_tests PRIVATE ${stb_SOURCE_DIR})
target_link_libraries(dataset_tests PRIVATE gtest_main autoencoder_lib gtest)
gtest_discover_tests(dataset_tests
    PROPERTIES LABELS "autoencoder_tests"
)

# Test for sgd
add_executable(sgd_tests sgd_tests.cc)
target_link_libraries(sgd_tests PRIVATE gtest_main autoencoder_lib gtest)
gtest_discover_tests(sgd_tests
    PROPERTIES LABELS "autoencoder_tests"
)