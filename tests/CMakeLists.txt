include(GoogleTest)

# Test for autoencoder
add_executable(autoencoder_tests autoencoder_tests.cc)
target_link_libraries(autoencoder_tests PRIVATE gtest_main autoencoder_lib)
gtest_discover_tests(autoencoder_tests)
# add_test(NAME AutoencoderTests_base COMMAND autoencoder_tests)
# set_property(TEST AutoencoderTests_base PROPERTY LABELS AutoencoderTests TestAutoencoder)

# Test for linear layers
add_executable(linear_layer_tests linear_layer_tests.cc)
target_link_libraries(linear_layer_tests PRIVATE gtest_main autoencoder_lib gtest)
gtest_discover_tests(linear_layer_tests)

# Alternative:
# include(GoogleTest)

# # Test for autoencoder
# add_executable(autoencoder_tests autoencoder_tests.cc)
# target_link_libraries(autoencoder_tests PRIVATE gtest_main autoencoder_lib)
# gtest_discover_tests(autoencoder_tests)

# # Add labels for autoencoder tests
# set_property(TEST autoencoder_tests PROPERTY LABELS AutoencoderTests "Integration" "Unit")

# # Test for linear layers
# add_executable(linear_layer_tests linear_layer_tests.cc)
# target_link_libraries(linear_layer_tests PRIVATE gtest_main autoencoder_lib gtest)
# gtest_discover_tests(linear_layer_tests)

# # Add labels for linear layer tests
# set_property(TEST LinearLayer.Banana PROPERTY LABELS LinearLayer "UnitTest" "Regression")
# set_property(TEST LinearLayer2.Asd PROPERTY LABELS LinearLayer "UnitTest" "BoundaryTest")
# set_property(TEST LinearLayer3.Bsd PROPERTY LABELS LinearLayer "Integration")

# # Example for adding labels for all tests in the executable
# set_property(TEST linear_layer_tests PROPERTY LABELS LinearLayerTests "UnitTests")
