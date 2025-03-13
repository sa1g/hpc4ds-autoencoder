set(This AutoencoderTests)

set(Sources
    autoencoder_tests.cc
)

add_executable(${This} ${Sources})

target_link_libraries(${This} PUBLIC
    gtest_main  # Google Test's main function
    autoencoder_lib  # Link the main project's code (without main.cc)
)

add_test(
    NAME ${This}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${workdir}"
    COMMAND ${This}
)

set_property(TEST "${This}" PROPERTY LABELS TESTLABEL AutoencoderTests)