# set(This autoencoder_tests)

# set(Sources
#     autoencoder_tests.cc
# )

# add_executable(${This} ${Sources})

# target_link_libraries(${This} PUBLIC
#     gtest_main
#     # autoencoder
# )

# add_test(
#     NAME ${This}
#     COMMAND ${This}
# )



set(This autoencoder_tests)

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
    COMMAND ${This}
)