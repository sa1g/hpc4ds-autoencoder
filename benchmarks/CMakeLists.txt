set(BENCHMARK_SOURCES
    benchmark0.cc
    benchmark1.cc
    # Add more benchmarks here...
)

foreach(SOURCE_FILE ${BENCHMARK_SOURCES})
    get_filename_component(EXEC_NAME ${SOURCE_FILE} NAME_WE)

    add_executable(${EXEC_NAME} ${SOURCE_FILE})

    target_link_libraries(${EXEC_NAME} PRIVATE
        benchmark::benchmark  
        autoencoder_lib       
        Eigen3::Eigen         
    )

    add_test(NAME ${EXEC_NAME} COMMAND ${EXEC_NAME})
endforeach()
