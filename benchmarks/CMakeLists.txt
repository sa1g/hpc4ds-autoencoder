set(BENCHMARK_SOURCES
    b_dataloader.cc
    b_linear.cc
    b_mse.cc
    b_relu.cc
    b_sgd.cc
    b_sigmoid.cc
    b_train.cc
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

    if(WITH_OPENMP)
        target_link_libraries(${EXEC_NAME} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${EXEC_NAME} PRIVATE EIGEN_USE_OPENMP)
    endif()

    add_test(NAME ${EXEC_NAME} COMMAND ${EXEC_NAME})
endforeach()

