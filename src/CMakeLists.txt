set(SOURCE_FILES
  # utils.cc
)


target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/data
  ${CMAKE_CURRENT_SOURCE_DIR}/utils

)
add_subdirectory(data)
add_subdirectory(utils)


target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})

