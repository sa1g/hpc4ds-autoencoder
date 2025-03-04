set(SOURCE_FILES
  prova.cc
)


target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/data
)
add_subdirectory(data)

target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})

