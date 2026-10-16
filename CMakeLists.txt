cmake_minimum_required(VERSION 3.24)
project(vecdb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(vecdb src/dataset.cpp)
target_include_directories(vecdb PUBLIC include)
target_compile_options(vecdb PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
  # Uses an installed GoogleTest if there is one, otherwise downloads this pinned version.
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    FIND_PACKAGE_ARGS NAMES GTest)
  FetchContent_MakeAvailable(googletest)

  add_executable(vecdb_tests tests/dataset_test.cpp)
  target_link_libraries(vecdb_tests PRIVATE vecdb GTest::gtest_main)
  target_compile_definitions(vecdb_tests PRIVATE VECDB_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

  include(GoogleTest)
  gtest_discover_tests(vecdb_tests)
endif()
