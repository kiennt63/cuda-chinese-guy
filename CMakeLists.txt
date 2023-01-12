cmake_minimum_required(VERSION 3.10)

project(cuda_chinese_guy)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(week-1)
