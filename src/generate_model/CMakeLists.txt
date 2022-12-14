cmake_minimum_required(VERSION 2.8.12)
project(generate_model)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
pybind11_add_module(generate_model ./generateModel/wrapper_bindings.cpp)
