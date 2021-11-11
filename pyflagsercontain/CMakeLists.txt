cmake_minimum_required(VERSION 2.8.12)
project(pyflagsercontain)

add_subdirectory(pybind11)

pybind11_add_module(pyflagsercontain src/flagser_count_bindings.cpp)
