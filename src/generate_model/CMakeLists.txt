cmake_minimum_required(VERSION 2.8.12)
project(genModels)
add_subdirectory(./generateModel/pybind11)
pybind11_add_module(genModels ./generateModel/wrapper_bindings.cpp)
