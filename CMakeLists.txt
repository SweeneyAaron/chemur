cmake_minimum_required(VERSION 3.18)
project(chemur_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_chemur_core cpp/chemur_core.cpp)
target_compile_features(_chemur_core PRIVATE cxx_std_17)

install(TARGETS _chemur_core LIBRARY DESTINATION chemur)

