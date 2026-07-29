cmake_minimum_required(VERSION 3.18)
project(chemeleonx_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_chemeleonx_core cpp/chemeleonx_core.cpp)
target_compile_features(_chemeleonx_core PRIVATE cxx_std_17)

install(TARGETS _chemeleonx_core LIBRARY DESTINATION chemeleonx)

