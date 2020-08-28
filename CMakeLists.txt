cmake_minimum_required(VERSION 3.14)

project(torchaudio)

option(BUILD_PYTHON_EXTENSION "Build Python extension" OFF)
option(BUILD_LIBTORCHAUDIO "Build C++ Library" ON)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard whose features are requested to build this target.")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(third_party)
add_subdirectory(csrc)
