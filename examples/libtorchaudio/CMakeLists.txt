cmake_minimum_required(VERSION 3.5)

project(libtorchaudio-cpp-example)

SET(BUILD_LIBTORCHAUDIO ON CACHE BOOL "Build libtorchaudio")
SET(BUILD_SOX ON CACHE BOOL "Build and link libsox statically to libtorchaudio")

# find_package(Torch REQUIRED)
message("libtorchaudio CMakeLists: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(../.. libtorchaudio)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}" "${TORCHAUDIO_LIBRARY}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
