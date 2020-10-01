cmake_minimum_required(VERSION 3.18)

project(libtorchaudio-cpp-example)

find_package(Torch REQUIRED)

add_subdirectory(third_party/torchaudio)


add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}" "${TORCHAUDIO_LIBRARY}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
