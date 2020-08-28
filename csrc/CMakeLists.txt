################################################################################
# libtorchaudio.so
################################################################################
if(BUILD_LIBTORCHAUDIO)
add_library(
  libtorchaudio
  SHARED
  sox_io.cpp
  sox_utils.cpp
  sox_effects.cpp
  sox_effects_chain.cpp
  register.cpp
)

set_target_properties(libtorchaudio PROPERTIES PREFIX "")

target_include_directories(
  libtorchaudio
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  libtorchaudio
  "${TORCH_LIBRARIES}"
  sox
  mad
  flac
  mp3lame
  opusfile
  opus
  vorbis
  ogg
)

install(
  TARGETS
  libtorchaudio
)
endif()

################################################################################
# _torchaudio.so
################################################################################
if (BUILD_PYTHON_EXTENSION)
find_package(PythonLibs REQUIRED)

add_library(
  _torchaudio
  SHARED
  sox.cpp
  sox_io.cpp
  sox_utils.cpp
  sox_effects.cpp
  sox_effects_chain.cpp
  register.cpp
)

set_target_properties(_torchaudio PROPERTIES PREFIX "")

target_include_directories(
  _torchaudio
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PYTHON_INCLUDE_DIRS}
)

# See https://github.com/pytorch/pytorch/issues/38122
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(
  _torchaudio
  "${TORCH_LIBRARIES}"
  "${TORCH_PYTHON_LIBRARY}"
  "${PYTHON_LIBRARIES}"
  sox
  mad
  flac
  mp3lame
  opusfile
  opus
  vorbis
  ogg
)

# We do not define install for _torchaudio.so
# The location of installation is controlled by "CMAKE_LIBRARY_OUTPUT_DIRECTORY"
endif()
