#!/usr/bin/env bash

set -eux

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
build_dir="${this_dir}/build"

model_file="${this_dir}/epoch_200.zip"
input_file="${this_dir}/input.wav"
output_file="${this_dir}/output.wav"

# rm -rf "${build_dir}"
mkdir -p "${build_dir}"

(
cd build
cmake -GNinja \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
      ..

cmake --build . --target help
cmake --build . --target main
./main "${model_file}" "${input_file}" "${output_file}"
)
