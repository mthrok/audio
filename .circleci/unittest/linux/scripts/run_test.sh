#!/usr/bin/env bash

set -e

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac


eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

apt install -y llvm
export PATH="${PWD}/third_party/install/bin/:${PATH}"
python -m torch.utils.collect_env
if [ "${os}" == MacOSX ] ; then
    export ASAN_OPTIONS=symbolize=1
    export ASAN_SYMBOLIZER_PATH="$(which llvm-symbolizer)"
    export LD_PRELOAD="${LD_PRELOAD} libasan.so"
    pytest -q -n auto --dist=loadscope --cov=torchaudio --junitxml=test-results/junit.xml --durations 20 test
else
    pytest -v --cov=torchaudio --junitxml=test-results/junit.xml --durations 20 test
fi
