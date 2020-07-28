import os
import json
import time
import tempfile
from typing import List

import torch
import torchaudio
import torchaudio._torchaudio


def test_torchscript(
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool,
        effects: List[List[str]],
        num_trials: int):
    signal = torch.classes.torchaudio.TensorSignal(src, sample_rate, channels_first)
    for _ in range(num_trials):
        torch.ops.torchaudio.sox_effects_apply_effects_tensor(signal, effects)


def test_pybind11(
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool,
        effects: List[List[str]],
        num_trials: int):
    signal = torchaudio._torchaudio.TensorSignal(src, sample_rate, channels_first)
    for _ in range(num_trials):
        torchaudio._torchaudio.apply_effects_tensor(signal, effects)


def run_test(num_trials: int = 1000, num_channels: int = 2, num_frames: int = 44100, sample_rate: int = 44100):
    print(f'Testing apply_effects(Tensor, bool, int, List[List[str]]) -> (Tensor, bool, int)')
    print(f'Tensor shape: ({num_channels}, {num_frames})')
    print(f'#Runs: {num_trials}')
    effects = [
        [
            "mcompand",
            "0.005,0.1 -47,-40,-34,-34,-17,-33", "100",
            "0.003,0.05 -47,-40,-34,-34,-17,-33", "400",
            "0.000625,0.0125 -47,-40,-34,-34,-15,-33", "1600",
            "0.0001,0.025 -47,-40,-34,-34,-31,-31,-0,-30", "6400",
            "0,0.025 -38,-31,-28,-28,-0,-25"
        ],
    ]
    print(f'Effects:\n{json.dumps(effects, indent=4)}')

    with tempfile.TemporaryDirectory() as temp_dir:
        channels_first = True
        src = 2.0 * torch.rand((num_channels, num_frames), dtype=torch.float32) - 1.0

        print('Testing pybind11')
        t0 = time.monotonic()
        test_pybind11(src, sample_rate, channels_first, effects, num_trials)
        elapsed = time.monotonic() - t0
        print(f'Elapsed: {elapsed} seconds: Average: {elapsed/num_trials}')

        print('Testing Torchscript')
        t0 = time.monotonic()
        test_torchscript(src, sample_rate, channels_first, effects, num_trials)
        elapsed = time.monotonic() - t0
        print(f'Elapsed: {elapsed} seconds: Average: {elapsed/num_trials}')

        print('Testing JIT')
        jit_path = os.path.join(temp_dir, 'jit.zip')
        torch.jit.script(test_torchscript).save(jit_path)
        test_func = torch.jit.load(jit_path)
        t0 = time.monotonic()
        test_func(src, sample_rate, channels_first, effects, num_trials)
        elapsed = time.monotonic() - t0
        print(f'Elapsed: {elapsed} seconds: Average: {elapsed/num_trials}')
