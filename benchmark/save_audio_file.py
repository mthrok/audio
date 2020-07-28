import os
import time
import tempfile

import torch
import torchaudio
import torchaudio._torchaudio


def test_torchscript(path: str, num_trials: int, num_channels: int, num_frames: int, sample_rate: int):
    channels_first = True
    src = 2.0 * torch.rand((num_channels, num_frames), dtype=torch.float32) - 1.0
    signal = torch.classes.torchaudio.TensorSignal(src, sample_rate, channels_first)
    for _ in range(num_trials):
        torch.ops.torchaudio.sox_io_save_audio_file(path, signal, 0.)


def test_pybind11(path: str, num_trials: int, num_channels: int, num_frames: int, sample_rate: int):
    channels_first = True
    src = 2.0 * torch.rand((num_channels, num_frames), dtype=torch.float32) - 1.0
    signal = torchaudio._torchaudio.TensorSignal(src, sample_rate, channels_first)
    for _ in range(num_trials):
        torchaudio._torchaudio.save_audio_file(path, signal, 0.)


def run_test(num_trials: int = 1000, num_channels: int = 2, duration: float = 1.0, sample_rate: int = 44100):
    num_frames: int = int(duration * sample_rate)
    print(f'* Testing save(str, Tensor, double) -> None')
    print(f'  - Tensor shape: ({num_channels}, {num_frames})')
    print(f'  - #Runs: {num_trials}')

    with tempfile.TemporaryDirectory() as temp_dir:
        print('  * Testing pybind11')
        audio_path = os.path.join(temp_dir, 'pybind11.wav')
        t0 = time.monotonic()
        test_pybind11(audio_path, num_trials, num_channels, num_frames, sample_rate)
        elapsed = time.monotonic() - t0
        print(f'    Elapsed: {elapsed} seconds: Average: {elapsed/num_trials}')

        print('  * Testing Torchscript')
        audio_path = os.path.join(temp_dir, 'torchscript.wav')
        t0 = time.monotonic()
        test_torchscript(audio_path, num_trials, num_channels, num_frames, sample_rate)
        elapsed = time.monotonic() - t0
        print(f'    Elapsed: {elapsed} seconds: Average: {elapsed/num_trials}')

        print('  * Testing JIT')
        audio_path = os.path.join(temp_dir, 'jit.wav')
        jit_path = os.path.join(temp_dir, 'jit.zip')
        torch.jit.script(test_torchscript).save(jit_path)
        test_func = torch.jit.load(jit_path)
        t0 = time.monotonic()
        test_func(audio_path, num_trials, num_channels, num_frames, sample_rate)
        elapsed = time.monotonic() - t0
        print(f'    Elapsed: {elapsed} seconds: Average: {elapsed/num_trials}')
