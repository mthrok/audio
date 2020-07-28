import os
import time
import tempfile

import torch
import torchaudio
import torchaudio._torchaudio


def test_torchscript(num_trials=100, audio_format='wav', num_channels=2, num_frames=44100, sample_rate=44100):
    channels_first = True
    src = 2.0 * torch.rand((num_channels, num_frames), dtype=torch.float32) - 1.0
    signal = torch.classes.torchaudio.TensorSignal(src, sample_rate, channels_first)
    compression = {
        'wav': 0.,
        'mp3': -4.5,
        'flac': 8.,
        'ogg': 3.,
        'vorbis': 3.,
    }[audio_format]
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, f'test.{audio_format}')
        t0 = time.time()
        for _ in range(num_trials):
            torch.ops.torchaudio.sox_io_save_audio_file(path, signal, compression)
        t1 = time.time()
    return t1 - t0


def test_pybind11(num_trials=100, audio_format='wav', num_channels=2, num_frames=44100, sample_rate=44100):
    channels_first = True
    src = 2.0 * torch.rand((num_channels, num_frames), dtype=torch.float32) - 1.0
    signal = torchaudio._torchaudio.TensorSignal(src, sample_rate, channels_first)
    compression = {
        'wav': 0.,
        'mp3': -4.5,
        'flac': 8.,
        'ogg': 3.,
        'vorbis': 3.,
    }[audio_format]
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, f'test.{audio_format}')
        t0 = time.time()
        for _ in range(num_trials):
            torchaudio._torchaudio.save_audio_file(path, signal, compression)
        t1 = time.time()
    return t1 - t0
