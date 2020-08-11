#!/usr/bin/env python3
"""Dump Con-TasNet as JIT object"""

import argparse
from pathlib import Path

import torch
import torchaudio


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", required=True, type=Path, help="Model data")
    parser.add_argument("--output-file", required=True, help="Output TorchScript path.")
    return parser.parse_args()


class SourceSeparationPipeline(torch.nn.Module):
    def __init__(self, conv_tasnet, sample_rate: int):
        super().__init__()
        self.conv_tasnet = conv_tasnet
        self.sample_rate = sample_rate

    def forward(self, input_path: str, output_path: str):
        data, sample_rate = torchaudio.load(input_path)
        assert sample_rate == self.sample_rate
        output = self.conv_tasnet(data.unsqueeze(0)).squeeze(0)
        torchaudio.save(output_path, output, self.sample_rate)


def _main():
    args = _parse_args()

    torchaudio.set_audio_backend("sox_io")

    state_dict = torch.load(args.model_file)
    sample_rate = state_dict["sample_rate"]
    num_speakers = state_dict["num_speakers"]

    model = torchaudio.models.ConvTasNet(
        num_sources=num_speakers, enc_kernel_size=sample_rate * 2 // 1000,
    )
    model.load_state_dict(state_dict["model"])

    pipeline = SourceSeparationPipeline(model, sample_rate)

    module = torch.jit.script(pipeline)
    module.save(args.output_file)


if __name__ == "__main__":
    _main()
