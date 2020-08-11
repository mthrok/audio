import argparse
import os.path
from pathlib import Path

import torch
import torchaudio
import torchaudio.models


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug log",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        required=True,
        help="Check point file from which model parameters are loaded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory where separaed audio files are saved.",
    )
    parser.add_argument(
        "input_files",
        type=Path,
        nargs="+",
        help="Files on which separation is performed.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    torchaudio.set_audio_backend("sox_io")

    state_dict = torch.load(args.model_file)
    sample_rate = state_dict["sample_rate"]
    num_speakers = state_dict["num_speakers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchaudio.models.ConvTasNet(
        num_sources=num_speakers, enc_kernel_size=sample_rate * 2 // 1000,
    )
    model.load_state_dict(state_dict["model"])
    model.to(device)

    for input_file in args.input_files:
        print(input_file)
        input, sr = torchaudio.load(str(input_file))
        assert sr == sample_rate
        output = model(input.unsqueeze(0).to(device)).squeeze(0).to('cpu')
        output_path = args.output_dir / f"{input_file.name}"
        torchaudio.save(str(output_path), output, sample_rate)


if __name__ == "__main__":
    main()
