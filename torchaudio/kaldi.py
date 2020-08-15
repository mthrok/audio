from typing import NamedTuple

import torch

from torchaudio._internal import (
    module_utils as _mod_utils,
    misc_ops as _misc_ops,
)

@_mod_utils.requires_module('torchaudio._torchaudio')
def compute_kaldi_pitch_feature(
        tensor: torch.Tensor, sample_rate: int, channels_first: bool,
        frame_shift_ms=10.,
        frame_length_ms=25.,
        min_f0=50.,
        max_f0=400.,
        soft_min_f0=10.,
        penalty_factor=0.1,
        lowpass_cutoff=1000.,
        resample_freq=4000.,
        delta_pitch=0.005,
        nccf_ballast=7000.,
        lowpass_filter_width=1,
        upsample_filter_width=5,
        max_frames_latency=0,
        frames_per_chunk=0,
        simulate_first_pass_online=False,
        recompute_frame=500,
        snip_edges=True,
    ) -> torch.Tensor:
    signal = torch.classes.torchaudio.TensorSignal(tensor, sample_rate, channels_first)
    return torch.ops.torchaudio.kaldi_compute_kaldi_pitch_feature(
        signal, frame_shift_ms, frame_length_ms, min_f0, max_f0, soft_min_f0,
        penalty_factor, lowpass_cutoff, resample_freq, delta_pitch, nccf_ballast, lowpass_filter_width,
        upsample_filter_width, max_frames_latency, frames_per_chunk, simulate_first_pass_online,
        recompute_frame, snip_edges,
    )
