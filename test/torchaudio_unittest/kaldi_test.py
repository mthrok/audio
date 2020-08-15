import itertools
from parameterized import parameterized

import torch
import torchaudio.kaldi

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    PytorchTestCase,
    skipIfNoExec,
    skipIfNoExtension,
    get_asset_path,
    get_wav_data,
    save_wav,
    sox_utils,
)


@skipIfNoExtension
class TestKaldi(TempDirMixin, PytorchTestCase):
    def test_tensor_conversion(self):
        tensor = torch.rand(1, 16000, dtype=torch.float)
        sample_rate = 16000
        print(tensor.shape)
        print(tensor)
        feat = torchaudio.kaldi.compute_kaldi_pitch_feature(tensor, sample_rate, True)
        print(feat.shape)
        print(feat)
