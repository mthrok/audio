import math
from typing import Optional, Tuple
from itertools import permutations

import torch


def _power(
        signal: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        keepdim: bool = False,
):
    power = signal.pow(2)
    if mask is None:
        return power.mean(axis=2, keepdim=keepdim)
    denom = mask.sum(axis=2, keepdim=keepdim)
    return (mask * power).sum(axis=2, keepdim=keepdim) / denom


def _normalize(
        signal: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
):
    if mask is not None:
        signal = signal * mask
    normalized = signal - signal.mean(axis=2, keepdim=True)
    if mask is not None:
        normalized = normalized * mask
    return normalized


def sdr(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes signal to distortion ratio (in decibel).

    Args:
        estimate (torch.Tensor): Estimated (reconstructed) signal.
            Shape: [batch, channels, time frame]
        reference (torch.Tensor): Reference signal.
            Shape: [batch, channels, time frame]
        mask (Optional[torch.Tensor]): Binary mask to indicate padded value (0) or valid value (1).
            Shape: [batch, 1, time frame],
        epsilon (float): Constant value for stabilizing division.

    Returns:
        torch.Tensor: Signal to distortion ratio (in decibel).
            Shape: [batch, speaker]
    """
    ref_pow = _power(reference, mask)
    err_pow = _power(estimate - reference, mask)
    return 10 * torch.log10(ref_pow) - 10 * torch.log10(err_pow)


def si_sdr(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        epsilon: float = 1e-8
) -> torch.Tensor:
    """Computes scale-invariant signal-to-distortion ratio. (in decibel).

    1. scale the reference signal with power(s_est * s_ref) / powr(s_ref * s_ref)
    2. compute SDR between adjusted estimate and reference.

    Args:
        estimate (torch.Tensor): Estimtaed signal.
            Shape: [batch, speakers (can be 1), time frame]
        reference (torch.Tensor): Reference signal.
            Shape: [batch, speakers, time frame]
        mask (Optional[torch.Tensor]): Binary mask to indicate padded value (0) or valid value (1).
            Shape: [batch, 1, time frame]
        epsilon (float): constant value used to stabilize division.

    Returns:
        torch.Tensor: Scale-invariant source-to-distortion ratio. (in decibel)
        Shape: [batch, speaker]

    References:
        - SDR - half-baked or well done?
          J. Le Roux, S. Wisdom, H. Erdogan and J. R. Hershey
          https://arxiv.org/abs/1811.02508
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Y. Luo and N. Mesgarani
          https://arxiv.org/abs/1809.07454

    Notes:
        This implementation is based on the following code
        https://github.com/naplab/Conv-TasNet/blob/e66d82a8f956a69749ec8a4ae382217faa097c5c/utility/sdr.py#L34-L56
    """
    estimate = _normalize(estimate, mask)
    reference = _normalize(reference, mask)

    ref_pow = _power(reference, mask, keepdim=True)
    mix_pow = _power((estimate * reference), mask, keepdim=True)

    scale = mix_pow / (ref_pow + epsilon)

    reference = reference * scale
    return sdr(estimate, reference, mask)


class PIT(torch.nn.Module):
    """Applies utterance-level source permutation

    Computes the maxium possible value of the given utility function
    over the permutations of the sources.

    Args:
        utility_func (function):
            Function that computes the utility (opposite of loss) with signature of
            (extimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor
            where input Tensors are shape of [batch, sources, frame] and
            the output Tensor is shape of [batch, sources].

    References:
        - Multi-talker Speech Separation with Utterance-level Permutation Invariant Training of
          Deep Recurrent Neural Networks
          Morten Kolbæk, Dong Yu, Zheng-Hua Tan and Jesper Jensen
          https://arxiv.org/abs/1703.06284
    """

    def __init__(self, utility_func):
        super().__init__()
        self.utility_func = utility_func

    def forward(
            self,
            estimate: torch.Tensor,
            reference: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute utterance-level PIT Loss

        Args:
            estimate (torch.Tensor): Estimated source signals.
                Shape: [bacth, sources, time frame]
            reference (torch.Tensor): Reference (original) source signals.
                Shape: [batch, sources, time frame]
            mask (Optional[torch.Tensor]): Binary mask to indicate padded value (0) or valid value (1).
                Shape: [batch, 1, time frame]
            epsilon (float): constant value used to stabilize division.

        Returns:
            torch.Tensor: Maximum criterion over the source permutation.
                Shape: [batch, ]
        """
        assert estimate.shape == reference.shape

        batch_size, num_sources = reference.shape[:2]
        num_permute = math.factorial(num_sources)

        util_mat = torch.zeros(
            batch_size, num_permute, dtype=estimate.dtype, device=estimate.device
        )
        for i, idx in enumerate(permutations(range(num_sources))):
            util = self.utility_func(estimate, reference[:, idx, :], mask=mask)
            util_mat[:, i] = util.mean(dim=1)  # take the average over source dimension
        return util_mat.max(dim=1).values


_sdr_pit = PIT(utility_func=sdr)
_si_sdr_pit = PIT(utility_func=si_sdr)


def si_sdr_pit(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mask: Optional[torch.Tensor] = None):
    """Computes the maximum SI-SDR over source permutations

    Args:
        estimate (torch.Tensor): Estimtaed signal.
            Shape: [batch, sources (can be 1), time frame]
        reference (torch.Tensor): Reference signal.
            Shape: [batch, sources, time frame]
        mask (Optional[torch.Tensor]): Binary mask to indicate padded value (0) or valid value (1).
            Shape: [batch, 1, time frame]

    Returns:
        torch.Tensor: scale-invariant source-to-distortion ratio.
        Shape: [batch, source]

    References:
        - Multi-talker Speech Separation with Utterance-level Permutation Invariant Training of
          Deep Recurrent Neural Networks
          Morten Kolbæk, Dong Yu, Zheng-Hua Tan and Jesper Jensen
          https://arxiv.org/abs/1703.06284
        - SDR - half-baked or well done?
          J. Le Roux, S. Wisdom, H. Erdogan and J. R. Hershey
          https://arxiv.org/abs/1811.02508
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454

    Notes:
        This function is tested to produce the exact same result as the reference implementation,
        *when the inputs have 0-mean*
        https://github.com/naplab/Conv-TasNet/blob/e66d82a8f956a69749ec8a4ae382217faa097c5c/utility/sdr.py#L107-L153
    """
    return _si_sdr_pit(estimate, reference, mask)


def sdr_pit(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the maximum SDR over source permutations.

    For computing ``SDR(estimate, reference)``, PIT (permutation invariant training) is applied,
    so that best combination of sources between the reference signals and the esimate signals
    are picked.

    Args:
        estimate (torch.Tensor): Estimated source signals.
            Shape: [batch, speakers, time frame]
        reference (torch.Tensor): Reference (original) source signals.
            Shape: [batch, speakers, time frame]
        mask (Optional[torch.Tensor]): Binary mask to indicate padded value (0) or valid value (1).
            Shape: [batch, 1, time frame]

    Returns:
        torch.Tensor: Improved SDR. Shape: [batch, ]

    References:
        - Multi-talker Speech Separation with Utterance-level Permutation Invariant Training of
          Deep Recurrent Neural Networks
          Morten Kolbæk, Dong Yu, Zheng-Hua Tan and Jesper Jensen
          https://arxiv.org/abs/1703.06284
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454
    """
    return _sdr_pit(estimate, reference, mask=mask)  # [batch, ]


def sdri(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the improvement of SDR from mixture signal to separated signal

    Args:
        estimate (torch.Tensor): Estimated source signals.
            Shape: [batch, speakers, time frame]
        reference (torch.Tensor): Reference (original) source signals.
            Shape: [batch, speakers, time frame]
        mix (torch.Tensor): Mixed souce signals, from which the setimated signals were generated.
            Shape: [batch, speakers == 1, time frame]
        mask (torch.Tensor): Mask to indicate padded value (0) or valid value (1).
            Shape: [batch, 1, time frame]

    Returns:
        torch.Tensor: Improved SDR. Shape: [batch, ]

    References:
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454
    """
    base_sdr = sdr(mix, reference, mask)  # [batch, sources]
    sdr_ = sdr_pit(estimate, reference, mask)  # [batch, ]
    return sdr_ - base_sdr.mean(dim=1)


def si_sdri(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    mix: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the improvement of scale-invariant SDR from mixture signal to separated signal

    Args:
        estimate (torch.Tensor): Estimated source signals.
            Shape: [batch, speakers, time frame]
        reference (torch.Tensor): Reference (original) source signals.
            Shape: [batch, speakers, time frame]
        mix (torch.Tensor): Mixed souce signals, from which the setimated signals were generated.
            Shape: [batch, speakers == 1, time frame]
        mask (torch.Tensor, optional): Mask to indicate padded value (0) or valid value (1).
            Shape: [batch, 1, time frame]

    Returns:
        torch.Tensor: Improved SI-SDR. Shape: [batch, ]

    References:
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454
    """
    base_si_sdr = si_sdr(mix, reference, mask)  # [batch, sources]
    si_sdr_ = si_sdr_pit(estimate, reference, mask)  # [batch, ]
    return si_sdr_ - base_si_sdr.mean(dim=1)
