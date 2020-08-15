#ifndef TORCHAUDIO_KALDI_H
#define TORCHAUDIO_KALDI_H

#include <torch/script.h>
#include <torchaudio/csrc/sox_utils.h>

namespace torchaudio {
namespace kaldi {

torch::Tensor compute_kaldi_pitch_feature(
  const c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal>& signal,
  const double frame_shift_ms=10.,
  const double frame_length_ms=25.,
  const double min_f0=50.,
  const double max_f0=400.,
  const double soft_min_f0=10.,
  const double penalty_factor=0.1,
  const double lowpass_cutoff=1000.,
  const double resample_freq=4000.,
  const double delta_pitch=0.005,
  const double nccf_ballast=7000.,
  const int64_t lowpass_filter_width=1,
  const int64_t upsample_filter_width=5,
  const int64_t max_frames_latency=0,
  const int64_t frames_per_chunk=0,
  const bool simulate_first_pass_online=false,
  const int64_t recompute_frame=500,
  const bool snip_edges=true);

}
} // torchaudio

#endif
