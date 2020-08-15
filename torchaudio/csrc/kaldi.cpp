#include <torchaudio/csrc/kaldi.h>
#include <matrix/kaldi-matrix.h>
#include <feat/pitch-functions.h>
#include <util/kaldi-io.h> // only for ::kaldi::Output

using kaldi::Matrix;

namespace {
  template<typename Real>
  void printMatSize(const Matrix<Real>& mat) {
    printf("kaldi matrix: (%d, %d)\n", mat.NumRows(), mat.NumCols());
  }
  template<typename Real>
  void printMatrix(const Matrix<Real>& mat) {
    ::kaldi::Output ko("", false);
    mat.Write(ko.Stream(), false);
  }
}

namespace torchaudio {
namespace kaldi {

template<typename Real>
Matrix<Real> convert_to_kaldi_matrix(const torch::Tensor& tensor) {
  if (tensor.dim() != 2) {
    throw std::runtime_error("Input Tensor has to be 2D.");
  }
  Matrix<Real> mat;
  mat.Resize(tensor.size(0), tensor.size(1), ::kaldi::kUndefined);
  const auto in_ptr = tensor.data_ptr<Real>();
  std::copy(in_ptr, in_ptr + tensor.numel(), mat.Data());
  return mat;
}

template<typename Real>
torch::Tensor convert_from_kaldi_matrix(const Matrix<Real>& mat, const c10::ScalarType dtype) {
  const auto rows = mat.NumRows();
  const auto cols = mat.NumCols();

  if (mat.Stride() == cols) {
    return torch::from_blob(
      reinterpret_cast<void*>(const_cast<Real*>(mat.Data())), {rows, cols}, dtype
    ).clone();
  }

  auto tensor = torch::empty({rows, cols}, dtype);
  auto dst = reinterpret_cast<Real*>(tensor.data_ptr());
  for (::kaldi::MatrixIndexT i = 0; i < (::kaldi::MatrixIndexT)rows; ++i) {
    const Real* src = mat.RowData(i);
    std::copy(src, src + cols, dst);
    dst += cols;
  }
  return tensor;
}

torch::Tensor compute_kaldi_pitch_feature(
  const c10::intrusive_ptr<torchaudio::sox_utils::TensorSignal>& signal,
  const double frame_shift_ms,
  const double frame_length_ms,
  const double min_f0,
  const double max_f0,
  const double soft_min_f0,
  const double penalty_factor,
  const double lowpass_cutoff,
  const double resample_freq,
  const double delta_pitch,
  const double nccf_ballast,
  const int64_t lowpass_filter_width,
  const int64_t upsample_filter_width,
  const int64_t max_frames_latency,
  const int64_t frames_per_chunk,
  const bool simulate_first_pass_online,
  const int64_t recompute_frame,
  const bool snip_edges) {

  const auto tensor = signal->getTensor();
  const auto dtype = tensor.dtype();
  if (!(dtype == torch::kFloat32))
    throw std::runtime_error("Input tensor has to be `float32` type.");

  auto mat = convert_to_kaldi_matrix<float>(tensor);
  
  ::kaldi::SubVector<float> waveform(mat, 0);
  ::kaldi::Matrix<float> features;
  ::kaldi::PitchExtractionOptions opts;
  opts.samp_freq = static_cast<float>(signal->getSampleRate());
  opts.frame_shift_ms = static_cast<float>(frame_shift_ms);
  opts.frame_length_ms = static_cast<float>(frame_length_ms);
  opts.min_f0 = static_cast<float>(min_f0);
  opts.max_f0 = static_cast<float>(max_f0);
  opts.soft_min_f0 = static_cast<float>(soft_min_f0);
  opts.penalty_factor = static_cast<float>(penalty_factor);
  opts.lowpass_cutoff = static_cast<float>(lowpass_cutoff);
  opts.resample_freq = static_cast<float>(resample_freq);
  opts.delta_pitch = static_cast<float>(delta_pitch);
  opts.nccf_ballast = static_cast<float>(nccf_ballast);
  opts.lowpass_filter_width = static_cast<int32>(lowpass_filter_width);
  opts.upsample_filter_width = static_cast<int32>(upsample_filter_width);
  opts.max_frames_latency = static_cast<int32>(max_frames_latency);
  opts.frames_per_chunk = static_cast<int32>(frames_per_chunk);
  opts.simulate_first_pass_online = simulate_first_pass_online;
  opts.recompute_frame = static_cast<int32>(recompute_frame);
  opts.snip_edges = snip_edges;

  try {
    ::kaldi::ComputeKaldiPitch(opts, waveform, &features);
  } catch (...) {
    throw std::runtime_error("Failed to compute pitch feature.");
  }

  return convert_from_kaldi_matrix(features, torch::kFloat32);
}

}  // namespace kaldi
}  // namespace torchaudio
