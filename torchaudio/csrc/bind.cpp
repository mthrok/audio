#include <torch/extension.h>
#include <torchaudio/csrc/sox_effects.h>
#include <torchaudio/csrc/sox_io.h>
#include <torchaudio/csrc/sox_utils.h>

PYBIND11_MODULE(_torchaudio, m) {
  py::class_<torchaudio::sox_utils::TensorSignal>(m, "TensorSignal")
    .def(py::init<torch::Tensor, int64_t, bool>())
    .def("get_tensor", &torchaudio::sox_utils::TensorSignal::getTensor)
    .def("get_sample_rate", &torchaudio::sox_utils::TensorSignal::getSampleRate)
    .def("get_channels_first", &torchaudio::sox_utils::TensorSignal::getChannelsFirst);

  py::class_<torchaudio::sox_io::SignalInfo>(m, "SignalInfo")
    .def("get_sample_rate", &torchaudio::sox_io::SignalInfo::getSampleRate)
    .def("get_num_channels", &torchaudio::sox_io::SignalInfo::getNumChannels)
    .def("get_num_frames", &torchaudio::sox_io::SignalInfo::getNumFrames);

  m.def("get_info", &torchaudio::sox_io::get_info, "Gets information about an audio file.");
  m.def("load_audio_file", &torchaudio::sox_io::load_audio_file2, "Load audio file into Tensor.");
  m.def("save_audio_file", &torchaudio::sox_io::save_audio_file2, "Save Tensor as audio file.");

  m.def("apply_effects_tensor", &torchaudio::sox_effects::apply_effects_tensor, "Apply sox effects to Tensor");
  m.def("apply_effects_file", &torchaudio::sox_effects::apply_effects_file, "Apply sox effects to Tensor");
}
