// pybind/feat/wave_reader_pybind.cc

// Copyright 2019   Microsoft Corporation (author: Xingyu Na)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/wave_reader_pybind.h"
#include "feat/wave-reader.h"
#include "util/table-types.h"

using namespace kaldi;

void pybind_wave_reader(py::module& m) {
  m.attr("kWaveSampleMax") = py::cast(kWaveSampleMax);

  py::class_<WaveInfo>(m, "WaveInfo")
      .def("IsStreamed", &WaveInfo::IsStreamed)
      .def("SampFreq", &WaveInfo::SampFreq)
      .def("SampleCount", &WaveInfo::SampleCount)
      .def("Duration", &WaveInfo::Duration)
      .def("NumChannels", &WaveInfo::NumChannels)
      .def("BlockAlign", &WaveInfo::BlockAlign)
      .def("DataBytes", &WaveInfo::DataBytes);

  py::class_<WaveData>(m, "WaveData")
      .def("Duration", &WaveData::Duration)
      .def("Data", &WaveData::Data);

  py::class_<SequentialTableReader<WaveHolder>>(m, "SequentialWaveReader")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("Open", &SequentialTableReader<WaveHolder>::Open)
      .def("Done", &SequentialTableReader<WaveHolder>::Done)
      .def("Key", &SequentialTableReader<WaveHolder>::Key)
      .def("FreeCurrent", &SequentialTableReader<WaveHolder>::FreeCurrent)
      .def("Value", &SequentialTableReader<WaveHolder>::Value)
      .def("Next", &SequentialTableReader<WaveHolder>::Next)
      .def("IsOpen", &SequentialTableReader<WaveHolder>::IsOpen)
      .def("Close", &SequentialTableReader<WaveHolder>::Close);

  py::class_<RandomAccessTableReader<WaveHolder>>(m, "RandomAccessWaveReader")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("Open", &RandomAccessTableReader<WaveHolder>::Open)
      .def("IsOpen", &RandomAccessTableReader<WaveHolder>::IsOpen)
      .def("Close", &RandomAccessTableReader<WaveHolder>::Close)
      .def("HasKey", &RandomAccessTableReader<WaveHolder>::HasKey)
      .def("Value", &RandomAccessTableReader<WaveHolder>::Value);

  py::class_<SequentialTableReader<WaveInfoHolder>>(m, "SequentialWaveInfoReader")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("Open", &SequentialTableReader<WaveInfoHolder>::Open)
      .def("Done", &SequentialTableReader<WaveInfoHolder>::Done)
      .def("Key", &SequentialTableReader<WaveInfoHolder>::Key)
      .def("FreeCurrent", &SequentialTableReader<WaveInfoHolder>::FreeCurrent)
      .def("Value", &SequentialTableReader<WaveInfoHolder>::Value)
      .def("Next", &SequentialTableReader<WaveInfoHolder>::Next)
      .def("IsOpen", &SequentialTableReader<WaveInfoHolder>::IsOpen)
      .def("Close", &SequentialTableReader<WaveInfoHolder>::Close);

  py::class_<RandomAccessTableReader<WaveInfoHolder>>(m, "RandomAccessWaveInfoReader")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("Open", &RandomAccessTableReader<WaveInfoHolder>::Open)
      .def("IsOpen", &RandomAccessTableReader<WaveInfoHolder>::IsOpen)
      .def("Close", &RandomAccessTableReader<WaveInfoHolder>::Close)
      .def("HasKey", &RandomAccessTableReader<WaveInfoHolder>::HasKey)
      .def("Value", &RandomAccessTableReader<WaveInfoHolder>::Value);

}

