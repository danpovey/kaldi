// pybind/tree-accu_pybind.cc

// Copyright (c) 2019, Johns Hopkins University (Yenda Trmal<jtrmal@gmail.com>)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "hmm/tree-accu_pybind.h"
#include "hmm/tree-accu.h"

using namespace kaldi;

void pybind_hmm_tree_accu(py::module& m) {
  py::class_<AccumulateTreeStatsOptions>(m, "AccumulateTreeStatsOptions")
    .def(py::init<>())
    .def_readwrite("var_floor", &AccumulateTreeStatsOptions::var_floor)
    .def_readwrite("ci_phones_str", &AccumulateTreeStatsOptions::ci_phones_str)
    .def_readwrite("phone_map_rxfilename", &AccumulateTreeStatsOptions::phone_map_rxfilename)
    .def_readwrite("collapse_pdf_classes", &AccumulateTreeStatsOptions::collapse_pdf_classes)
    .def_readwrite("context_width", &AccumulateTreeStatsOptions::context_width)
    .def_readwrite("central_position", &AccumulateTreeStatsOptions::central_position)
  ;

  py::class_<AccumulateTreeStatsInfo>(m, "AccumulateTreeStatsInfo")
    .def(py::init<const AccumulateTreeStatsOptions &>())
    .def_readwrite("var_floor", &AccumulateTreeStatsInfo::var_floor)
    .def_readwrite("context_width", &AccumulateTreeStatsInfo::context_width)
    .def_readwrite("central_position", &AccumulateTreeStatsInfo::central_position)
    .def_readwrite("ci_phones",  &AccumulateTreeStatsInfo::ci_phones)
    .def_readwrite("phone_map",  &AccumulateTreeStatsInfo::phone_map)
  ;

  m.def("ReadPhoneMap", &ReadPhoneMap,
        "Read a mapping from one phone set to another",
        py::arg("phone_map_rxfilename"), py::arg("phone_map"));
}
