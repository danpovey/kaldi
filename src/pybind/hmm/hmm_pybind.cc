// hmm_pybind.cc

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

#include <string>
#include "kaldi_pybind.h"
#include "hmm/tree-accu_pybind.h"


PYBIND11_MODULE(kaldi_hmm_pybind, m) {
  m.doc() =
      "pybind11 binding of some things from kaldi's src/hmm directory."
      "Source is in $(KALDI_ROOT)/src/pybind/hmm/hmm_pybind.cc";
  pybind_hmm_tree_accu(m);
}

