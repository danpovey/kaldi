// featbin/signal-distort.h

// Copyright 2016 Pegah Ghahremani

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

#ifndef KALDI_SIGNAL_DISTORT_H_
#define KALDI_SIGNAL_DISTORT_H_

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

#include "base/kaldi-error.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"

#include "feat/resample.h"
#include "matrix/matrix-functions.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {

// options class for distorting signals in egs
struct XvectorPerturbOptions {
  BaseFloat max_shift; 
  BaseFloat max_time_stretch;
  int32 frame_dim;
  bool negation; 
  bool rand_distort;
  std::string noise_egs;
  XvectorPerturbOptions(): max_shift(0.2),
                           max_time_stretch(0.2),
                           frame_dim(80),
                           negation(false),
                           rand_distort(false) { }
  void Register(OptionsItf *opts) { 
    opts->Register("max-shift", &max_shift, "Maximum random shift relative"
                "to frame length applied to egs.");
    opts->Register("max-speed-perturb", &max_time_stretch,
                   "Max speed perturbation applied on egs.");
    opts->Register("frame-dim", &frame_dim,
                   "The numebr of samples in input frame as product of frame_length by samp_freq.");
    opts->Register("negation", &negation, "If true, the input value is negated randomly.");
    opts->Register("noise-egs", &noise_egs, "If supplied, the additive noise is added.");
    opts->Register("rand_distort", &rand_distort, "If true, the signal is slightly changes"
                   "using some designed FIR filter with no zeros.");
  }
};

class PerturbXvectorSignal {
 public:
  PerturbXvectorSignal(XvectorPerturbOptions opts): opts_(opts) { };

  void ComputeAndApplyRandDistortion(const MatrixBase<BaseFloat> &input_egs,
                                     Matrix<BaseFloat> *perturb_egs);

  void TimeStretch(const MatrixBase<BaseFloat> &input_egs,
                   Matrix<BaseFloat> *perturb_egs);

 private:
  XvectorPerturbOptions opts_;
};

} // end of namespace kaldi
#endif // KALDI_SIGNAL_DISTORT_H_
