// featbin/signal-distort.cc

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



#include "feat/signal-distort.h"

namespace kaldi {

// randomly disturb the input signal using a band-pass filter with no zeros.
void PerturbXvectorSignal::ComputeAndApplyRandDistortion(const MatrixBase<BaseFloat> &input_egs,
                                                    Matrix<BaseFloat> *perturb_egs) {
  // Generate impluse response |H(w)| using nonzero random sequence and smooth them 
  // using moving-average window with small window size.
  // For simplicity, assume zero-phase response and H(w) = |H(w)|.
  // num_fft_samp = 512
  int32 num_fft_samp = 512;
  Vector<BaseFloat> im_response(num_fft_samp);

}

// Stretches the time axis for input egs without fixing the pitch value.
// It changes the speed and duration of the input signal without fixing pitch.
// The output  y w.r.t input x is going to be y(t - offset) = x(stretch * (t - offset)),
// where offset is the time index which the signal is stretches along that and the input
// and output are the same for t = offset.
// ArbitraryResample class is used to generate resampled output for different time-stretches.
// The output y is the stretched form of the input, x, and stretch value is randomely generated
// between [1 - max_stretch, 1 + max_stretch].
// y[(m - n + 2 * t)/2] = x[(1 + stretch) * (m - n + 2 * t)/2] for t = 0,..,n   
void PerturbXvectorSignal::TimeStretch(const MatrixBase<BaseFloat> &input_egs,  
                                       Matrix<BaseFloat> *perturb_egs) {
  Matrix<BaseFloat> in_mat(input_egs), 
    out_mat(perturb_egs->NumRows(), perturb_egs->NumCols());
  int32 input_dim = input_egs.NumCols(), 
    dim = perturb_egs->NumCols();
  Vector<BaseFloat> samp_points_secs(dim);
  BaseFloat samp_freq = 2000, 
    max_stretch = opts_.max_time_stretch;
  // we stretch the middle part of the example and the input should be expanded
  // by extra frame to be larger than the output length => s * (m+n)/2 < m.
  // y((m - n + 2 * t)/2) = x(s * (m - n + 2 * t)/2) for t = 0,..,n 
  // where m = dim(x) and n = dim(y).
  KALDI_ASSERT(input_dim > dim * ((1.0 + max_stretch) / (1.0 - max_stretch)));
  // Generate random stretch value between -max_stretch, max_stretch.
  int32 max_stretch_int = static_cast<int32>(max_stretch * 1000);
  BaseFloat stretch = static_cast<BaseFloat>(RandInt(-max_stretch_int, max_stretch_int) / 1000.0); 
  if (abs(stretch) > 0) {
    int32 num_zeros = 4; // Number of zeros of the sinc function that the window extends out to.
    BaseFloat filter_cutoff_hz = samp_freq * 0.475; // lowpass frequency that's lower than 95% of 
                                                    // the Nyquist.
    for (int32 i = 0; i < dim; i++) 
      samp_points_secs(i) = static_cast<BaseFloat>(((1.0 + stretch) * 
        (0.5 * (input_dim - dim) + i))/ samp_freq);

    ArbitraryResample time_resample(input_dim, samp_freq,
                                    filter_cutoff_hz, 
                                    samp_points_secs,
                                    num_zeros);
    time_resample.Resample(in_mat, &out_mat);
  } else {
    int32 offset = static_cast<BaseFloat>(0.5 * (input_egs.NumCols() - perturb_egs->NumCols()));
    out_mat.CopyFromMat(input_egs.Range(0, input_egs.NumRows(), offset, perturb_egs->NumCols()));
  }
  perturb_egs->CopyFromMat(out_mat);
}


} // end of namespace kaldi
