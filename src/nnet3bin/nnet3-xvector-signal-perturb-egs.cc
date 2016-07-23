// nnet3bin/nnet3-xvector-signal-perturb-egs.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/signal-distort.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h" 
namespace kaldi {
namespace nnet3 {

// This function applies different type of perturbation to input_egs.
// random distortion of inputs, random shifts, adding additive noise,
// random time stretch and random negations are different type of 
// distortions used in this function.
void ApplyPerturbation(XvectorPerturbOptions opts,
                       const Matrix<BaseFloat> &input_egs,
                       Matrix<BaseFloat> *noise_egs,
                       Matrix<BaseFloat> *perturb_egs) {

  PerturbXvectorSignal perturb_xvector(opts);
  
  Matrix<BaseFloat> shifted_egs(input_egs);
  // Generate random shift samples to shift egs. 
  if (opts.max_shift != 0.0) {
    int32 max_shift_int = static_cast<int32>(opts.max_shift * opts.frame_dim);
    // shift input_egs using random shift. 
    int32 eg_dim = input_egs.NumCols() - opts.frame_dim,
      shift = RandInt(0, max_shift_int);
    shifted_egs.CopyFromMat(input_egs.Range(0, input_egs.NumRows(), shift, eg_dim));
  }
  
  Matrix<BaseFloat> rand_distort_shifted_egs(shifted_egs);
  if (opts.rand_distort) {
    // randomly generate an zero-phase FIR filter with no zeros.
    // In future, we can select trucated part of room impluse response
    // and convolve it with input_egs.
    perturb_xvector.ComputeAndApplyRandDistortion(shifted_egs,
                                  &rand_distort_shifted_egs);
  }

  if (noise_egs) { 
    // select random block of noise egs and add to input_egs
    // number of additive noises should be larger than number of input-egs.
    KALDI_ASSERT(noise_egs->NumRows() >= input_egs.NumRows());
    if (noise_egs->NumRows() < input_egs.NumRows()) {
      // repeat the noise_egs_mat blocks to have same length block
      // and randomly perturb the rows.
    } else {
      // Select random submatrix out of noise_egs and add it to perturb_egs.
      // we should shuffle noise_egs before passing them to this binary.
      int32 start_row_ind = RandInt(0, noise_egs->NumRows() - input_egs.NumRows()),
        start_col_ind = RandInt(0, noise_egs->NumCols() - input_egs.NumCols()); 
      rand_distort_shifted_egs.AddMat(1.0, noise_egs->Range(start_row_ind, input_egs.NumRows(),
                                      start_col_ind, input_egs.NumCols()));
    }
  }
  // Perturb speed of signal egs
  Matrix<BaseFloat> warped_distorted_shifted_egs(rand_distort_shifted_egs);
  if (opts.max_time_stretch != 0.0) 
    perturb_xvector.TimeStretch(rand_distort_shifted_egs, 
                                &warped_distorted_shifted_egs);
   
  // If nagation is true, the sample values are randomly negated
  // with some probability.
  if (opts.negation) {
   
  }
}

} // end of namespace nnet3
} // end of namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;   
    typedef kaldi::int64 int64;


    const char *usage =
        "Corrupts  the examples supplied via input pipe with different type of distortions\n"
        "such as additive noise, negation, random time shifts or random distortion.\n"
        "Usage: nnet3-xvector-signal-perturb-egs [options...] <egs-especifier> <egs-wspecifier>\n"
        "e.g.\n"
        "nnet3-xvector-signal-perturb-egs --noise-egs=noise.egs\n"
        "--max-shift=0.2 --max-speed-perturb=0.1 --negation=true\n"
        "ark:input.egs akr:distorted.egs\n";
    ParseOptions po(usage);

    XvectorPerturbOptions perturb_opts;
    perturb_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string examples_rspecifier = po.GetArg(1),
      examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
     
    NnetExampleWriter example_writer(examples_wspecifier);

    int64 num_read = 0, num_written = 0;

    Matrix<BaseFloat> *noise_mat = NULL;
    // read additive noise egs if it is specified.
    if (!perturb_opts.noise_egs.empty()) {
      SequentialNnetExampleReader noise_reader(perturb_opts.noise_egs);
      const NnetExample &noise_egs = noise_reader.Value();
      const NnetIo &noise_io = noise_egs.io[0];
      noise_io.features.CopyToMat(noise_mat);
       
    }

    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      std::string key = example_reader.Key();
      const NnetExample &input_eg = example_reader.Value();
      const NnetIo &input_eg_io = input_eg.io[0];
      NnetExample *perturb_eg = new NnetExample();
      Matrix<BaseFloat> perturb_eg_mat, 
        input_eg_mat;
      input_eg_io.features.CopyToMat(&input_eg_mat);
      ApplyPerturbation(perturb_opts, input_eg_mat, noise_mat, &perturb_eg_mat);
      perturb_eg->io.resize(1.0);
      perturb_eg->io[0].features.SwapFullMatrix(&perturb_eg_mat);
      example_writer.Write(key, *perturb_eg);
      num_written++;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
