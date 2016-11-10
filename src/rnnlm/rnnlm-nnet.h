// Copyright    2016  Hainan Xu

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

#ifndef KALDI_RNNLM_RNNLM_NNET_H
#define KALDI_RNNLM_RNNLM_NNET_H

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet3/nnet-nnet.h"
#include "rnnlm/rnnlm-component.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>

namespace kaldi {
namespace rnnlm {

class LmNnet {
 public:
  LmNnet() {
    nnet_ = new Nnet();
  }

  ~LmNnet() {
    delete input_projection_;
    delete output_projection_;
    delete output_layer_;
    delete nnet_;
  }

  Nnet* GetNnet() {
    return nnet_;
  }

  void Read(std::istream &is, bool binary);

  void ReadConfig(std::istream &config_file);

  void Write(std::ostream &os, bool binary) const;

  LmNnet* Copy() {
    LmNnet* other = new LmNnet();
    other->input_projection_ = input_projection_->Copy();
    other->output_projection_ = output_projection_->Copy();
    other->output_layer_ = output_layer_->Copy();
    other->nnet_ = nnet_->Copy();

    return other;
  }

  LmAffineComponent* I() {
    return dynamic_cast<LmAffineComponent*>(input_projection_);
  }
  LmAffineComponent* O() {
    return dynamic_cast<LmAffineComponent*>(output_projection_);
  }
  LmNonlinearComponent* N() {
    return dynamic_cast<LmNonlinearComponent*>(output_layer_);
  }

 private:
  LmComponent* input_projection_;      // Affine
  LmComponent* output_projection_;    // Affine
  LmComponent* output_layer_;         // Softmax 
  Nnet* nnet_;
};

}
}
#endif
