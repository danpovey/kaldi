// nnet3/nnet-component-itf.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen

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

#include <iterator>
#include <sstream>
#include <iomanip>
#include "rnnlm/rnnlm-component-itf.h"
#include "rnnlm/rnnlm-component.h"
#include "nnet3/nnet-general-component.h"
#include "rnnlm/nnet-parse.h"
#include "nnet3/nnet-computation-graph.h"


namespace kaldi {
//using namespace nnet3;
namespace rnnlm {

LmInputComponent* LmInputComponent::ReadNew(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token); // e.g. "<SigmoidComponent>".
  token.erase(0, 1); // erase "<".
  token.erase(token.length()-1); // erase ">".
  LmInputComponent *ans = NewComponentOfType(token);
  if (!ans)
    KALDI_ERR << "Unknown component type " << token;
  ans->Read(is, binary);
  return ans;
}

// static
LmInputComponent* LmInputComponent::NewComponentOfType(const std::string &component_type) {
  LmInputComponent *ans = NULL;
  if (component_type == "LmLinearComponent") {
    ans = new LmLinearComponent();
  } else if (component_type == "LmNaturalGradientLinearComponent") {
    ans = new LmNaturalGradientLinearComponent();
  }
  if (ans != NULL) {
    KALDI_ASSERT(component_type == ans->Type());
  }
  return ans;
}

LmOutputComponent* LmOutputComponent::ReadNew(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token); // e.g. "<SigmoidComponent>".
  token.erase(0, 1); // erase "<".
  token.erase(token.length()-1); // erase ">".
  LmOutputComponent *ans = NewComponentOfType(token);
  if (!ans)
    KALDI_ERR << "Unknown component type " << token;
  ans->Read(is, binary);
  return ans;
}

// static
LmOutputComponent* LmOutputComponent::NewComponentOfType(const std::string &component_type) {
  LmOutputComponent *ans = NULL;
  if (component_type == "NaturalGradientAffineImportanceSamplingComponent") {
    ans = new NaturalGradientAffineImportanceSamplingComponent();
  } else if (component_type == "AffineImportanceSamplingComponent") {
    ans = new AffineImportanceSamplingComponent();
  }
  if (ans != NULL) {
    KALDI_ASSERT(component_type == ans->Type());
  }
  return ans;
}

void LmInputComponent::InitLearningRatesFromConfig(ConfigLine *cfl) {
  cfl->GetValue("learning-rate", &learning_rate_);
  cfl->GetValue("learning-rate-factor", &learning_rate_factor_);
  max_change_ = 0.0;
  cfl->GetValue("max-change", &max_change_);
  if (learning_rate_ < 0.0 || learning_rate_factor_ < 0.0 || max_change_ < 0.0)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void LmInputComponent::ReadUpdatableCommon(std::istream &is, bool binary) {
  std::ostringstream opening_tag;
  opening_tag << '<' << this->Type() << '>';
  std::string token;
  ReadToken(is, binary, &token);
  if (token == opening_tag.str()) {
    // if the first token is the opening tag, then
    // ignore it and get the next tag.
    ReadToken(is, binary, &token);
  }
  if (token == "<LearningRateFactor>") {
    ReadBasicType(is, binary, &learning_rate_factor_);
    ReadToken(is, binary, &token);
  } else {
    learning_rate_factor_ = 1.0;
  }
  if (token == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient_);
    ReadToken(is, binary, &token);
  } else {
    is_gradient_ = false;
  }
  if (token == "<MaxChange>") {
    ReadBasicType(is, binary, &max_change_);
    ReadToken(is, binary, &token);
  } else {
    max_change_ = 0.0;
  }
  if (token == "<LearningRate>") {
    ReadBasicType(is, binary, &learning_rate_);
  } else {
    KALDI_ERR << "Expected token <LearningRate>, got "
              << token;
  }
}

void LmInputComponent::WriteUpdatableCommon(std::ostream &os,
                                              bool binary) const {
  std::ostringstream opening_tag;
  opening_tag << '<' << this->Type() << '>';
  std::string token;
  WriteToken(os, binary, opening_tag.str());
  if (learning_rate_factor_ != 1.0) {
    WriteToken(os, binary, "<LearningRateFactor>");
    WriteBasicType(os, binary, learning_rate_factor_);
  }
  if (is_gradient_) {
    WriteToken(os, binary, "<IsGradient>");
    WriteBasicType(os, binary, is_gradient_);
  }
  if (max_change_ > 0.0) {
    WriteToken(os, binary, "<MaxChange>");
    WriteBasicType(os, binary, max_change_);
  }
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
}


std::string LmInputComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim() << ", learning-rate="
         << LearningRate();
  if (is_gradient_)
    stream << ", is-gradient=true";
  if (learning_rate_factor_ != 1.0)
    stream << ", learning-rate-factor=" << learning_rate_factor_;
  return stream.str();
}

void LmOutputComponent::InitLearningRatesFromConfig(ConfigLine *cfl) {
  cfl->GetValue("learning-rate", &learning_rate_);
  cfl->GetValue("learning-rate-factor", &learning_rate_factor_);
  max_change_ = 0.0;
  cfl->GetValue("max-change", &max_change_);
  if (learning_rate_ < 0.0 || learning_rate_factor_ < 0.0 || max_change_ < 0.0)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}


void LmOutputComponent::ReadUpdatableCommon(std::istream &is, bool binary) {
  std::ostringstream opening_tag;
  opening_tag << '<' << this->Type() << '>';
  std::string token;
  ReadToken(is, binary, &token);
  if (token == opening_tag.str()) {
    // if the first token is the opening tag, then
    // ignore it and get the next tag.
    ReadToken(is, binary, &token);
  }
  if (token == "<LearningRateFactor>") {
    ReadBasicType(is, binary, &learning_rate_factor_);
    ReadToken(is, binary, &token);
  } else {
    learning_rate_factor_ = 1.0;
  }
  if (token == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient_);
    ReadToken(is, binary, &token);
  } else {
    is_gradient_ = false;
  }
  if (token == "<MaxChange>") {
    ReadBasicType(is, binary, &max_change_);
    ReadToken(is, binary, &token);
  } else {
    max_change_ = 0.0;
  }
  if (token == "<LearningRate>") {
    ReadBasicType(is, binary, &learning_rate_);
  } else {
    KALDI_ERR << "Expected token <LearningRate>, got "
              << token;
  }
}

void LmOutputComponent::WriteUpdatableCommon(std::ostream &os,
                                              bool binary) const {
  std::ostringstream opening_tag;
  opening_tag << '<' << this->Type() << '>';
  std::string token;
  WriteToken(os, binary, opening_tag.str());
  if (learning_rate_factor_ != 1.0) {
    WriteToken(os, binary, "<LearningRateFactor>");
    WriteBasicType(os, binary, learning_rate_factor_);
  }
  if (is_gradient_) {
    WriteToken(os, binary, "<IsGradient>");
    WriteBasicType(os, binary, is_gradient_);
  }
  if (max_change_ > 0.0) {
    WriteToken(os, binary, "<MaxChange>");
    WriteBasicType(os, binary, max_change_);
  }
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
}

std::string LmOutputComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim() << ", learning-rate="
         << LearningRate();
  if (is_gradient_)
    stream << ", is-gradient=true";
  if (learning_rate_factor_ != 1.0)
    stream << ", learning-rate-factor=" << learning_rate_factor_;
  return stream.str();
}

} // namespace nnet3
} // namespace kaldi
