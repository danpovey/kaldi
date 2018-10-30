// nnet3/convolution-cudnn-test.cc

// Copyright 2018    Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet3/convolution-cudnn.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet3 {
namespace cudnn_convolution {

// for testing purposes, create a random ConvolutionComputation
static void GetRandomConfig(ConvolutionComputationConfig *config) {
  config->num_images = RandInt(1, 10);
  config->num_channels_out = RandInt(1, 10);
  config->num_channels_in = RandInt(1, 10);

  config->filter_height = RandInt(1, 3);
  config->filter_width = RandInt(1, 3);

  config->filter_stride_vertical = RandInt(1, 2);
  config->filter_stride_horizontal = RandInt(1, 2);
  // Setting a dilation other than 1 is currently causing a segfault.
  // It is documented that when the format used is CUDNN_TENSOR_NHWC,
  // dilation must be 1 for all dimensions, so the segfault is just
  // letting us know that dilated convolutions are not supported.
  config->filter_dilation_vertical = 1; //RandInt(1, 2);
  config->filter_dilation_horizontal = 1; //RandInt(1, 2);

  config->input_image_height = RandInt(10, 20);
  config->input_image_width = RandInt(10, 20);

  config->zero_padding_vertical = RandInt(0, 1);
  config->zero_padding_horizontal = RandInt(0, 1);

  config->Check();
  config->ComputeOutputImageSize();
}

void TestConvolutionComputationConfig() {
  for (int32 i = 0; i < 100; i++) {
    ConvolutionComputationConfig config;
    GetRandomConfig(&config);
    std::ostringstream os;
    bool binary = true;
    config.Write(os, binary);

    ConvolutionComputationConfig config2;
    std::istringstream is(os.str());
    config2.Read(is, binary);
    std::ostringstream os2;
    config2.Write(os2, binary);
    KALDI_ASSERT(os.str() == os2.str());
  }
}


void ConvolveForwardWithGpu(
    const ConvolutionComputation &computation,
    const MatrixBase<BaseFloat> &input,
    const MatrixBase<BaseFloat> &params,
    const VectorBase<BaseFloat> &bias,
    MatrixBase<BaseFloat> *output) {
  CuMatrix<BaseFloat> input_gpu(input.NumRows(), input.NumCols(),
                                kUndefined, kStrideEqualNumCols);
  input_gpu.CopyFromMat(input);
  CuMatrix<BaseFloat> params_gpu(params.NumRows(), params.NumCols(),
                                kUndefined, kStrideEqualNumCols);
  params_gpu.CopyFromMat(params);
  CuVector<BaseFloat> bias_gpu(bias);
  CuMatrix<BaseFloat> output_gpu(output->NumRows(), output->NumCols(),
                                 kSetZero, kStrideEqualNumCols);
  computation.ConvolveForward(input_gpu, params_gpu, &bias_gpu, &output_gpu);
  output->CopyFromMat(output_gpu);
}



// Tests that the CPU version gives the expected results.  The output
// has to be inspected by a human.
void TestConvolutionComputationForward(
    const ConvolutionComputation &computation,
    bool use_gpu) {
  const ConvolutionComputationConfig &c = computation.Config();

  Matrix<BaseFloat> input(c.num_images * c.input_image_width,
                          c.input_image_height * c.num_channels_in,
                          kSetZero, kStrideEqualNumCols),
      output(c.num_images * c.output_image_width,
             c.output_image_height * c.num_channels_out,
             kSetZero, kStrideEqualNumCols),
      params(c.num_channels_out,
             c.num_channels_in * c.filter_width * c.filter_height,
             kSetZero, kStrideEqualNumCols);

  // One parameter and one channel of input pixel will be nonzero-- for testing purposes.

  int32 n = RandInt(0, c.num_images - 1),
      input_w = RandInt(0, c.input_image_width - 1),
      input_h = RandInt(0, c.input_image_height - 1),
      input_c = RandInt(0, c.num_channels_in - 1);
  input(n * c.input_image_width + input_w,
        input_h * c.num_channels_in + input_c) = 2.0;

  int32 output_c = RandInt(0, c.num_channels_out - 1),
      filter_w = RandInt(0, c.filter_width - 1),
      filter_h = RandInt(0, c.filter_height - 1);

  params(output_c,
         input_c * c.filter_width * c.filter_height +
         filter_w * c.filter_height +
         filter_h) = 3.0;

  Vector<BaseFloat> bias(c.num_channels_out);

  if (use_gpu) {
    ConvolveForwardWithGpu(computation, input, params, bias, &output);
  } else {
    computation.ConvolveForward(input, params, &bias,
                                &output);
  }

  KALDI_LOG << "Have nonzero input for n=" << n
            << ", w=" << input_w << ", h=" << input_h
            << ", input_channel=" << input_c;
  KALDI_LOG << "Have nonzero filter for w="
            << filter_w << ", h=" << filter_h
            << ", output_channel=" << output_c;
  bool found_nonzero = false;
  for (int32 n = 0; n < c.num_images; n++) {
    for (int32 w = 0; w < c.output_image_width; w++) {
      for (int32 h = 0; h < c.output_image_height; h++) {
        for (int32 ch = 0; ch < c.num_channels_out; ch++) {
          BaseFloat val = output(n * c.output_image_width + w,
                                 h * c.num_channels_out + ch);
          if (val != 0.0) {
            found_nonzero = true;
            KALDI_LOG << "Found nonzero value " << val << " for image n="
                      << n << ", w=" << w << ", h=" << h
                      << ", output_channel=" << ch;
          }
        }
      }
    }
  }
  if (!found_nonzero)
    KALDI_WARN << "Found no nonzero value, sum is " << output.Sum();


}

void TestConvolutionComputation() {
  for (int32 i = 0; i < 100; i++) {
    ConvolutionComputationConfig config;
    GetRandomConfig(&config);

    {
      std::ostringstream os;
      config.Write(os, false);
      KALDI_LOG << "Config is: " << os.str();
    }

    ConvolutionComputation computation(config);

    std::ostringstream os;
    bool binary = true;
    computation.Write(os, binary);

    ConvolutionComputation computation2;
    std::istringstream is(os.str());
    computation2.Read(is, binary);
    std::ostringstream os2;
    computation2.Write(os2, binary);
    KALDI_ASSERT(os.str() == os2.str());
    KALDI_LOG << "About to test without GPU.";
    TestConvolutionComputationForward(computation2, false);
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().Enabled()) {
      KALDI_LOG << "About to test with GPU";
      TestConvolutionComputationForward(computation2, true);
    }
#endif
  }
}


/**


// for testing purposes, create a set of input and output indexes for
// a convolution computation that are computable given this model.
static void GetRandomConvolutionIndexes(const ConvolutionModel &model,
                                        std::vector<Index> *input_indexes,
                                        std::vector<Index> *output_indexes) {
  KALDI_ASSERT(model.Check());

  std::vector<std::pair<int32, int32> > n_x_pairs;
  int32 num_n_x_pairs = RandInt(1, 3);
  for (int32 i = 0; i < num_n_x_pairs; i++) {
    int32 n = RandInt(0, 3), x = RandInt(0, 1);
    n_x_pairs.push_back(std::pair<int32, int32>(n, x));
  }
  SortAndUniq(&n_x_pairs);
  num_n_x_pairs = n_x_pairs.size();


  // 'output_t_values' is the set of *possible* output
  // t values; we'll later sub-sample from these.
  std::vector<int32> output_t_values;

  {
    int32 out_t_start = RandInt(-5, 5), out_t_step = RandInt(1, 3),
        num_t_out = RandInt(1, 4);
    for (int32 i = 0; i < num_t_out; i++)
      output_t_values.push_back(out_t_start + i * out_t_step);
  }

  input_indexes->clear();
  output_indexes->clear();
  for (size_t i = 0; i < n_x_pairs.size(); i++) {
    std::vector<int32> chosen_output_t_values;
    while (chosen_output_t_values.empty()) {
      for (size_t j = 0; j < output_t_values.size(); j++)
        if (RandInt(0, 1) != 0)
          chosen_output_t_values.push_back(output_t_values[j]);
    }
    KALDI_ASSERT(IsSortedAndUniq(chosen_output_t_values));

    std::set<int32> required_input_t_values,
        usable_input_t_values;
    for (size_t j = 0; j < chosen_output_t_values.size(); j++) {
      std::set<int32>::const_iterator iter;
      int32 t_out = chosen_output_t_values[j];
      for (iter = model.required_time_offsets.begin();
           iter != model.required_time_offsets.end(); iter++) {
        int32 offset = *iter;
        required_input_t_values.insert(t_out + offset);
      }
      for (iter = model.all_time_offsets.begin();
           iter != model.all_time_offsets.end(); iter++) {
        int32 offset = *iter;
        usable_input_t_values.insert(t_out + offset);
      }
    }

    // add to output_indexes
    for (size_t j = 0; j < chosen_output_t_values.size(); j++) {
      int32 t_out = chosen_output_t_values[j];
      Index index;
      index.n = n_x_pairs[i].first;
      index.x = n_x_pairs[i].second;
      index.t = t_out;
      output_indexes->push_back(index);
    }

    std::vector<int32> chosen_input_t_values(required_input_t_values.begin(),
                                             required_input_t_values.end());
    for (std::set<int32>::const_iterator iter = usable_input_t_values.begin();
         iter != usable_input_t_values.end(); ++iter) {
      int32 t = *iter;
      if (RandInt(0, 1) == 0)
        chosen_input_t_values.push_back(t);
    }
    SortAndUniq(&chosen_input_t_values);

    // add to input_indexes
    for (size_t j = 0; j < chosen_input_t_values.size(); j++) {
      int32 t_in = chosen_input_t_values[j];
      Index index;
      index.n = n_x_pairs[i].first;
      index.x = n_x_pairs[i].second;
      index.t = t_in;
      input_indexes->push_back(index);
    }
  }
}


void UnitTestTimeHeightConvolutionIo() {
  for (int32 i = 0; i < 10; i++) {
    KALDI_LOG << "iter = " << i;
    // Create a ConvolutionModel and test its I/O.
    ConvolutionModel conv_model;
    GetRandomConvolutionModel(&conv_model);
    std::ostringstream os1, os2;
    bool binary = (RandInt(0, 1) == 0);
    conv_model.Write(os1, binary);
    std::istringstream is(os1.str());
    ConvolutionModel conv_model2;
    conv_model2.Read(is, binary);
    conv_model2.Write(os2, binary);
    KALDI_ASSERT(os1.str() == os2.str() && conv_model2.Check());
  }
}

void TestComputationIo(const ConvolutionComputation &computation) {
  std::ostringstream os1, os2;
  bool binary = (RandInt(0, 1) == 0);
  computation.Write(os1, binary);
  std::istringstream is(os1.str());
  ConvolutionComputation computation2;
  computation2.Read(is, binary);
  computation2.Write(os2, binary);
  KALDI_ASSERT(os1.str() == os2.str());
  computation2.Check();
}


// This function exects indexes.size() == matrix->NumRows();
// it sets to zero any row i of the matrix for which
// indexes[i].t == kNoTime.
void ZeroBlankRows(const std::vector<Index> &indexes,
                   CuMatrix<BaseFloat> *matrix) {
  KALDI_ASSERT(static_cast<int32>(indexes.size()) == matrix->NumRows());
  int32 num_rows = matrix->NumRows();
  if (num_rows == 0) return;
  Vector<BaseFloat> mask(num_rows, kUndefined);
  mask.Set(1.0);
  const Index *indexes_ptr = &(indexes[0]);
  BaseFloat *mask_ptr = mask.Data();
  for (int32 r = 0; r < num_rows; r++) {
    if (indexes_ptr[r].t == kNoTime)
      mask_ptr[r] = 0.0;
  }
  CuVector<BaseFloat> cu_mask;
  cu_mask.Swap(&mask);
  matrix->MulRowsVec(cu_mask);
}

// This is a 'dumb' implementation of convolution, created to compare
// with ConvolveForward.
void ConvolveForwardSimple(
    const ConvolutionModel &model,
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    const CuMatrixBase<BaseFloat> &input_cu,
    const CuMatrixBase<BaseFloat> &params_cu,
    CuMatrixBase<BaseFloat> *output_cu) {
  // these loops will be very slow on GPU, so do it all on CPU.
  Matrix<BaseFloat> input(input_cu), params(params_cu),
      output(*output_cu);
  std::unordered_map<Index, int32, IndexHasher> index_to_row;
  int32 input_rows = input.NumRows(),
      output_rows = output.NumRows();
  for (int32 r_in = 0; r_in < input_rows; r_in++) {
    if (input_indexes[r_in].t != kNoTime) {
      index_to_row[input_indexes[r_in]] = r_in;
    }
  }
  int32 num_offsets = model.offsets.size(),
      num_filters_in = model.num_filters_in,
      num_filters_out = model.num_filters_out,
      height_in = model.height_in,
      height_out = model.height_out,
      height_subsample_out = model.height_subsample_out;
  for (int32 r_out = 0; r_out < output_rows; r_out++) {
    Index index_out = output_indexes[r_out];
    if (index_out.t == kNoTime)
      continue;
    SubVector<BaseFloat> output_row(output, r_out);
    for (int32 o = 0; o < num_offsets; o++) {
      int32 time_offset = model.offsets[o].time_offset,
          height_offset = model.offsets[o].height_offset;
      Index index_in(index_out);
      index_in.t += time_offset;
      std::unordered_map<Index, int32, IndexHasher>::const_iterator iter =
          index_to_row.find(index_in);
      if (iter != index_to_row.end()) {
        SubMatrix<BaseFloat> params_part(params, 0, params.NumRows(),
                                         o * num_filters_in, num_filters_in);
        int32 r_in = iter->second;
        SubVector<BaseFloat> input_row(input, r_in);
        for (int32 h_out_subsampled = 0;
             h_out_subsampled < height_out;
             h_out_subsampled++) {
          int32 h_out = h_out_subsampled * height_subsample_out,
              h_in = h_out + height_offset;
          if (h_in < 0 || h_in >= height_in)
            continue;
          SubVector<BaseFloat> output_part(output_row,
                                           h_out_subsampled * num_filters_out,
                                           num_filters_out),
              input_part(input_row, h_in * num_filters_in, num_filters_in);
          output_part.AddMatVec(1.0, params_part, kNoTrans, input_part, 1.0);
        }
      }
    }
  }
  output_cu->CopyFromMat(output);
}



void TestRunningComputation(const ConvolutionModel &conv_model,
                            const std::vector<Index> &input_indexes,
                            const std::vector<Index> &output_indexes,
                            const ConvolutionComputation &computation) {
  CuMatrix<BaseFloat> input(input_indexes.size(), conv_model.InputDim(),
                            kSetZero, kStrideEqualNumCols),
      output(output_indexes.size(), conv_model.OutputDim(),
             kSetZero, kStrideEqualNumCols),
      output2(output),
      params(conv_model.ParamRows(), conv_model.ParamCols());
  input.SetRandn();
  params.SetRandn();
  ZeroBlankRows(input_indexes, &input);
  ConvolveForward(computation, input, params, &output);
  ZeroBlankRows(output_indexes, &output);

  ConvolveForwardSimple(conv_model, input_indexes, output_indexes,
                        input, params, &output2);
  KALDI_LOG << "Tested convolution for model: "
            << conv_model.Info();
  if (!output.ApproxEqual(output2, 0.001)) {
    KALDI_LOG << "Output is: " << output;
    KALDI_LOG << "Output2 is: " << output2;
    KALDI_ERR << "Convolution test failure.";
  }
}


void TestDataBackprop(const ConvolutionModel &conv_model,
                      const std::vector<Index> &input_indexes,
                      const std::vector<Index> &output_indexes,
                      const ConvolutionComputation &computation) {
  CuMatrix<BaseFloat>
      input_deriv(input_indexes.size(), conv_model.InputDim(),
                  kSetZero, kStrideEqualNumCols),
      input(input_indexes.size(), conv_model.InputDim(),
            kSetZero, kStrideEqualNumCols),
      output(output_indexes.size(), conv_model.OutputDim(),
             kSetZero, kStrideEqualNumCols),
      output_deriv(output_indexes.size(), conv_model.OutputDim(),
                   kSetZero, kStrideEqualNumCols),
      params(conv_model.ParamRows(), conv_model.ParamCols());

  input.SetRandn();
  params.SetRandn();
  output_deriv.SetRandn();

  ZeroBlankRows(output_indexes, &output_deriv);
  ConvolveBackwardData(computation, params, output_deriv, &input_deriv);
  ZeroBlankRows(input_indexes, &input_deriv);
  ZeroBlankRows(input_indexes, &input);

  // define the objf as TraceMatMat(output_deriv, output, kTrans).
  // we can work it out from the backpropagated data-derivative.
  BaseFloat expected_objf = TraceMatMat(input_deriv, input, kTrans);

  ConvolveForward(computation, input, params, &output);
  ZeroBlankRows(output_indexes, &output);

  BaseFloat observed_objf = TraceMatMat(output, output_deriv, kTrans);

  KALDI_LOG << "Expected objf = " << expected_objf
            << ", observed objf = " << observed_objf;
  if (!ApproxEqual(expected_objf, observed_objf, 0.1) &&
      fabs(expected_objf) < 1.0) {
    KALDI_ERR << "Difference in objf too large.";
  }
}


void TestParamsBackprop(const ConvolutionModel &conv_model,
                        const std::vector<Index> &input_indexes,
                        const std::vector<Index> &output_indexes,
                        const ConvolutionComputation &computation) {
  CuMatrix<BaseFloat>
      input(input_indexes.size(), conv_model.InputDim(),
            kSetZero, kStrideEqualNumCols),
      output(output_indexes.size(), conv_model.OutputDim(),
             kSetZero, kStrideEqualNumCols),
      output_deriv(output_indexes.size(), conv_model.OutputDim(),
                   kSetZero, kStrideEqualNumCols),
      params(conv_model.ParamRows(), conv_model.ParamCols()),
      params_deriv(conv_model.ParamRows(), conv_model.ParamCols());

  input.SetRandn();
  params.SetRandn();
  output_deriv.SetRandn();

  BaseFloat alpha = 0.5 * RandInt(1, 3);

  ZeroBlankRows(output_indexes, &output_deriv);
  ZeroBlankRows(input_indexes, &input);

  ConvolveBackwardParams(computation, input, output_deriv, alpha,
                         &params_deriv);

  BaseFloat expected_objf = TraceMatMat(params_deriv, params, kTrans) / alpha;

  ConvolveForward(computation, input, params, &output);

  ZeroBlankRows(output_indexes, &output);

  BaseFloat observed_objf = TraceMatMat(output, output_deriv, kTrans);

  KALDI_LOG << "Expected objf = " << expected_objf
            << ", observed objf = " << observed_objf;
  if (!ApproxEqual(expected_objf, observed_objf, 0.1) &&
      fabs(expected_objf) < 1.0) {
    KALDI_ERR << "Difference in objf too large.";
  }
}



void UnitTestTimeHeightConvolutionCompile() {
  for (int32 i = 0; i < 10; i++) {
    KALDI_LOG << "iter = " << i;
    // Create a ConvolutionModel
    ConvolutionModel conv_model;
    GetRandomConvolutionModel(&conv_model);
    std::vector<Index> input_indexes, output_indexes;
    GetRandomConvolutionIndexes(conv_model, &input_indexes, &output_indexes);

    ConvolutionComputationOptions opts;
    ConvolutionComputation computation;
    std::vector<Index> input_indexes_modified, output_indexes_modified;
    CompileConvolutionComputation(conv_model, input_indexes, output_indexes,
                                  opts, &computation,
                                  &input_indexes_modified,
                                  &output_indexes_modified);
    TestComputationIo(computation);
    TestRunningComputation(conv_model,
                           input_indexes_modified,
                           output_indexes_modified,
                           computation);
    TestDataBackprop(conv_model,
                     input_indexes_modified,
                     output_indexes_modified,
                     computation);
    TestParamsBackprop(conv_model,
                       input_indexes_modified,
                       output_indexes_modified,
                       computation);
    std::ostringstream os;
    os << "\nInput-indexes: ";
    WriteIndexVector(os, false, input_indexes);
    os << "\nInput-indexes-modified: ";
    WriteIndexVector(os, false, input_indexes_modified);
    os << "\nOutput-indexes: ";
    WriteIndexVector(os, false, output_indexes);
    os << "\nOutput-indexes-modified: ";
    WriteIndexVector(os, false, output_indexes_modified);
    KALDI_LOG << os.str();
  }
}


void UnitTestTimeHeightConvolution() {
  UnitTestTimeHeightConvolutionIo();
  UnitTestTimeHeightConvolutionCompile();
}

*/

} // namespace cudnn_convolution
} // namespace nnet3
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  using namespace kaldi::nnet3::cudnn_convolution;


  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // -2 .. automatic selection
#endif
    TestConvolutionComputationConfig();
    TestConvolutionComputation();
  }
}
