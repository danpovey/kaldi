// matrix/matrix-functions-test.cc

// Copyright 2018  Johns Hopkins University (author: Daniel Povey)
//           2018  Institute of Acoustics, CAS (Gaofeng Cheng)

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

#include "matrix/matrix-functions.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {
void SvdRescalerTestInit() {
    std::cout << "Init Test1.\n";
    int32 rows = 10, cols = 10;
    Matrix<BaseFloat> mat(rows, cols);
    mat.SetRandn();
    SvdRescaler sc;
    sc.Init(&mat, false);
    SvdRescaler sc2(mat);

    VectorBase<BaseFloat> &vec1 = sc.InputSingularValues();
    VectorBase<BaseFloat> &vec2 = *sc.OutputSingularValues(),
                          &vec3 = *sc.OutputSingularValueDerivs();

    KALDI_ASSERT(vec1.Dim() == vec2.Dim() &&
                 vec2.Dim() == vec3.Dim() &&
                 vec1.Max() == vec2.Max() &&
                 vec2.Max() == vec3.Max() &&
                 vec1.Min() == vec2.Min() &&
                 vec2.Min() == vec3.Min());

    VectorBase<BaseFloat> &vec12 = sc2.InputSingularValues();
    VectorBase<BaseFloat> &vec22 = *sc2.OutputSingularValues(),
                          &vec32 = *sc2.OutputSingularValueDerivs();

    KALDI_ASSERT(vec12.Dim() == vec22.Dim() &&
                 vec22.Dim() == vec32.Dim() &&
                 vec12.Max() == vec22.Max() &&
                 vec22.Max() == vec32.Max() &&
                 vec12.Min() == vec22.Min() &&
                 vec22.Min() == vec32.Min());
}

void SvdRescalerTestWrite() {
    int32 rows = 10, cols = 10;
    Matrix<BaseFloat> mat(rows, cols);
    mat.SetRandn();
    SvdRescaler sc;
    sc.Init(&mat, false);
    SvdRescaler sc2(mat);

    VectorBase<BaseFloat> &vec1 = sc.InputSingularValues();
    VectorBase<BaseFloat> &vec2 = *sc.OutputSingularValues(),
                          &vec3 = *sc.OutputSingularValueDerivs();
    for(int32 i = 0; i < rows; i++)
    {
        KALDI_ASSERT((vec1)(i) == (vec2)(i));
        KALDI_ASSERT((vec2)(i) == (vec3)(i));
    }

    VectorBase<BaseFloat> &vec12 = sc2.InputSingularValues();
    VectorBase<BaseFloat> &vec22 = *sc2.OutputSingularValues(),
                          &vec32 = *sc2.OutputSingularValueDerivs();
    for(int32 i = 0; i < rows; i++)
    {
        KALDI_ASSERT((vec12)(i) == (vec22)(i));
        KALDI_ASSERT((vec22)(i) == (vec32)(i));
    }
}
} // namespace kaldi

int main() {
  kaldi::SvdRescalerTestInit();
  std::cout << "Init Test OK.\n";
  kaldi::SvdRescalerTestWrite();
  std::cout << "Write Test OK.\n";
}