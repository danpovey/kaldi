// nnet3bin/nnet3-copy-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2016  Pegah Ghahremani

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
#include "hmm/transition-model.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Combine examples for neural network training and supports multiple rspecifiers, in which case it will reads the inputs \n"
        "round-robin and writes to the output"
        "\n"
        "Usage:  nnet3-fold-egs [options] <egs-rspecifier1> [<egs-rspecifier2> ...] <egs-wspecifier>\n"
        "\n"
        "e.g.\n"
        "nnet3-fold-egs ark:1.egs ark:2.egs ark,t:text.egs\n"
        "or:\n"
        "nnet3-fold-egs ark:train.egs ark:1.egs ark:2.egs\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    //
    int32 num_inputs = po.NumArgs() - 1;
    std::vector<SequentialNnetExampleReader*> example_readers(num_inputs);
    for (int32 i = 0; i < num_inputs; i++)
      example_readers[i] = new SequentialNnetExampleReader(po.GetArg(i+1));

    std::string examples_wspecifier(po.GetArg(num_inputs+1));
    NnetExampleWriter example_writer(examples_wspecifier);
    int64 num_written = 0;
    std::vector<int64> num_read(num_inputs);
    
    //for (; !example_readers[0]->Done(); tot_num_read++) {
    while (!example_readers[0]->Done()) {
      for (int32 reader = 0; reader < num_inputs; reader++) { 
        if (!example_readers[reader]->Done()) {
          example_readers[reader]->Next();
          num_read[reader]++;
          std::string key = example_readers[reader]->Key();
          const NnetExample &eg = example_readers[reader]->Value();
          example_writer.Write(key, eg);
          num_written++;
        }
      }
    }
    for (int32 i = 0; i < num_inputs; i++)
      delete example_readers[i];
    
    KALDI_LOG << "Read " << num_read[0] << "neural-network training examples "
              << "from " << num_inputs << " inputs, wrote "
              << num_written; 

    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
