// rnnlm/rnnlm-utils-test.cc

#include "rnnlm/rnnlm-utils.h"
#include "arpa-sampling.h"

#include <math.h>
#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

namespace kaldi {
namespace rnnlm {

void PrepareVector(int n, int ones_size, std::set<int>* must_sample_set,
                   vector<BaseFloat>* selection_probs) {
  BaseFloat prob = 0;
  BaseFloat prob_sum = 0;
  for (int i = 0; i < n; i++) {
    prob = RandUniform();
    prob_sum += prob;
    (*selection_probs).push_back(prob);
  }
  for (int i = 0; i < n; i++) {
    (*selection_probs)[i] /= prob_sum;
  }
  for (int i = 0; i < ones_size; i++) {
    (*must_sample_set).insert(rand() % n);
  }
}

void UnitTestNChooseKSamplingConvergence(int n, int k, int ones_size) {
  std::set<int> must_sample_set;
  vector<BaseFloat> selection_probs;
  PrepareVector(n, ones_size, &must_sample_set, &selection_probs);
  NormalizeVec(k, must_sample_set, &selection_probs);

  vector<std::pair<int, BaseFloat> > u(selection_probs.size());
  for (int i = 0; i < u.size(); i++) {
    u[i].first = i;
    u[i].second = selection_probs[i];
  }
  // normalize the selection_probs
  BaseFloat sum = 0;
  for (int i = 0; i < u.size(); i++) {
    sum += std::min(BaseFloat(1.0), selection_probs[i]);
  }
  KALDI_ASSERT(ApproxEqual(sum, k));
  for (int i = 0; i < u.size(); i++) {
    selection_probs[i] = std::min(BaseFloat(1.0), selection_probs[i]) / sum;
  }

  vector<BaseFloat> samples_counts(u.size(), 0);
  int count = 0;
  for (int i = 0; ; i++) {
    count++;
    vector<int> samples;
    SampleWithoutReplacement(u, k, &samples);
    for (int j = 0; j < samples.size(); j++) {
      samples_counts[samples[j]] += 1;
    }
    // update Euclidean distance between the two pdfs every 1000 iters
    if (count % 1000 == 0) {
      BaseFloat distance = 0;
      vector<BaseFloat> samples_probs(u.size());
      for (int j = 0; j < samples_probs.size(); j++) {
        samples_probs[j] = samples_counts[j] / (count * k);
      }
      for (int j = 0; j < u.size(); j++) {
        distance += pow(samples_probs[j] - selection_probs[j], 2);
      }
      distance = sqrt(distance);

      KALDI_LOG << "distance after " << count << " runs is " << distance;

      if (distance < 0.005) {
        KALDI_LOG << "Sampling convergence test: passed for sampling " << k <<
          " items from " << n << " unigrams";
        break;
      }
    }
    // if the Euclidean distance is small enough, break the loop
  }
}

void UnitTestSamplingConvergence() {
  // number of unigrams
  int n = rand() % 10000 + 100;
  // sample size
  int k;
  // number of ones
  int ones_size;
  ones_size = rand() % (n / 2);
  k = rand() % (n - ones_size) + ones_size + 1;
  UnitTestNChooseKSamplingConvergence(n, k, ones_size);
  // test when k = 1
  UnitTestNChooseKSamplingConvergence(n, 1, 0);
  // test when k = 2
  UnitTestNChooseKSamplingConvergence(n, 2, rand() % 1);
  // test when k = n
  ones_size = rand() % (n / 2);
  UnitTestNChooseKSamplingConvergence(n, n, ones_size);
}

// test that probabilities 1.0 are always sampled
void UnitTestSampleWithProbOne(int iters) {
  // number of unigrams
  int n = rand() % 1000 + 100;
  // generate a must_sample_set with ones
  int ones_size = rand() % (n / 2);
  std::set<int> must_sample_set;
  vector<BaseFloat> selection_probs;

  PrepareVector(n, ones_size, &must_sample_set, &selection_probs);

  // generate a random number k from ones_size + 1 to n
  int k = rand() % (n - ones_size) + ones_size + 1;
  NormalizeVec(k, must_sample_set, &selection_probs);

  vector<std::pair<int, BaseFloat> > u(selection_probs.size());
  for (int i = 0; i < u.size(); i++) {
    u[i].first = i;
    u[i].second = selection_probs[i];
  }

  int N = iters;
  for (int i = 0; i < N; i++) {
    vector<int> samples;
    SampleWithoutReplacement(u, k, &samples);
    if (must_sample_set.size() > 0) {
      // assert every item in must_sample_set is sampled
      for (set<int>::iterator it = must_sample_set.begin(); it != must_sample_set.end(); ++it) {
        KALDI_ASSERT(std::find(samples.begin(), samples.end(), *it) !=
            samples.end());
      }
    }
  }
  KALDI_LOG << "Test sampling with prob = 1.0 successful";
}

void UnitTestSamplingTime(int iters) {
  // number of unigrams
  int n = rand() % 1000 + 100;
  // generate a must_sample_set with ones
  int ones_size = rand() % (n / 2);
  std::set<int> must_sample_set;
  vector<BaseFloat> selection_probs;

  PrepareVector(n, ones_size, &must_sample_set, &selection_probs);

  // generate a random number k from ones_size + 1 to n
  int k = rand() % (n - ones_size) + ones_size + 1;
  NormalizeVec(k, must_sample_set, &selection_probs);

  vector<std::pair<int, BaseFloat> > u(selection_probs.size());
  for (int i = 0; i < u.size(); i++) {
    u[i].first = i;
    u[i].second = selection_probs[i];
  }

  int N = iters;
  Timer t;
  t.Reset();
  BaseFloat total_time;
  for (int i = 0; i < N; i++) {
    vector<int> samples;
    SampleWithoutReplacement(u, k, &samples);
  }
  total_time = t.Elapsed();
  KALDI_LOG << "Time test: Sampling " << k << " items from " << n <<
    " unigrams for " << N << " times takes " << total_time << " totally.";
}

}  // end namespace rnnlm
}  // end namespace kaldi.

int main(int argc, char **argv) {
  using namespace kaldi;
  using namespace rnnlm;
  int N = 10000;
  UnitTestSampleWithProbOne(N);
  UnitTestSamplingTime(N);
  UnitTestSamplingConvergence();

  const char *usage = "";
  ParseOptions po(usage);
  po.Read(argc, argv);
  std::string arpa_file = po.GetArg(1), history_file = po.GetArg(2);
  
  ArpaParseOptions options;
  fst::SymbolTable symbols;
  // Use spaces on special symbols, so we rather fail than read them by mistake.
  symbols.AddSymbol(" <eps>", kEps);
  // symbols.AddSymbol(" #0", kDisambig);
  options.bos_symbol = symbols.AddSymbol("<s>", kBos);
  options.eos_symbol = symbols.AddSymbol("</s>", kEos);
  options.unk_symbol = symbols.AddSymbol("<unk>", kUnk);
  options.oov_handling = ArpaParseOptions::kAddToSymbols;
  ArpaSampling mdl(options, &symbols);
  
  bool binary;
  Input k1(arpa_file, &binary);
  mdl.Read(k1.Stream(), binary);
  mdl.TestReadingModel();
   
  Input k2(history_file, &binary);
  std::vector<HistType> histories;
  histories = mdl.ReadHistories(k2.Stream(), binary);
  unordered_map<int32, BaseFloat> pdf_hist_weight;
  mdl.ComputeOutputWords(histories, &pdf_hist_weight);
  // command for running the test binary: ./test-binary arpa-file history-file
  // arpa-file is the ARPA-format language model
  // history-file has lines of histories, one history per line

  // this test can be slow
  /*
  KALDI_LOG << "Start weighted histories test...";
  for (int i = 0; i < N / 100; i++) {
    mdl.TestPdfsEqual(); 
  }
  KALDI_LOG << "Successfuly pass the test.";
  */
  return 0;
}
