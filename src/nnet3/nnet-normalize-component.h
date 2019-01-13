// nnet3/nnet-normalize-component.h

// Copyright 2011-2013  Karel Vesely
//           2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang
//           2014-2015  Vijayaditya Peddinti
//           2014-2015  Guoguo Chen
//                2015  Daniel Galvez
//                2015  Tom Ko

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

#ifndef KALDI_NNET3_NNET_NORMALIZE_COMPONENT_H_
#define KALDI_NNET3_NNET_NORMALIZE_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  nnet-normalize-component.h
///
///   This file contains declarations of components that in one way or
///   another normalize their input: NormalizeComponent and BatchNormComponent.

/*
   NormalizeComponent implements the function:

         y = x * (sqrt(dim(x)) * target-rms) / |x|

   where |x| is the 2-norm of the vector x.  I.e. its output is its input
   scaled such that the root-mean-square values of its elements equals
   target-rms.  (As a special case, if the input is zero, it outputs zero).
   This is like Hinton's layer-norm, except not normalizing the mean, only
   the variance.


    Note: if you specify add-log-stddev=true, it adds an extra element to
     y which equals log(|x| / sqrt(dim(x))).


   Configuration values accepted:
      dim, or input-dim    Input dimension of this component, e.g. 1024.
                           Will be the same as the output dimension if add-log-stddev=false.
      block-dim            Defaults to 'dim' you may specify a divisor
                           of 'dim'.  In this case the input dimension will
                           be interpreted as blocks of dimension 'block-dim'
                           to which the nonlinearity described above is applied
                           separately.
      add-log-stddev       You can set this to true to add an extra output
                           dimension which will equal |x| / sqrt(dim(x)).
                           If block-dim is specified, this is done per block.
      target-rms           This defaults to 1.0, but if set it to another
                           (nonzero) value, the output will be scaled by this
                           factor.
 */
class NormalizeComponent: public Component {
 public:
  explicit NormalizeComponent(const NormalizeComponent &other);

  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsInput|kBackpropAdds|
        (add_log_stddev_ ? 0 : kPropagateInPlace|kBackpropInPlace) |
        (block_dim_ != input_dim_ ? kInputContiguous|kOutputContiguous : 0);
  }
  NormalizeComponent() { }
  virtual std::string Type() const { return "NormalizeComponent"; }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual Component* Copy() const { return new NormalizeComponent(*this); }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                          const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const {
    return (input_dim_ + (add_log_stddev_ ? (input_dim_ / block_dim_) : 0));
  }
  virtual std::string Info() const;
 private:
  NormalizeComponent &operator = (const NormalizeComponent &other); // Disallow.
  enum { kExpSquaredNormFloor = -66 };
  // kSquaredNormFloor is about 0.7e-20.  We need a value that's exactly representable in
  // float and whose inverse square root is also exactly representable
  // in float (hence, an even power of two).
  static const BaseFloat kSquaredNormFloor;
  int32 input_dim_;
  int32 block_dim_;
  BaseFloat target_rms_; // The target rms for outputs, default 1.0.

  bool add_log_stddev_; // If true, log(max(epsi, sqrt(row_in^T row_in / D)))
                        // is an extra dimension of the output.
};


/*
  BatchNormComponent

  This implements batch normalization; for each dimension of the
  input it normalizes the data to be zero-mean, unit-variance.  You
  can set the block-dim configuration value to implement spatial
  batch normalization, see the comment for the variable.

  If you want to combine this with the trainable offset and scale that the
  original BatchNorm paper used, then follow this by the
  ScaleAndOffsetComponent.

  It's a simple component (uses the kSimpleComponent flag), but it is unusual in
  that it will give different results if you call it on half the matrix at a
  time.  Most of the time this would be pretty harmless, so we still return the
  kSimpleComponent flag.  We may have to modify the test code a little to
  account for this, or possibly remove the kSimpleComponent flag.  In some sense
  each output Index depends on every input Index, but putting those dependencies
  explicitly into the dependency-tracking framework as a GeneralComponent
  would be very impractical and might lead to a lot of unnecessary things being
  computed.  You have to be a bit careful where you put this component, and understand
  what you're doing e.g. putting it in the path of a recurrence is a bit problematic
  if the minibatch size is small.

    Accepted configuration values:
           dim          Dimension of the input and output
           block-dim    Defaults to 'dim', but may be set to a divisor
                        of 'dim'.  In this case, each block of dimension 'block-dim'
                        is treated like a separate row of the input matrix, which
                        means that the stats from n'th element of each
                        block are pooled into one class, for each n.
           epsilon      Small term added to the variance that is used to prevent
                        division by zero.  Defaults to 0.001.
           target-rms   This defaults to 1.0, but if set, for instance, to 2.0,
                        it will normalize the standard deviation of the output to
                        2.0. 'target-stddev' might be a more suitable name, but this
                        was chosen for consistency with NormalizeComponent.
 */
class BatchNormComponent: public Component {
 public:

  BatchNormComponent() { }

  // call this with 'true' to set 'test mode' where the batch normalization is
  // done with stored stats.  There won't normally be any need to specially
  // accumulate these stats; they are stored as a matter of course on each
  // iteration of training, as for NonlinearComponents, and we'll use the stats
  // from the most recent [script-level] iteration.
  // (Note: it will refuse to actually set test-mode to true if there
  // are no stats stored.)
  void SetTestMode(bool test_mode);

  // constructor using another component
  BatchNormComponent(const BatchNormComponent &other);

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "BatchNormComponent"; }
  virtual int32 Properties() const {
    // If the block-dim is less than the dim, we need the input and output
    // matrices to be contiguous (stride==num-cols), as we'll be reshaping
    // internally.  This is not much of a cost, because this will be used
    // in convnets where we have to do this anyway.
    return kSimpleComponent|kBackpropNeedsOutput|kPropagateInPlace|
        kBackpropInPlace|
        (block_dim_ < dim_ ? kInputContiguous|kOutputContiguous : 0)|
        (test_mode_ ? 0 : kUsesMemo|kStoresStats);
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const { return new BatchNormComponent(*this); }

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void ZeroStats();

  virtual void DeleteMemo(void *memo) const { delete static_cast<Memo*>(memo); }

  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);

  // Members specific to this component type.
  // Note: the offset and scale will only be nonempty in 'test mode'.
  const CuVector<BaseFloat> &Offset() const { return offset_; }
  const CuVector<BaseFloat> &Scale() const { return scale_; }

 private:

  struct Memo {
    // number of frames (after any reshaping).
    int32 num_frames;
    // 'sum_sumsq_scale' is of dimension 5 by block_dim_:
    // Row 0 = mean = the mean of the rows of the input
    // Row 1 = uvar = the uncentered variance of the input (= sumsq / num_frames).
    // Row 2 = scale = the scale of the renormalization.
    // Rows 3 and 4 are used as temporaries in Backprop.
    CuMatrix<BaseFloat> mean_uvar_scale;
  };

  void Check() const;

  // this function is used in a couple of places; it turns the raw stats into
  // the offset/scale term of a normalizing transform.
  static void ComputeOffsetAndScale(double count,
                                    BaseFloat epsilon,
                                    const Vector<double> &stats_sum,
                                    const Vector<double> &stats_sumsq,
                                    Vector<BaseFloat> *offset,
                                    Vector<BaseFloat> *scale);
  // computes derived parameters offset_ and scale_.
  void ComputeDerived();

  // Dimension of the input and output.
  int32 dim_;
  // This would normally be the same as dim_, but if it's less (and it must be >
  // 0 and must divide dim_), then each separate block of the input of dimension
  // 'block_dim_' is treated like a separate frame for the purposes of
  // normalization.  This can be used to implement spatial batch normalization
  // for convolutional setups-- assuming the filter-dim has stride 1, which it
  // always will in the new code in nnet-convolutional-component.h.
  int32 block_dim_;

  // Used to avoid exact-zero variances, epsilon has the dimension of a
  // covariance.
  BaseFloat epsilon_;

  // This value will normally be 1.0, which is the default, but you can set it
  // to other values as a way to control how fast the following layer learns
  // (smaller -> slower).  The same config exists in NormalizeComponent.
  BaseFloat target_rms_;

  // This is true if we want the batch normalization to operate in 'test mode'
  // meaning the data mean and stddev used for the normalization are fixed
  // quantities based on previously accumulated stats.  Note: the stats we use
  // for this are based on the same 'StoreStats' mechanism as we use for
  // components like SigmoidComponent and ReluComponent; we'll be using
  // the stats from the most recent [script-level] iteration of training.
  bool test_mode_;


  // total count of stats stored by StoreStats().
  double count_;
  // sum-of-data component of stats of input data.
  CuVector<double> stats_sum_;
  // sum-of-squared component of stats of input data.
  CuVector<double> stats_sumsq_;

  // offset_ and scale_ are derived from stats_sum_ and stats_sumsq_; they
  // dictate the transform that is done in 'test mode'.  They are set only when
  // reading the model from disk and when calling SetTestMode(true); they are
  // resized to empty when the stats are updated, to ensure that out-of-date
  // values are not kept around.
  CuVector<BaseFloat> offset_;
  CuVector<BaseFloat> scale_;
};



/*
  MeanNormComponent

  This component is intended to be used just before VarNormComponent, as a
  substitute for BatchBormComponent that is suitable when different minibatches
  might have different data distributions (e.g., might come from different
  languages).  So together they perform the same function as Batch
  Renormalization.

  It adds an offset to its input that ensures that (averaged over time), the
  input has zero mean.  It also, optionally, mean-normalizes the backpropagated
  derivatives per minibatch, which might help stabilize training (although this
  is kind of non-ideal from a theoretical convergence perspective; it would be
  better to do some averaging over time, but this might interact in a nontrivial
  way with max-change and the like).

  What this component does can be summarized as follows; we'll write this for
  a single dimension.  Assume the current minibatch size is n.  Then this
  component does:

        count += n
            m += \sum_{i=1}^n  x_i

  and in the forward propagation, it does:

            y_i := x_i - (m / count)

  In the backprop, it does:

         \hat{x}_i := \hat{y}_i  -  backprop_normalize_scale * (\sum_i \hat{y}_i) / n

  and the user can set backprop_normalize_scale to a value in the range [0, 1].

  In test mode it doesn't update the stats.

  You'd actually want to have 'count', and 'm', decaying over time, but that's
  done as a call to this class, by calling Scale(), which scales these stats
  (probably down).  You can call ScaleBatchnormStats() to have this called.

    Accepted configuration values:
           dim          Dimension of the input and output
           block-dim    Defaults to 'dim', but may be set to a divisor
                        of 'dim'.  In this case, each block of dimension 'block-dim'
                        is treated like a separate row of the input matrix, which
                        means that the stats from n'th element of each
                        block are pooled into one class, for each n.
   backprop-normalize-scale   Scaling value between 0 and 1 (default: 0)
                        which affects the behavior in backprop.
 */
class MeanNormComponent: public Component {
 public:

  MeanNormComponent() { }

  // call this with 'true' to set 'test mode', which will turn off accumulation
  // of statistics.
  void SetTestMode(bool test_mode);

  // constructor using another component
  MeanNormComponent(const MeanNormComponent &other);

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "MeanNormComponent"; }
  virtual int32 Properties() const {
    // If the block-dim is less than the dim, we need the input and output
    // matrices to be contiguous (stride==num-cols), as we'll be reshaping
    // internally.
    return kSimpleComponent|kPropagateInPlace|kBackpropInPlace|
        (block_dim_ < dim_ ? kInputContiguous|kOutputContiguous : 0)|
        (test_mode_ ? 0 : kUsesMemo|kStoresStats);
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const { return new MeanNormComponent(*this); }

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  // Note: there is no ZeroStats() function-- we use the base-class one that
  // does nothing-- since zeroing the stats at the beginning of each training
  // jobs could adversely affect the performance of this method.  (The stats
  // stored here are not purely diagnostic like they are for most components
  // that store stats).

  virtual void DeleteMemo(void *memo) const { delete static_cast<Memo*>(memo); }

  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);

  // Members specific to this component type.
  const CuVector<BaseFloat> &Offset() const { return offset_; }

 private:

  struct Memo {
    // number of frames (after any reshaping).
    int32 num_frames;
    // Row 0 = sum = the sum of the rows of the input
    // Row 1 is the offset we'll be using for this minibatch.
    // Row 2 is a temporary that we use in backprop.
    CuMatrix<BaseFloat> sum_offset_temp;
  };

  void Check() const;

  // Dimension of the input and output.
  int32 dim_;

  // block_dim_ would normally be the same as dim_, but if it's less (and it
  // must be > 0 and must divide dim_), then each separate block of the input of
  // dimension 'block_dim_' is treated like a separate frame for the purposes of
  // normalization.  This can be used to implement spatial batch normalization
  // for convolutional setups-- assuming the filter-dim has stride 1, which it
  // always will in the new code in nnet-convolutional-component.h.
  int32 block_dim_;


  // Value in the range [0.0, 1.0] that affects how the backprop works;
  // see comment above the class definition for explanation.
  BaseFloat backprop_normalize_scale_;

  // If test mode is set this component doesn't store any further stats, but
  // uses the stats that were previously stored.
  bool test_mode_;

  // total count of stats stored by StoreStats().
  double count_;
  // sum of the input data.
  CuVector<BaseFloat> stats_sum_;

  // offset_  is derived from stats_sum_ and count_; it's updated every
  // time stats_sum_ and count_ are updated (e.g. when StoreStats() is called).
  CuVector<BaseFloat> offset_;
};


/*
  VarNormComponent

  This component is intended to be used just after MeanNormComponent, as a
  substitute for BatchBormComponent that is suitable when different minibatches
  might have different data distributions (e.g., might come from different
  languages).  So together they perform the same function as Batch
  Renormalization.

  This component scales its input per dimension, where the scale is computed,
  based on moving average stats over multiple minibatches.  It also, per
  minibatch, modifies the backpropagated statistics so as to reflect invariance
  w.r.t. a scaling factor.  This is similar to Batch Renormalization.

  What is does can be summarized as follows; we'll write this for a single
  dimension.  Assume the current minibatch size is n.  Then this component does:

        count += n
            s += \sum_{i=1}^n  x_i^2
        scale := (epsilon + s / count)^{-0.5}

  and in the forward propagation, it does:

            y_i := x_i * scale

  In the backprop, it does (note: this is what we do in principle, but
  wee the actual formula we use in practice further below):

             r :=  scale * (\sum_{i=1}^n \hat{y}_i y_i) / (\sum_{i=1}^n y_i^2)

   then we do:
      \hat{x}_i := scale * \hat{y}_i - r * y_i

   The formula above for r is the one that ensures that:
          \sum_i \hat{x}_i x_i  = 0
   which is the condition where the output is insensitive to a scaling of the
   input.  This is kind of an approximation to what we really want; it would
   be more ideal to accumulate a version of 'r' that was an average over many
   minibatches, and use that.  We can have an option average_r to
   allow time-averaging of r.

   We can derive the formula for r as follows.  First replace x_i with y_i,
    since the expression differs only by a positive scalar 'scale', and y_i
    ends up being more convenient:
          \sum_i \hat{x}_i y_i  = 0

          \sum_i (scale * \hat{y}_i - r * y_i) * y_i = 0
          \sum_i scale * \hat{y}_i * y_i - r * y_i^2 = 0
          \sum_i scale * \hat{y}_i * y_i  = r * \sum_i y_i^2
           r :=   scale * (\sum_i \hat{y}_i * y_i) / (\sum_i y_i^2)
              \simeq  scale * (\sum_i \hat{y}_i * y_i) / n

   Note: to simplify the formula for r, we make the approximation that
   (\sum_{i=1}^n y_i^2) == n, which it is, almost, except when the input size is
   comparable to epsilon_ or smaller (and we don't really care about this term
   in that pathological case anyway-- this term's function is to prevent blowup
   of the input, and we *want* blowup in that case).

    so we let r be as follows:
             r :=  scale * (\sum_{i=1}^n \hat{y}_i y_i) / n

  You'd actually want to have 'count', and 's', decaying over time, but that's
  done as a call to this class, by calling Scale(), which scales these stats
  (probably down).  You can call ScaleBatchnormStats() to have this called.

    Accepted configuration values:
           dim          Dimension of the input and output
           block-dim    Defaults to 'dim', but may be set to a divisor
                        of 'dim'.  In this case, each block of dimension 'block-dim'
                        is treated like a separate row of the input matrix, which
                        means that the stats from n'th element of each
                        block are pooled into one class, for each n.
           epsilon      Minimum variance (added to the actual variance).
                        Defaults to 0.001.
          average-r     Boolean, default true.  If true, 'r' (the term that
                        ensures that the input derivatives are insensitive to
                        scaling of the input) is averaged over time, instead of
                        being computed on the current minibatch.  As for the
                        forward stats, the averaging over time (downweighting of
                        old frames) is controlled externally to this class, via
                        the utility function ScaleBatchnormStats(), which calls
                        the Scale() function of this component.
 */
class VarNormComponent: public Component {
 public:

  VarNormComponent() { }

  // call this with 'true' to set 'test mode', which will turn off accumulation
  // of statistics.
  void SetTestMode(bool test_mode);

  // constructor using another component
  VarNormComponent(const VarNormComponent &other);

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "VarNormComponent"; }
  virtual int32 Properties() const {
    // If the block-dim is less than the dim, we need the input and output
    // matrices to be contiguous (stride==num-cols), as we'll be reshaping
    // internally.
    return kSimpleComponent|kPropagateInPlace|kBackpropInPlace|
        kBackpropNeedsOutput|
        (block_dim_ < dim_ ? kInputContiguous|kOutputContiguous : 0)|
        (test_mode_ ? 0 : kUsesMemo|kStoresStats);
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const { return new VarNormComponent(*this); }

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  // Note: there is no ZeroStats() function-- we use the base-class one that
  // does nothing-- since zeroing the stats at the beginning of each training
  // jobs could adversely affect the performance of this method.  (The stats
  // stored here are not purely diagnostic like they are for most components
  // that store stats).

  virtual void DeleteMemo(void *memo) const { delete static_cast<Memo*>(memo); }

  virtual void StoreStats(const CuMatrixBase<BaseFloat> &in_value,
                          const CuMatrixBase<BaseFloat> &out_value,
                          void *memo);

  // Members specific to this component type.
  const CuVector<BaseFloat> &Scale() const { return scale_; }

 private:

  struct Memo {
    // number of frames (after any reshaping).
    int32 num_frames;
    // Row 0 = sumsq = contains \sum_{i=1}^n x_i^2
    // Row 1 = scale   (the scale we used in the forward propagation)
    // Row 2 is used as a temporary in Backprop.
    CuMatrix<BaseFloat> sumsq_scale_temp;
  };

  void Check() const;

  // Dimension of the input and output.
  int32 dim_;

  // block_dim_ would normally be the same as dim_, but if it's less (and it
  // must be > 0 and must divide dim_), then each separate block of the input of
  // dimension 'block_dim_' is treated like a separate frame for the purposes of
  // normalization.  This can be used to implement spatial batch normalization
  // for convolutional setups-- assuming the filter-dim has stride 1, which it
  // always will in the new code in nnet-convolutional-component.h.
  int32 block_dim_;

  // A configuration variable which is like a variance floor (but added, not
  // floored).  Defaults to 0.001.
  BaseFloat epsilon_;

  // True if, in the backprop pass, we use stats averaged over multiple
  // minibatches to compute the term that captures scale invariance of the
  // derivatives w.r.t. the input.  (Note: MeanNormComponent does not have this
  // feature because I don't think that cancelation of the derivative will even
  // be necessary at all in case of the mean)
  bool average_r_;

  // If test mode is set this component doesn't store any further stats, but
  // uses the stats that were previously stored.
  bool test_mode_;

  // total count of stats stored by StoreStats().
  BaseFloat count_;
  // sum of the input data squared, i.e. \sum_i x_i^2
  CuVector<BaseFloat> stats_sumsq_;

  // scale_ is derived from stats_sumsq_ and count_; it's updated every time
  // stats_sumsq_ and count_ are updated (e.g. when StoreStats() is called).
  CuVector<BaseFloat> scale_;

  // The count of stats corresponding to 'r' (a quantity used in
  // the backprop to cancel out the derivative w.r.t. the scaling
  // factor).  Has the dimension of a number of frames, like count_,
  // but we store it separately in case Backprop() is not called,
  // and also as it's computed using a different path that has
  // max-change code involved, so that may make the count differ.
  BaseFloat r_count_;
  // The sum of the 'r', scaled by r_count_; you'd divide by
  // r_count_ to get the actual 'r' value.
  CuVector<BaseFloat> r_sum_;

};



} // namespace nnet3
} // namespace kaldi


#endif
