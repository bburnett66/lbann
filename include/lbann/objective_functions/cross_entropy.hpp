////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OBJECTIVE_FUNCTION_CROSS_ENTROPY_HPP_INCLUDED
#define LBANN_OBJECTIVE_FUNCTION_CROSS_ENTROPY_HPP_INCLUDED

#include "lbann/objective_functions/objective_function.hpp"

namespace lbann {

namespace objective_functions {

/** Cross entropy objective function. */
class cross_entropy : public objective_function {

 public:
  /** Default constructor. */
  cross_entropy(bool use_softmax_shortcut = false);
  /** Copy constructor. */
  cross_entropy(const cross_entropy& other) = default;
  /** Copy assignment operator. */
  cross_entropy& operator=(const cross_entropy& other) = default;
  /** Destructor. */
  ~cross_entropy();
  /** Copy function. */
  cross_entropy* copy() const {
    return new cross_entropy(*this);
  }

  /** Compute the cross entropy objective function.
   *  Given a predicted distribution \f$y\f$ and ground truth
   *  distribution \f$\hat{y}\f$, the cross entropy is
   *    \f[
   *    CE(y,\hat{y}) = - \sigma\limits_{i} \hat{y}_i \log y_i
   *    \f]
   *  This function updates the objective function value with the mean
   *  cross entropy across the mini-batch. Note that each column of
   *  the predictions and ground truth matrices should have
   *  non-negative entries that add up to one.
   */
  void compute_value(const AbsDistMat& predictions,
                     const AbsDistMat& ground_truth);

  /** Compute the gradient of the cross entropy objective function.
   *  Given a predicted distribution \f$y\f$ and ground truth
   *  distribution \f$\hat{y}\f$, the gradient of the cross entropy
   *  is
   *    \f[
   *    \nabla_y CE (y,\hat{y}) = - \frac{y} ./ y
   *    \f]
   *  If the softmax shortcut is activated (see description for
   *  m_using_softmax_shortcut), the returned gradient is with respect
   *  to the softmax layer input.
   */
  void compute_gradient(const AbsDistMat& predictions,
                        const AbsDistMat& ground_truth,
                        AbsDistMat& gradient);

  /** Set flag for softmax shortcut.
   *  See description for m_using_softmax_shortcut.
   */
  void set_softmax_shortcut(bool use_shortcut);

  /** Get the name of the objective function. */
  std::string name() const { return "cross entropy"; }

 private:

  /** Whether to use softmax shortcut.
   *  If the penultimate layer is a softmax layer and the ground truth
   *  is categorical, then we can use a mathematical trick. Given a
   *  predicted distribution \f$y\f$ and ground truth distribution
   *  \f$\hat{y}\f$, the gradient of the categorical cross entropy
   *  with respect to the softmax layer input is
   *    \f[
   *      \nabla CE (y,\hat{y}) = y - \hat{y}
   *    \f]
   */
  bool m_using_softmax_shortcut;

};

}  // namespace objective_functions

}  // namespace lbann

#endif  // LBANN_OBJECTIVE_FUNCTION_CROSS_ENTROPY_HPP_INCLUDED