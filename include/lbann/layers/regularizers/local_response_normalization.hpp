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
//
// lbann_layer_local_response_normalization .hpp .cpp - LRN layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_LOCAL_RESPONSE_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYER_LOCAL_RESPONSE_NORMALIZATION_HPP_INCLUDED

#include <vector>
#include "lbann/base.hpp"
#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/// Local Response Normalization layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class local_response_normalization_layer : public regularizer_layer {
 private:

  /// Normalization window width
  int m_window_width;
  /// LRN alpha scaling parameter
  DataType m_lrn_alpha;
  /// LRN beta power parameter
  DataType m_lrn_beta;
  /// LRN k parameter
  DataType m_lrn_k;

#ifdef __LIB_CUDNN
  /// Pooling descriptor
  cudnnLRNDescriptor_t m_lrn_desc;
#endif // __LIB_CUDNN

 public:
  local_response_normalization_layer
  (int index,
   lbann_comm *comm,
   int window_width,
   DataType lrn_alpha,
   DataType lrn_beta,
   DataType lrn_k,
   cudnn::cudnn_manager *cudnn = NULL)
    : regularizer_layer(index, comm),
      m_window_width(window_width), m_lrn_alpha(lrn_alpha), m_lrn_beta(lrn_beta),
      m_lrn_k(lrn_k) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "local_response_normalization only supports DATA_PARALLEL");
    // Setup the data distribution
    initialize_distributed_matrices();

  #ifdef __LIB_CUDNN

    // Initialize cuDNN objects
    m_lrn_desc = NULL;

    // Initialize GPU memory if using GPU
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN
  }

  local_response_normalization_layer(
    const local_response_normalization_layer&) = default;
  local_response_normalization_layer& operator=(
    const local_response_normalization_layer&) = default;

  ~local_response_normalization_layer() {
  #ifdef __LIB_CUDNN
    // Destroy cuDNN objects
    if(m_lrn_desc) {
      CHECK_CUDNN(cudnnDestroyLRNDescriptor(m_lrn_desc));
    }
  #endif // __LIB_CUDNN
  }

  local_response_normalization_layer* copy() const {
    return new local_response_normalization_layer(*this);
  }

  std::string get_name() const { return "local response normalization"; }

  virtual inline void initialize_distributed_matrices() {
    regularizer_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  /// Initialize GPU objects
  void setup_gpu() {
    regularizer_layer::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
  #else

    // Initialize local response normalization descriptor
    CHECK_CUDNN(cudnnCreateLRNDescriptor(&m_lrn_desc));
    CHECK_CUDNN(cudnnSetLRNDescriptor(m_lrn_desc,
                                      (unsigned int) m_window_width,
                                      (double) m_lrn_alpha,
                                      (double) m_lrn_beta,
                                      (double) m_lrn_k));

  #endif // #ifndef __LIB_CUDNN
  }

  void fp_compute() {
    if(this->m_using_gpus) {
      fp_compute_cudnn();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
      bp_compute_cudnn();
    } else {
      bp_compute_cpu();
    }
  }

 private:
  /// GPU implementation of forward propagation
  void fp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Perform local response normalization with each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnLRNCrossChannelForward(this->m_cudnn->get_handle(i),
                                              m_lrn_desc,
                                              CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                              &one,
                                              this->m_prev_neurons_cudnn_desc,
                                              this->m_prev_activations_d[i],
                                              &zero,
                                              this->m_neurons_cudnn_desc,
                                              this->m_activations_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

  /// GPU implementation of backward propagation
  void bp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Get number of GPUs
    const int num_gpus = this->m_cudnn->get_num_gpus();

    // Perform back propagation on each GPU
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnLRNCrossChannelBackward(this->m_cudnn->get_handle(i),
                                               m_lrn_desc,
                                               CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                               &one,
                                               this->m_neurons_cudnn_desc,
                                               this->m_activations_d[i],
                                               this->m_neurons_cudnn_desc,
                                               this->m_prev_error_signal_d[i],
                                               this->m_prev_neurons_cudnn_desc,
                                               this->m_prev_activations_d[i],
                                               &zero,
                                               this->m_prev_neurons_cudnn_desc,
                                               this->m_error_signal_d[i]));
    }

  #endif // #ifndef __LIB_CUDNN
  }

  /// CPU implementation of forward propagation
  void fp_compute_cpu() {

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    Mat& activations_local = this->m_activations_v->Matrix();

    // Input and output entries are divided amongst channels
    const int num_channels = this->m_neuron_dims[0];
    const int num_per_channel = this->m_num_neurons / num_channels;

    ////////////////////////////////////////////////////////////////
    // activations(i) = prev_activations(i) / scale_factor(i) ^ beta
    // scale_factor(i)
    //   = k + alpha / window_width * sum( prev_activations(j) ^ 2 )
    // Note: The sum is over entries in the normalization window.
    ////////////////////////////////////////////////////////////////

    // Iterate through data samples in mini-batch
    #pragma omp parallel for collapse(2)
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {
      // Iterate through positions in sample
      for(int pos = 0; pos < num_per_channel; ++pos) {

        // Initialize normalization window
        int window_start = - m_window_width / 2;
        int window_end = m_window_width / 2;
        DataType window_sum = 0;
        for(int c = std::max(window_start, 0);
            c <= std::min(window_end, num_channels-1);
            ++c) {
          const DataType x
            = prev_activations_local(pos + num_per_channel*c, sample);
          window_sum += x * x;
        }

        // Iterate through channels at current position
        for(int channel = 0; channel < num_channels; ++channel) {
          const int index = pos + num_per_channel * channel;

          // Apply local response normalization to current entry
          const DataType input_entry = prev_activations_local.Get(index, sample);
          const DataType scale_factor = m_lrn_k + m_lrn_alpha / m_window_width * window_sum;
          const DataType output_entry = input_entry *
            std::pow(scale_factor, -m_lrn_beta);
          activations_local(index, sample) = output_entry;

          // Shift normalization window by one entry
          if(window_start >= 0) {
            const int i = pos + num_per_channel*window_start;
            const DataType x = prev_activations_local(i, sample);
            window_sum -= x * x;
          }
          ++window_start;
          ++window_end;
          if(window_end < num_channels) {
            const int i = pos + num_per_channel*window_end;
            const DataType x = prev_activations_local(i, sample);
            window_sum += x * x;
          }

        }

      }

    }

  }

  /// CPU implementation of backward propagation
  void bp_compute_cpu() {

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& activations_local = this->m_activations_v->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal_v->LockedMatrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();

    // Initialize error signal to zero
    Zero(error_signal_local);

    // Input and output entries are divided amongst channels
    const int num_channels = this->m_neuron_dims[0];
    const int num_per_channel = this->m_num_neurons / num_channels;

    ////////////////////////////////////////////////////////////////
    // error_signal(i)
    //   = prev_error_signal(i) / scale_factor(i) ^ beta
    //     - 2 * alpha * beta / window_width * prev_activations(i)
    //       * sum( prev_error_signal(j) * activations(j)
    //              / scale_factor(j) )
    // Note: See comments in fp_linearity_cpu for a definition of
    //   scale_factor. The sum is over entries in the normalization
    //   window.
    ////////////////////////////////////////////////////////////////

    // Iterate through data samples in mini-batch
    #pragma omp parallel for collapse(2)
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {
      // Iterate through positions in sample
      for(int pos = 0; pos < num_per_channel; ++pos) {

        // Initialize normalization window
        int window_start = - m_window_width / 2;
        int window_end = m_window_width / 2;
        DataType window_sum = 0;
        for(int c = std::max(window_start, 0);
            c <= std::min(window_end, num_channels-1);
            ++c) {
          const DataType x
            = prev_activations_local.Get(pos + num_per_channel*c, sample);
          window_sum += x * x;
        }

        // Iterate through channels at current position
        DataType error_signal_update;
        for(int channel = 0; channel < num_channels; ++channel) {
          const int index = pos + num_per_channel * channel;

          // Get data for current entry
          const DataType activations_entry = activations_local.Get(index, sample);
          const DataType prev_error_signal_entry = prev_error_signal_local.Get(index, sample);
          const DataType scale_factor = m_lrn_k + m_lrn_alpha / m_window_width * window_sum;

          // Update current error signal entry
          error_signal_update = prev_error_signal_entry *
            std::pow(scale_factor, -m_lrn_beta);
          error_signal_local.Update(index, sample, error_signal_update);

          // Update error signal entries in normalization window
          for(int c = std::max(window_start, 0);
              c <= std::min(window_end, num_channels-1);
              ++c) {
            const int i = pos + num_per_channel * c;
            const DataType prev_activations_entry = prev_activations_local.Get(i, sample);
            error_signal_update
              = (-2 * m_lrn_alpha * m_lrn_beta / m_window_width * prev_activations_entry
                 * prev_error_signal_entry * activations_entry / scale_factor);
            error_signal_local.Update(i, sample, error_signal_update);
          }

          // Shift normalization window by one entry
          if(window_start >= 0) {
            const int i = pos + num_per_channel*window_start;
            const DataType x = prev_activations_local.Get(i, sample);
            window_sum -= x * x;
          }
          ++window_start;
          ++window_end;
          if(window_end < num_channels) {
            const int i = pos + num_per_channel*window_end;
            const DataType x = prev_activations_local.Get(i, sample);
            window_sum += x * x;
          }

        }

      }

    }

  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_POOLING_HPP_INCLUDED
