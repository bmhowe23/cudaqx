/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include <cassert>
#include <vector>

namespace cudaq::qec {

/// @brief This is a simple LUT (LookUp Table) decoder that demonstrates how to
/// build a simple decoder that can correctly decode errors during a single bit
/// flip in the block.
class sliding_window : public decoder {
private:
  // Input parameters.
  std::size_t window_size = 1;
  std::size_t step_size = 1;
  std::size_t num_syndromes_per_round = 0;
  std::vector<cudaq::qec::float_t> error_vec;
  std::string inner_decoder_name;
  cudaqx::heterogeneous_map inner_decoder_params;

  // Derived parameters.
  std::size_t num_windows = 0;
  std::size_t num_rounds = 0;
  std::size_t num_syndromes_per_window = 0;
  std::vector<std::unique_ptr<decoder>> inner_decoders;
  std::vector<std::size_t> first_columns;

public:
  sliding_window(const cudaqx::tensor<uint8_t> &H,
                   const cudaqx::heterogeneous_map &params)
      : decoder(H) {
    // Fetch parameters from the params map.
    window_size = params.get<std::size_t>("window_size", window_size);
    step_size = params.get<std::size_t>("step_size", step_size);
    num_syndromes_per_round = params.get<std::size_t>("num_syndromes_per_round",
                                                      num_syndromes_per_round);
    error_vec =
        params.get<std::vector<cudaq::qec::float_t>>("error_vec", error_vec);
    inner_decoder_name = params.get<std::string>("inner_decoder_name",
                                                 inner_decoder_name);
    inner_decoder_params = params.get<cudaqx::heterogeneous_map>(
        "inner_decoder_params", inner_decoder_params);

    // Perform error checking on the inputs.
    if (num_syndromes_per_round == 0) {
      throw std::invalid_argument(
          "sliding_window constructor: num_syndromes_per_round must be non-zero");
    }
    if (H.shape()[0] % num_syndromes_per_round != 0) {
      throw std::invalid_argument(
          "sliding_window constructor: Number of rows in H must be divisible "
          "by num_syndromes_per_round");
    }
    if (inner_decoder_name.empty()) {
      throw std::invalid_argument(
          "sliding_window constructor: inner_decoder_name must be non-empty");
    }
    if (inner_decoder_params.empty()) {
      throw std::invalid_argument(
          "sliding_window constructor: inner_decoder_params must be non-empty");
    }
    if (step_size == 0) {
      throw std::invalid_argument(
          "sliding_window constructor: step_size must be non-zero");
    }
    if (error_vec.empty()) {
      throw std::invalid_argument(
          "sliding_window constructor: error_vec must be non-empty");
    }

    // Enforce that H is already sorted.
    if (!cudaq::qec::pcm_is_sorted(H, num_syndromes_per_round)) {
      throw std::invalid_argument("sliding_window constructor: PCM must be "
                                  "sorted. See cudaq::qec::simplify_pcm.");
    }

    num_rounds = H.shape()[0] / num_syndromes_per_round;
    num_windows = (num_rounds - window_size) / step_size + 1;

    // Create the inner decoders.
    for (std::size_t w = 0; w < num_windows; ++w) {
      std::size_t start_round = w * step_size;
      std::size_t end_round = start_round + window_size - 1;
      auto [H_round, first_column, last_column] = cudaq::qec::get_pcm_for_rounds(
          H, num_syndromes_per_round, start_round, end_round,
          /*straddle_start_round=*/false, /*straddle_end_round=*/true);
      first_columns.push_back(first_column);
      printf("Creating a decoder for window %zu-%zu (dims %zu x %zu) "
            "first_column = %u, last_column = %u\n",
            start_round, end_round, H_round.shape()[0], H_round.shape()[1],
            first_column, last_column);
      auto inner_decoder = decoder::get(inner_decoder_name, H_round, inner_decoder_params);
      inner_decoders.push_back(std::move(inner_decoder));
    }
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) {
    decoder_result result{false, std::vector<float_t>(block_size, 0.0)};

    return result;
  }

  virtual ~sliding_window() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      sliding_window, static std::unique_ptr<decoder> create(
                            const cudaqx::tensor<uint8_t> &H,
                            const cudaqx::heterogeneous_map &params) {
        return std::make_unique<sliding_window>(H, params);
      })
};

CUDAQ_REGISTER_TYPE(sliding_window)

} // namespace cudaq::qec
