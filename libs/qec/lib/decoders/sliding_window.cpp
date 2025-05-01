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
  std::vector<cudaqx::tensor<uint8_t>> decoder_pcms;
  std::vector<std::size_t> first_columns;
  cudaqx::tensor<std::uint8_t> full_pcm;

public:
  sliding_window(const cudaqx::tensor<uint8_t> &H,
                 const cudaqx::heterogeneous_map &params)
      : decoder(H), full_pcm(H) {
    // Fetch parameters from the params map.
    window_size = params.get<std::size_t>("window_size", window_size);
    step_size = params.get<std::size_t>("step_size", step_size);
    num_syndromes_per_round = params.get<std::size_t>("num_syndromes_per_round",
                                                      num_syndromes_per_round);
    error_vec =
        params.get<std::vector<cudaq::qec::float_t>>("error_vec", error_vec);
    inner_decoder_name =
        params.get<std::string>("inner_decoder_name", inner_decoder_name);
    inner_decoder_params = params.get<cudaqx::heterogeneous_map>(
        "inner_decoder_params", inner_decoder_params);

    // Perform error checking on the inputs.
    if (num_syndromes_per_round == 0) {
      throw std::invalid_argument("sliding_window constructor: "
                                  "num_syndromes_per_round must be non-zero");
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
    num_syndromes_per_window = num_syndromes_per_round * window_size;

    // Create the inner decoders.
    for (std::size_t w = 0; w < num_windows; ++w) {
      std::size_t start_round = w * step_size;
      std::size_t end_round = start_round + window_size - 1;
      auto [H_round, first_column, last_column] =
          cudaq::qec::get_pcm_for_rounds(
              H, num_syndromes_per_round, start_round, end_round,
              /*straddle_start_round=*/false, /*straddle_end_round=*/true);
      first_columns.push_back(first_column);

      // Slice the error vector to only include the current window.
      auto inner_decoder_params_mod = inner_decoder_params;
      std::vector<cudaq::qec::float_t> error_vec_mod(
          error_vec.begin() + first_column,
          error_vec.begin() + last_column + 1);
      inner_decoder_params_mod.insert("error_vec", error_vec_mod);

      printf("Creating a decoder for window %zu-%zu (dims %zu x %zu) "
             "first_column = %u, last_column = %u\n",
             start_round, end_round, H_round.shape()[0], H_round.shape()[1],
             first_column, last_column);
      auto inner_decoder =
          decoder::get(inner_decoder_name, H_round, inner_decoder_params_mod);
      inner_decoders.push_back(std::move(inner_decoder));
      decoder_pcms.emplace_back(std::move(H_round));
    }
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) override {
    // Initialize converged to true and set to false if any inner decoder
    // fails to converge.
    decoder_result outer_result{true, std::vector<float_t>(block_size, 0.0)};
    cudaqx::tensor<std::uint8_t> result_tensor(
        std::vector<std::size_t>{block_size});

    for (std::size_t w = 0; w < num_windows; ++w) {
      // printf("Processing syndrome %zu for window %zu\n", i, w);
      std::size_t syndrome_start = w * step_size * num_syndromes_per_round;
      std::size_t syndrome_end = syndrome_start + num_syndromes_per_window - 1;
      std::vector<float_t> syndrome_slice(syndrome.begin() + syndrome_start,
                                          syndrome.begin() + syndrome_end + 1);
      cudaqx::tensor<uint8_t> syndrome_mods(
          std::vector<std::size_t>{this->syndrome_size});
      if (w > 0) {
        // Modify the syndrome slice to account for the previous windows.
        // FIXME we can make this more efficient.
        cudaqx::tensor<uint8_t> committed_results(
            std::vector<std::size_t>{this->block_size});
        committed_results.borrow(result_tensor.data());
        syndrome_mods = full_pcm.dot(committed_results);
        for (std::size_t r = 0; r < num_syndromes_per_window; ++r) {
          auto &slice_val = syndrome_slice.at({r});
          slice_val =
              static_cast<double>(static_cast<std::uint8_t>(slice_val) ^
                                  syndrome_mods.at({r + syndrome_start}));
        }
      }
      // printf("Window %zu: syndrome_start = %zu, syndrome_end = %zu length1 = "
      //        "%zu length2 = %zu\n",
      //        w, syndrome_start, syndrome_end, syndrome_slice.size(),
      //        syndrome_end - syndrome_start + 1);
      auto inner_result = inner_decoders[w]->decode(syndrome_slice);
      // if (!inner_result.converged) {
      //   printf("Window %zu: inner decoder failed to converge\n", w);
      // }
      outer_result.converged &= inner_result.converged;
      cudaqx::tensor<uint8_t> window_result;
      cudaq::qec::convert_vec_soft_to_tensor_hard(inner_result.result,
                                                  window_result);
      const auto &window_pcm = decoder_pcms[w];
      // printf("PCM dims: %zu x %zu, window_result dims: %zu\n",
      //        window_pcm.shape()[0], window_pcm.shape()[1],
      //        window_result.shape()[0]);
      // auto result = window_pcm.dot(window_result);
      // Commit to everything up to the first column of the next window.
      if (w < num_windows - 1) {
        // Prepare for the next window.
        auto next_window_first_column = first_columns[w + 1];
        auto this_window_first_column = first_columns[w];
        auto num_to_commit =
            next_window_first_column - this_window_first_column;
        // printf("  Committing %u bits from window %zu\n", num_to_commit, w);
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          result_tensor.at({c + this_window_first_column}) =
              window_result.at({c});
        }
      } else {
        // This is the last window. Append ALL of window_result to
        // decoded_result.
        auto this_window_first_column = first_columns[w];
        auto num_to_commit = window_result.shape()[0];
        // printf("  Committing %zu bits from window %zu\n", num_to_commit, w);
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          result_tensor.at({c + this_window_first_column}) =
              window_result.at({c});
        }
      }
    }
    // Convert back to a vector of floats.
    for (std::size_t i = 0; i < block_size; ++i) {
      outer_result.result[i] = result_tensor.at({i});
    }

    return outer_result;
  }

  virtual std::vector<decoder_result>
  decode_batch(const std::vector<std::vector<float_t>> &syndromes) override {
    printf("Decoding batch of size %zu\n", syndromes.size());
    std::vector<decoder_result> results(syndromes.size());
    for (std::size_t i = 0; i < syndromes.size(); ++i) {
      results[i] = decode(syndromes[i]);
    }
    cudaqx::tensor<std::uint8_t> result_tensor(
        std::vector<std::size_t>{syndromes.size(), this->block_size});
    for (std::size_t w = 0; w < num_windows; ++w) {
      std::size_t syndrome_start = w * step_size * num_syndromes_per_round;
      std::size_t syndrome_end = syndrome_start + num_syndromes_per_window - 1;
      std::vector<std::vector<cudaq::qec::float_t>> syndrome_slices(
          syndromes.size());
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        syndrome_slices[s] = std::vector<cudaq::qec::float_t>(
            syndromes[s].begin() + syndrome_start,
            syndromes[s].begin() + syndrome_end + 1);
      }
      std::vector<cudaqx::tensor<uint8_t>> syndrome_mods(
          syndromes.size(), std::vector<std::size_t>{this->syndrome_size});
      if (w > 0) {
        // Modify the syndrome slice to account for the previous windows.
        // FIXME we can make this more efficient.
        // syndrome_mods is syndrome_size x num_syndromes_in_batch_call
        auto syndrome_mods = full_pcm.dot(result_tensor.transpose());
        for (std::size_t s = 0; s < syndromes.size(); ++s) {
          for (std::size_t r = 0; r < num_syndromes_per_window; ++r) {
            auto &slice_val = syndrome_slices[s].at({r});
            slice_val =
                static_cast<double>(static_cast<std::uint8_t>(slice_val) ^
                                    syndrome_mods.at({r + syndrome_start, s}));
          }
        }
      }
      // printf("Window %zu: syndrome_start = %zu, syndrome_end = %zu length1 = "
      //        "%zu length2 = %zu\n",
      //        w, syndrome_start, syndrome_end, syndrome_slice.size(),
      //        syndrome_end - syndrome_start + 1);
      auto inner_results = inner_decoders[w]->decode_batch(syndrome_slices);
      // if (!inner_result.converged) {
      //   printf("Window %zu: inner decoder failed to converge\n", w);
      // }
      std::vector<cudaqx::tensor<uint8_t>> window_results(syndromes.size());
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        results[s].converged &= inner_results[s].converged;
        cudaq::qec::convert_vec_soft_to_tensor_hard(inner_results[s].result,
                                                    window_results[s]);
      }
      const auto &window_pcm = decoder_pcms[w];
      // printf("PCM dims: %zu x %zu, window_result dims: %zu\n",
      //        window_pcm.shape()[0], window_pcm.shape()[1],
      //        window_result.shape()[0]);
      // auto result = window_pcm.dot(window_result);
      // Commit to everything up to the first column of the next window.
      if (w < num_windows - 1) {
        // Prepare for the next window.
        auto next_window_first_column = first_columns[w + 1];
        auto this_window_first_column = first_columns[w];
        auto num_to_commit =
            next_window_first_column - this_window_first_column;
        // printf("  Committing %u bits from window %zu\n", num_to_commit, w);
        for (std::size_t s = 0; s < syndromes.size(); ++s) {
          for (std::size_t c = 0; c < num_to_commit; ++c) {
            result_tensor.at({s, c + this_window_first_column}) =
                window_results[s].at({c});
          }
        }
      } else {
        // This is the last window. Append ALL of window_result to
        // decoded_result.
        auto this_window_first_column = first_columns[w];
        auto num_to_commit = window_results[0].shape()[0];
        // printf("  Committing %zu bits from window %zu\n", num_to_commit, w);
        for (std::size_t s = 0; s < syndromes.size(); ++s) {  
          for (std::size_t c = 0; c < num_to_commit; ++c) {
            result_tensor.at({s, c + this_window_first_column}) =
                window_results[s].at({c});
          }
        }
      }
    }
    return results;
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
