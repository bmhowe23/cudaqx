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
    }
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) override {
    // Initialize converged to true and set to false if any inner decoder
    // fails to converge.
    decoder_result outer_result{true, std::vector<float_t>(block_size, 0.0)};
    cudaqx::tensor<std::uint8_t> result_tensor(
        std::vector<std::size_t>{block_size});

    // A buffer containing the syndrome modifications necessary to account for
    // already-committed errors (used during the windowing process).
    cudaqx::tensor<uint8_t> syndrome_mods(
        std::vector<std::size_t>{this->syndrome_size});
    for (std::size_t w = 0; w < num_windows; ++w) {
      // printf("Processing syndrome %zu for window %zu\n", i, w);
      std::size_t syndrome_start = w * step_size * num_syndromes_per_round;
      std::size_t syndrome_end = syndrome_start + num_syndromes_per_window - 1;
      std::size_t syndrome_start_next_window = syndrome_end + 1;
      std::size_t syndrome_end_next_window =
          syndrome_start_next_window + num_syndromes_per_round - 1;
      std::vector<float_t> syndrome_slice(syndrome.begin() + syndrome_start,
                                          syndrome.begin() + syndrome_end + 1);
      if (w > 0) {
        // Modify the syndrome slice to account for the previous windows.
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
        // We are committing to some errors that would affect the next round's
        // syndrome measurements. Therefore, we need to modify some of the
        // syndrome measurements for the next round to "back out" the errors
        // that we already know about (or more specifically, the errors we think
        // we've already accounted for).
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          if (result_tensor.at({c + this_window_first_column})) {
            // This bit is a 1, so we need to modify the syndrome measurements
            // for the next window to account for this already-accounted-for
            // error. We do this by flipping the bit in the syndrome
            // measurements if the corresponding entry in the PCM is a 1.
            for (auto r = syndrome_start_next_window;
                 r <= syndrome_end_next_window; ++r) {
              syndrome_mods.at({r}) ^=
                  full_pcm.at({r, c + this_window_first_column});
            }
          }
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
    auto t0 = std::chrono::high_resolution_clock::now();
    printf("Decoding batch of size %zu\n", syndromes.size());
    std::vector<decoder_result> results(syndromes.size());
    cudaqx::tensor<std::uint8_t> result_tensor(
        std::vector<std::size_t>{syndromes.size(), this->block_size});
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<std::chrono::duration<double>> window_times(num_windows);
    // A buffer containing the syndrome modifications necessary to account for
    // already-committed errors (used during the windowing process).
    std::vector<cudaqx::tensor<uint8_t>> syndrome_mods(
        syndromes.size(), std::vector<std::size_t>{this->syndrome_size});
    for (std::size_t w = 0; w < num_windows; ++w) {
      auto t2 = std::chrono::high_resolution_clock::now();
      std::size_t syndrome_start = w * step_size * num_syndromes_per_round;
      std::size_t syndrome_end = syndrome_start + num_syndromes_per_window - 1;
      std::size_t syndrome_start_next_window = syndrome_end + 1;
      std::size_t syndrome_end_next_window =
          syndrome_start_next_window + num_syndromes_per_round - 1;
      std::vector<std::vector<cudaq::qec::float_t>> syndrome_slices(
          syndromes.size());
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        syndrome_slices[s] = std::vector<cudaq::qec::float_t>(
            syndromes[s].begin() + syndrome_start,
            syndromes[s].begin() + syndrome_end + 1);
      }
      auto t3 = std::chrono::high_resolution_clock::now();
      if (w > 0) {
        // Modify the syndrome slice to account for the previous windows.
        for (std::size_t s = 0; s < syndromes.size(); ++s) {
          for (std::size_t r = 0; r < num_syndromes_per_window; ++r) {
            auto &slice_val = syndrome_slices[s].at({r});
            slice_val =
                static_cast<double>(static_cast<std::uint8_t>(slice_val) ^
                                    syndrome_mods[s].at({r + syndrome_start}));
          }
        }
      }
      auto t4 = std::chrono::high_resolution_clock::now();
      // printf("Window %zu: syndrome_start = %zu, syndrome_end = %zu length1 = "
      //        "%zu length2 = %zu\n",
      //        w, syndrome_start, syndrome_end, syndrome_slice.size(),
      //        syndrome_end - syndrome_start + 1);
      auto inner_results = inner_decoders[w]->decode_batch(syndrome_slices);
      // if (!inner_result.converged) {
      //   printf("Window %zu: inner decoder failed to converge\n", w);
      // }
      auto t5 = std::chrono::high_resolution_clock::now();
      std::vector<cudaqx::tensor<uint8_t>> window_results(syndromes.size());
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        results[s].converged &= inner_results[s].converged;
        cudaq::qec::convert_vec_soft_to_tensor_hard(inner_results[s].result,
                                                    window_results[s]);
      }
      // Commit to everything up to the first column of the next window.
      auto t6 = std::chrono::high_resolution_clock::now();
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
        // We are committing to some errors that would affect the next round's
        // syndrome measurements. Therefore, we need to modify some of the
        // syndrome measurements for the next round to "back out" the errors
        // that we already know about (or more specifically, the errors we think
        // we've already accounted for).
        for (std::size_t s = 0; s < syndromes.size(); ++s) {
          for (std::size_t c = 0; c < num_to_commit; ++c) {
            if (result_tensor.at({s, c + this_window_first_column})) {
              // This bit is a 1, so we need to modify the syndrome measurements
              // for the next window to account for this already-accounted-for
              // error. We do this by flipping the bit in the syndrome
              // measurements if the corresponding entry in the PCM is a 1.
              for (auto r = syndrome_start_next_window;
                   r <= syndrome_end_next_window; ++r) {
                syndrome_mods[s].at({r}) ^=
                    full_pcm.at({r, c + this_window_first_column});
              }
            }
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
      auto t7 = std::chrono::high_resolution_clock::now();
      window_times[w] = t7 - t2;
      printf("Window %zu time: %.3f ms (3:%.3fms 4:%.3fms 5:%.3fms 6:%.3fms "
             "7:%.3fms)\n",
             w, std::chrono::duration<double>(window_times[w]).count() * 1000,
             std::chrono::duration<double>(t3 - t2).count() * 1000,
             std::chrono::duration<double>(t4 - t3).count() * 1000,
             std::chrono::duration<double>(t5 - t4).count() * 1000,
             std::chrono::duration<double>(t6 - t5).count() * 1000,
             std::chrono::duration<double>(t7 - t6).count() * 1000);
    }
    auto t8 = std::chrono::high_resolution_clock::now();
    // Convert back to a vector of floats.
    for (std::size_t s = 0; s < syndromes.size(); ++s) {
      results[s].result.resize(block_size);
      for (std::size_t j = 0; j < block_size; ++j) {
        results[s].result[j] = result_tensor.at({s, j});
      }
    }
    auto t9 = std::chrono::high_resolution_clock::now();
    printf("Total time: %.3f ms\n",
           std::chrono::duration<double>(t9 - t0).count() * 1000);
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
