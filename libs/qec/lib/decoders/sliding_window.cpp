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
  std::vector<cudaq::qec::float_t> error_rate_vec;
  std::string inner_decoder_name;
  cudaqx::heterogeneous_map inner_decoder_params;

  // Derived parameters.
  std::size_t num_windows = 0;
  std::size_t num_rounds = 0;
  std::size_t num_syndromes_per_window = 0;
  std::vector<std::unique_ptr<decoder>> inner_decoders;
  std::vector<std::size_t> first_columns;
  cudaqx::tensor<std::uint8_t> full_pcm;

  // State data
  std::vector<std::vector<cudaq::qec::float_t>>
      rolling_window; // [batch_size, num_syndromes_per_window]
  std::size_t rw_filled = 0;
  std::size_t num_windows_decoded = 0;
  std::vector<std::vector<bool>> syndrome_mods; // [batch_size, syndrome_size]
  std::vector<decoder_result> rw_results;       // [batch_size]
  std::vector<double> window_proc_times;
  std::array<double, 10> window_proc_times_arr;

public:
  sliding_window(const cudaqx::tensor<uint8_t> &H,
                 const cudaqx::heterogeneous_map &params)
      : decoder(H), full_pcm(H) {
    // Fetch parameters from the params map.
    window_size = params.get<std::size_t>("window_size", window_size);
    step_size = params.get<std::size_t>("step_size", step_size);
    num_syndromes_per_round = params.get<std::size_t>("num_syndromes_per_round",
                                                      num_syndromes_per_round);
    error_rate_vec = params.get<std::vector<cudaq::qec::float_t>>(
        "error_rate_vec", error_rate_vec);
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
    if (error_rate_vec.empty()) {
      throw std::invalid_argument(
          "sliding_window constructor: error_rate_vec must be non-empty");
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
          error_rate_vec.begin() + first_column,
          error_rate_vec.begin() + last_column + 1);
      inner_decoder_params_mod.insert("error_rate_vec", error_vec_mod);

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
    if (syndrome.size() == this->syndrome_size) {
      // printf("%s:%d Decoding whole block\n", __FILE__, __LINE__);
      // Decode the whole thing, iterating over windows manually.
      decoder_result result;
      for (std::size_t w = 0; w < num_rounds; ++w) {
        std::vector<float_t> syndrome_round(
            syndrome.begin() + w * step_size * num_syndromes_per_round,
            syndrome.begin() + (w + 1) * step_size * num_syndromes_per_round);
        result = decode(syndrome_round);
      }
      return result;
    }
    // Else we're receiving a single round.
    if (rw_filled == 0) {
      // Initialize the syndrome mods and rw_results.
      syndrome_mods.resize(1);
      syndrome_mods[0].clear();
      syndrome_mods[0].resize(this->syndrome_size);
      rw_results.clear();
      rw_results.resize(1);
      rw_results[0].converged = true; // Gets set to false if we fail to decode
      rw_results[0].result.resize(this->block_size);
      rolling_window.resize(1);
      rolling_window[0].clear();
      rolling_window[0].resize(num_syndromes_per_window);
      window_proc_times.resize(num_windows);
      std::fill(window_proc_times.begin(), window_proc_times.end(), 0.0);
      rw_filled = 0;
      // printf("%s:%d Initializing window\n", __FILE__, __LINE__);
    }
    if (this->rw_filled == num_syndromes_per_window) {
      // printf("%s:%d Window is full, sliding the window\n", __FILE__,
      // __LINE__); The window is full. Slide existing data to the left and
      // write the new data at the end.
      std::copy(this->rolling_window[0].begin() + num_syndromes_per_round,
                this->rolling_window[0].end(), this->rolling_window[0].begin());
      std::copy(syndrome.begin(), syndrome.end(),
                this->rolling_window[0].end() - num_syndromes_per_round);
    } else {
      // Just copy the data to the end of the rolling window.
      // printf("%s:%d Copying data to the end of the rolling window\n",
      // __FILE__,
      //        __LINE__);
      std::copy(syndrome.begin(), syndrome.end(),
                this->rolling_window[0].begin() + this->rw_filled);
      this->rw_filled += num_syndromes_per_round;
    }
    if (rw_filled == num_syndromes_per_window) {
      // printf("%s:%d Decoding window %lu/%lu\n", __FILE__, __LINE__,
      //        num_windows_decoded + 1, num_windows);
      decode_window();

      num_windows_decoded++;
      if (num_windows_decoded == num_windows) {
        num_windows_decoded = 0;
        rw_filled = 0;
        // for (std::size_t w = 0; w < num_windows; ++w) {
        //   printf("Window %zu time: %.3f ms\n", w, window_proc_times[w]);
        // }
        // printf("%s:%d Returning decoder_result\n", __FILE__, __LINE__);
        return std::move(this->rw_results[0]);
      }
    }
    // printf("%s:%d Returning empty decoder_result\n", __FILE__, __LINE__);
    return decoder_result(); // empty return value
  }

  virtual std::vector<decoder_result>
  decode_batch(const std::vector<std::vector<float_t>> &syndromes) override {
    if (syndromes[0].size() == this->syndrome_size) {
      // printf("%s:%d Decoding whole block\n", __FILE__, __LINE__);
      // Decode the whole thing, iterating over windows manually.
      std::vector<decoder_result> results;
      std::vector<std::vector<float_t>> syndromes_round(syndromes.size());
      for (std::size_t w = 0; w < num_rounds; ++w) {
        for (std::size_t s = 0; s < syndromes.size(); ++s) {
          syndromes_round[s].resize(num_syndromes_per_round);
          std::copy(syndromes[s].begin() +
                        w * step_size * num_syndromes_per_round,
                    syndromes[s].begin() +
                        (w + 1) * step_size * num_syndromes_per_round,
                    syndromes_round[s].begin());
        }
        results = decode_batch(syndromes_round);
      }
      return results;
    }
    // Else we're receiving a single round.
    if (rw_filled == 0) {
      // Initialize the syndrome mods and rw_results.
      syndrome_mods.resize(syndromes.size());
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        syndrome_mods[s].clear();
        syndrome_mods[s].resize(this->syndrome_size);
      }
      rw_results.clear();
      rw_results.resize(syndromes.size());
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        rw_results[s].converged =
            true; // Gets set to false if we fail to decode
        rw_results[s].result.resize(this->block_size);
      }
      rolling_window.resize(syndromes.size());
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        rolling_window[s].clear();
        rolling_window[s].resize(num_syndromes_per_window);
      }
      window_proc_times.resize(num_windows);
      rw_filled = 0;
      // printf("%s:%d Initializing window\n", __FILE__, __LINE__);
    }
    if (this->rw_filled == num_syndromes_per_window) {
      // printf("%s:%d Window is full, sliding the window\n", __FILE__, __LINE__);
      // The window is full. Slide existing data to the left and write the new
      // data at the end.
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        std::copy(this->rolling_window[s].begin() + num_syndromes_per_round,
                  this->rolling_window[s].end(),
                  this->rolling_window[s].begin());
        std::copy(syndromes[s].begin(), syndromes[s].end(),
                  this->rolling_window[s].end() - num_syndromes_per_round);
      }
    } else {
      // Just copy the data to the end of the rolling window.
      // printf("%s:%d Copying data to the end of the rolling window\n", __FILE__,
      //        __LINE__);
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        std::copy(syndromes[s].begin(), syndromes[s].end(),
                  this->rolling_window[s].begin() + this->rw_filled);
      }
      this->rw_filled += num_syndromes_per_round;
    }
    if (rw_filled == num_syndromes_per_window) {
      // printf("%s:%d Decoding window %lu/%lu\n", __FILE__, __LINE__,
      //        num_windows_decoded + 1, num_windows);
      decode_window();

      num_windows_decoded++;
      if (num_windows_decoded == num_windows) {
        num_windows_decoded = 0;
        rw_filled = 0;
        // Dump the per window processing times.
        // for (std::size_t w = 0; w < num_windows; ++w) {
        //   printf("Window %zu time: %.3f ms\n", w, window_proc_times[w]);
        // }
        // printf("%s:%d Returning decoder_result\n", __FILE__, __LINE__);
        return std::move(this->rw_results);
      }
    }
    // printf("%s:%d Returning empty decoder_result\n", __FILE__, __LINE__);
    return std::vector<decoder_result>(); // empty return value
  }

  void decode_window() {
    auto t0 = std::chrono::high_resolution_clock::now();
    const auto &w = this->num_windows_decoded;
    std::size_t syndrome_start = w * step_size * num_syndromes_per_round;
    std::size_t syndrome_end = syndrome_start + num_syndromes_per_window - 1;
    std::size_t syndrome_start_next_window = syndrome_end + 1;
    std::size_t syndrome_end_next_window =
        syndrome_start_next_window + num_syndromes_per_round - 1;
    auto t3 = std::chrono::high_resolution_clock::now();
    if (w > 0) {
      // Modify the syndrome slice to account for the previous windows.
      for (std::size_t s = 0; s < this->rolling_window.size(); ++s) {
        for (std::size_t r = 0; r < num_syndromes_per_window; ++r) {
          auto &slice_val = this->rolling_window[s].at(r);
          slice_val =
              static_cast<double>(static_cast<std::uint8_t>(slice_val) ^
                                  syndrome_mods[s].at(r + syndrome_start));
        }
      }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    // printf("Window %zu: syndrome_start = %zu, syndrome_end = %zu length1 = "
    //        "%zu length2 = %zu\n",
    //        w, syndrome_start, syndrome_end, syndrome_slice.size(),
    //        syndrome_end - syndrome_start + 1);
    std::vector<decoder_result> inner_results;
    if (this->rolling_window.size() == 1) {
      inner_results.push_back(
          inner_decoders[w]->decode(this->rolling_window[0]));
    } else {
      inner_results = inner_decoders[w]->decode_batch(this->rolling_window);
    }
    // if (!inner_result.converged) {
    //   printf("Window %zu: inner decoder failed to converge\n", w);
    // }
    auto t5 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<uint8_t>> window_results(
        this->rolling_window.size());
    for (std::size_t s = 0; s < this->rolling_window.size(); ++s) {
      this->rw_results[s].converged &= inner_results[s].converged;
      cudaq::qec::convert_vec_soft_to_hard(inner_results[s].result,
                                           window_results[s]);
    }
    // Commit to everything up to the first column of the next window.
    auto t6 = std::chrono::high_resolution_clock::now();
    if (w < num_windows - 1) {
      // Prepare for the next window.
      auto next_window_first_column = first_columns[w + 1];
      auto this_window_first_column = first_columns[w];
      auto num_to_commit = next_window_first_column - this_window_first_column;
      // printf("  Committing %lu bits from window %zu\n", num_to_commit, w);
      for (std::size_t s = 0; s < this->rolling_window.size(); ++s) {
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          rw_results[s].result.at(c + this_window_first_column) =
              window_results[s].at(c);
        }
      }
      // We are committing to some errors that would affect the next round's
      // syndrome measurements. Therefore, we need to modify some of the
      // syndrome measurements for the next round to "back out" the errors
      // that we already know about (or more specifically, the errors we think
      // we've already accounted for).
      for (std::size_t s = 0; s < this->rolling_window.size(); ++s) {
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          if (rw_results[s].result.at(c + this_window_first_column)) {
            // This bit is a 1, so we need to modify the syndrome measurements
            // for the next window to account for this already-accounted-for
            // error. We do this by flipping the bit in the syndrome
            // measurements if the corresponding entry in the PCM is a 1.
            for (auto r = syndrome_start_next_window;
                 r <= syndrome_end_next_window; ++r) {
              syndrome_mods[s][r] =
                  syndrome_mods[s][r] ^ static_cast<bool>(full_pcm.at(
                                            {r, c + this_window_first_column}));
            }
          }
        }
      }
    } else {
      // This is the last window. Append ALL of window_result to
      // decoded_result.
      auto this_window_first_column = first_columns[w];
      auto num_to_commit = window_results[0].size();
      // printf("  Committing %zu bits from window %zu\n", num_to_commit, w);
      for (std::size_t s = 0; s < this->rolling_window.size(); ++s) {
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          rw_results[s].result.at(c + this_window_first_column) =
              window_results[s].at(c);
        }
      }
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    window_proc_times.at(w) +=
        std::chrono::duration<double>(t7 - t0).count() * 1000;
    window_proc_times_arr[3] += std::chrono::duration<double>(t3 - t0).count() * 1000;
    window_proc_times_arr[4] += std::chrono::duration<double>(t4 - t3).count() * 1000;
    window_proc_times_arr[5] += std::chrono::duration<double>(t5 - t4).count() * 1000;
    window_proc_times_arr[6] += std::chrono::duration<double>(t6 - t5).count() * 1000;
    window_proc_times_arr[7] += std::chrono::duration<double>(t7 - t6).count() * 1000;
    // printf("Window %zu time: %.3f ms (3:%.3fms 4:%.3fms 5:%.3fms 6:%.3fms "
    //        "7:%.3fms)\n",
    //        w, window_proc_times[w],
    //        std::chrono::duration<double>(t3 - t0).count() * 1000,
    //        std::chrono::duration<double>(t4 - t3).count() * 1000,
    //        std::chrono::duration<double>(t5 - t4).count() * 1000,
    //        std::chrono::duration<double>(t6 - t5).count() * 1000,
    //        std::chrono::duration<double>(t7 - t6).count() * 1000);
    // printf("Window %zu time: %.3f ms (3:%.3fms 4:%.3fms 5:%.3fms 6:%.3fms "
    //        "7:%.3fms)\n",
    //        w, window_proc_times[w], window_proc_times_arr[3],
    //        window_proc_times_arr[4], window_proc_times_arr[5],
    //        window_proc_times_arr[6], window_proc_times_arr[7]);
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
