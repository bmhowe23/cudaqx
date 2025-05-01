/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include <cmath>
#include <future>
#include <gtest/gtest.h>

TEST(DecoderUtils, CovertHardToSoft) {
  std::vector<int> in = {1, 0, 1, 1};
  std::vector<float> out;
  std::vector<float> expected_out = {1.0, 0.0, 1.0, 1.0};

  cudaq::qec::convert_vec_hard_to_soft(in, out);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  expected_out = {0.9, 0.1, 0.9, 0.9};
  cudaq::qec::convert_vec_hard_to_soft(in, out, 0.9f, 0.1f);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  std::vector<std::vector<int>> in2 = {{1, 0}, {0, 1}};
  std::vector<std::vector<double>> out2;
  std::vector<std::vector<double>> expected_out2 = {{0.9, 0.1}, {0.1, 0.9}};
  cudaq::qec::convert_vec_hard_to_soft(in2, out2, 0.9, 0.1);
  for (int r = 0; r < out2.size(); r++) {
    ASSERT_EQ(out2.size(), expected_out2.size());
    for (int c = 0; c < out2.size(); c++)
      ASSERT_EQ(out2[r][c], expected_out2[r][c]);
  }
}

TEST(DecoderUtils, CovertSoftToHard) {
  std::vector<float> in = {0.6, 0.4, 0.7, 0.8};
  std::vector<bool> out;
  std::vector<bool> expected_out = {true, false, true, true};

  cudaq::qec::convert_vec_soft_to_hard(in, out);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  expected_out = {true, true, true, true};
  cudaq::qec::convert_vec_soft_to_hard(in, out, 0.4f);
  ASSERT_EQ(out.size(), expected_out.size());
  for (int i = 0; i < out.size(); i++)
    ASSERT_EQ(out[i], expected_out[i]);

  std::vector<std::vector<double>> in2 = {{0.6, 0.4}, {0.7, 0.8}};
  std::vector<std::vector<int>> out2;
  std::vector<std::vector<int>> expected_out2 = {{1, 0}, {1, 1}};
  cudaq::qec::convert_vec_soft_to_hard(in2, out2);
  for (int r = 0; r < out2.size(); r++) {
    ASSERT_EQ(out2.size(), expected_out2.size());
    for (int c = 0; c < out2.size(); c++)
      ASSERT_EQ(out2[r][c], expected_out2[r][c]);
  }
}

TEST(SampleDecoder, checkAPI) {
  using cudaq::qec::float_t;

  std::size_t block_size = 10;
  std::size_t syndrome_size = 4;
  cudaqx::tensor<uint8_t> H({syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("sample_decoder", H);
  std::vector<float_t> syndromes(syndrome_size);
  auto dec_result = d->decode(syndromes);
  ASSERT_EQ(dec_result.result.size(), block_size);
  for (auto x : dec_result.result)
    ASSERT_EQ(x, 0.0f);

  // Async test
  dec_result = d->decode_async(syndromes).get();
  ASSERT_EQ(dec_result.result.size(), block_size);
  for (auto x : dec_result.result)
    ASSERT_EQ(x, 0.0f);

  // Test the move constructor and move assignment operator

  // Multi test
  auto dec_results = d->decode_batch({syndromes, syndromes});
  ASSERT_EQ(dec_results.size(), 2);
  for (auto &m : dec_results)
    for (auto x : m.result)
      ASSERT_EQ(x, 0.0f);
}

TEST(SteaneLutDecoder, checkAPI) {
  using cudaq::qec::float_t;

  // Use Hx from the [7,1,3] Steane code from
  // https://en.wikipedia.org/wiki/Steane_code.
  std::size_t block_size = 7;
  std::size_t syndrome_size = 3;
  cudaqx::heterogeneous_map custom_args;

  std::vector<uint8_t> H_vec = {0, 0, 0, 1, 1, 1, 1,  // IIIXXXX
                                0, 1, 1, 0, 0, 1, 1,  // IXXIIXX
                                1, 0, 1, 0, 1, 0, 1}; // XIXIXIX
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("single_error_lut", H, custom_args);

  // Run decoding on all possible syndromes.
  const std::size_t num_syndromes_to_check = 1 << syndrome_size;
  bool convergeTrueFound = false;
  bool convergeFalseFound = false;
  assert(syndrome_size <= 64); // Assert due to "1 << bit" below.
  for (std::size_t syn_idx = 0; syn_idx < num_syndromes_to_check; syn_idx++) {
    // Construct a syndrome.
    std::vector<float_t> syndrome(syndrome_size, 0.0);
    for (int bit = 0; bit < syndrome_size; bit++)
      if (syn_idx & (1 << bit))
        syndrome[bit] = 1.0;

    // Perform decoding.
    auto dec_result = d->decode(syndrome);

    // Check results.
    ASSERT_EQ(dec_result.result.size(), block_size);
    const auto printResults = true;
    if (printResults) {
      std::string syndrome_str(syndrome_size, '0');
      for (std::size_t j = 0; j < syndrome_size; j++)
        if (syndrome[j] >= 0.5)
          syndrome_str[j] = '1';
      std::cout << "Syndrome " << syndrome_str
                << " returned: {converged: " << dec_result.converged
                << ", result: {";
      for (std::size_t j = 0; j < block_size; j++) {
        std::cout << dec_result.result[j];
        if (j < block_size - 1)
          std::cout << ",";
        else
          std::cout << "}}\n";
      }
    }
    convergeTrueFound |= dec_result.converged;
    convergeFalseFound |= !dec_result.converged;
  }
  ASSERT_TRUE(convergeTrueFound);
  ASSERT_FALSE(convergeFalseFound);
}

void check_pcm_equality(const cudaqx::tensor<uint8_t> &a,
                        const cudaqx::tensor<uint8_t> &b,
                        bool use_assert = true) {
  if (a.rank() != 2 || b.rank() != 2) {
    throw std::runtime_error("PCM must be a 2D tensor");
  }
  ASSERT_EQ(a.shape(), b.shape());
  auto num_rows = a.shape()[0];
  auto num_cols = a.shape()[1];
  for (std::size_t r = 0; r < num_rows; ++r) {
    for (std::size_t c = 0; c < num_cols; ++c) {
      if (a.at({r, c}) != b.at({r, c})) {
        if (use_assert)
          ASSERT_EQ(a.at({r, c}), b.at({r, c}))
              << "a.at({" << r << ", " << c << "}) = " << a.at({r, c})
              << ", b.at({" << r << ", " << c << "}) = " << b.at({r, c})
              << "\n";
        else
          EXPECT_EQ(a.at({r, c}), b.at({r, c}))
              << "a.at({" << r << ", " << c << "}) = " << a.at({r, c})
              << ", b.at({" << r << ", " << c << "}) = " << b.at({r, c})
              << "\n";
      }
    }
  }
}

TEST(SlidingWindowDecoder, SlidingWindowDecoderTest) {
  std::size_t n_rounds = 8;
  std::size_t n_errs_per_round = 30;
  std::size_t n_syndromes_per_round = 10;
  std::size_t n_cols = n_rounds * n_errs_per_round;
  std::size_t n_rows = n_rounds * n_syndromes_per_round;
  std::size_t weight = 3;

  cudaqx::tensor<uint8_t> pcm = cudaq::qec::generate_random_pcm(
      n_rounds, n_errs_per_round, n_syndromes_per_round, weight,
      std::mt19937_64(13));
  ASSERT_EQ(pcm.shape()[0], n_rows);
  ASSERT_EQ(pcm.shape()[1], n_cols);
  std::vector<double> weights(n_cols, 0.01);
  auto [simplified_pcm, simplified_weights] =
      cudaq::qec::simplify_pcm(pcm, weights, n_syndromes_per_round);
  ASSERT_TRUE(cudaq::qec::pcm_is_sorted(simplified_pcm, n_syndromes_per_round));

  const std::size_t window_size = 3;
  const std::size_t step_size = 1;
  const std::size_t commit_size = window_size - step_size;
  const std::size_t n_windows = (n_rounds - window_size) / step_size + 1;
  const std::size_t num_syndromes_per_window =
      window_size * n_syndromes_per_round;

  cudaqx::heterogeneous_map custom_args;
  custom_args.insert("window_size", window_size);
  custom_args.insert("step_size", step_size);
  custom_args.insert("num_syndromes_per_round", n_syndromes_per_round);
  custom_args.insert("error_vec", simplified_weights);
  custom_args.insert("inner_decoder_name", "single_error_lut");

  cudaqx::heterogeneous_map inner_decoder_params;
  inner_decoder_params.insert("circuit_level_like", true);
  custom_args.insert("inner_decoder_params", inner_decoder_params);

  auto d =
      cudaq::qec::decoder::get("sliding_window", simplified_pcm, custom_args);

  return;

  // Store a list of decoders and the PCM for each window.
  std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders;
  std::vector<cudaqx::tensor<uint8_t>> decoder_pcms;
  std::vector<std::uint32_t> first_columns;
  printf("Original PCM size: %zu x %zu\n", pcm.shape()[0], pcm.shape()[1]);
  // pcm.dump_bits();
  for (std::size_t w = 0; w < n_windows; ++w) {
    std::size_t start_round = w * step_size;
    std::size_t end_round = start_round + window_size - 1;
    auto [H, first_column, last_column] = cudaq::qec::get_pcm_for_rounds(
        pcm, n_syndromes_per_round, start_round, end_round,
        /*straddle_start_round=*/false, /*straddle_end_round=*/true);
    first_columns.push_back(first_column);
    printf("Creating a decoder for window %zu-%zu (dims %zu x %zu) "
           "first_column = %u, last_column = %u\n",
           start_round, end_round, H.shape()[0], H.shape()[1], first_column,
           last_column);
    // printf("H:\n");
    // H.dump_bits();
    auto d = cudaq::qec::decoder::get("single_error_lut", H, custom_args);
    decoders.push_back(std::move(d));
    decoder_pcms.push_back(std::move(H));
  }

  // Create some random syndromes.
  const int num_syndromes = 1000;
  std::vector<cudaqx::tensor<uint8_t>> syndromes(num_syndromes);

  // Set a fixed number of error mechanisms to be non-zero. Since we are using
  // "single_error_lut", let's only set 1 error mechanism for now.
  const int num_error_mechanisms_to_set = 1;
  std::uniform_int_distribution<uint32_t> dist(0, n_cols - 1);
  std::mt19937_64 rng(13);
  for (std::size_t i = 0; i < num_syndromes; ++i) {
    syndromes[i] = cudaqx::tensor<uint8_t>(std::vector<std::size_t>{n_rows});
    for (int e = 0; e < num_error_mechanisms_to_set; ++e) {
      auto col = dist(rng);
      // printf("For syndrome %zu, setting error mechanism %d at column %u\n",
      // i, e, col);
      for (std::size_t r = 0; r < n_rows; ++r)
        syndromes[i].at({r}) ^= pcm.at({r, col});
      // syndromes[i].dump_bits();
    }
  }

  // First decode the syndromes using a global decoder.
  std::vector<std::vector<uint8_t>> global_decoded_results(num_syndromes);
  {
    printf("Generating global_decoder with PCM dims %zu x %zu\n",
           pcm.shape()[0], pcm.shape()[1]);
    auto global_decoder =
        cudaq::qec::decoder::get("single_error_lut", pcm, custom_args);
    printf("Done\n");
    for (std::size_t i = 0; i < num_syndromes; ++i) {
      // printf("Decoding syndrome %zu\n", i);
      // syndromes[i].dump_bits();
      auto d = global_decoder->decode(syndromes[i]);
      ASSERT_TRUE(d.converged);
      cudaq::qec::convert_vec_soft_to_hard(d.result, global_decoded_results[i]);
    }
  }

  // Now decode each syndrome using a windowed approach.
  std::vector<std::vector<uint8_t>> windowed_decoded_results(num_syndromes);
  for (std::size_t i = 0; i < num_syndromes; ++i) {
    std::vector<uint8_t> decoded_result(n_cols);
    auto decoded_result_it = decoded_result.begin();

    for (std::size_t w = 0; w < n_windows; ++w) {
      // printf("Processing syndrome %zu for window %zu\n", i, w);
      std::size_t syndrome_start = w * step_size * n_syndromes_per_round;
      std::size_t syndrome_end = syndrome_start + num_syndromes_per_window - 1;
      auto syndrome_slice = cudaqx::tensor<uint8_t>(
          std::vector<std::size_t>{num_syndromes_per_window});
      for (std::size_t r = 0; r < num_syndromes_per_window; ++r)
        syndrome_slice.at({r}) = syndromes[i].at({syndrome_start + r});
      cudaqx::tensor<uint8_t> syndrome_mods(syndromes[i].shape());
      if (w > 0) {
        // Modify the syndrome slice to account for the previous windows.
        // FIXME we can make this more efficient.
        cudaqx::tensor<uint8_t> committed_results(
            std::vector<std::size_t>{n_cols});
        committed_results.borrow(decoded_result.data());
        syndrome_mods = pcm.dot(committed_results);
        for (std::size_t r = 0; r < num_syndromes_per_window; ++r)
          syndrome_slice.at({r}) ^= syndrome_mods.at({r + syndrome_start});
      }
      auto d = decoders[w]->decode(syndrome_slice);
      cudaqx::tensor<uint8_t> window_result;
      cudaq::qec::convert_vec_soft_to_tensor_hard(d.result, window_result);
      const auto &window_pcm = decoder_pcms[w];
      // printf("PCM dims: %zu x %zu, window_result dims: %zu\n",
      //        window_pcm.shape()[0], window_pcm.shape()[1],
      //        window_result.shape()[0]);
      auto result = window_pcm.dot(window_result);
      // Commit to everything up to the first column of the next window.
      if (w < n_windows - 1) {
        // Prepare for the next window.
        auto next_window_first_column = first_columns[w + 1];
        auto this_window_first_column = first_columns[w];
        auto num_to_commit =
            next_window_first_column - this_window_first_column;
        // printf("  Committing %u bits from window %zu\n", num_to_commit, w);
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          ASSERT_NE(decoded_result_it, decoded_result.end());
          *decoded_result_it++ = window_result.at({c});
        }
      } else {
        // This is the last window. Append ALL of window_result to
        // decoded_result.
        auto num_to_commit = window_result.shape()[0];
        // printf("  Committing %zu bits from window %zu\n", num_to_commit, w);
        for (std::size_t c = 0; c < num_to_commit; ++c) {
          ASSERT_NE(decoded_result_it, decoded_result.end());
          *decoded_result_it++ = window_result.at({c});
        }
      }
    }
    EXPECT_EQ(decoded_result.end(), decoded_result_it);
    EXPECT_EQ(decoded_result.size(), n_cols);
    windowed_decoded_results[i] = std::move(decoded_result);
  }

  // Check that the global and windowed decoders agree.
  auto print_as_bits = [](const std::vector<uint8_t> &v) {
    std::string s;
    s.reserve(v.size());
    for (auto r : v)
      s += (r == 0) ? '.' : '1';
    return s;
  };
  for (std::size_t i = 0; i < num_syndromes; ++i) {
    bool decoder_agreement =
        global_decoded_results[i] == windowed_decoded_results[i];
    EXPECT_EQ(decoder_agreement, true)
        << "Comparison failed for syndrome " << i;
    if (!decoder_agreement) {
      printf("Global   decoder result: %s\n",
             print_as_bits(global_decoded_results[i]).c_str());
      printf("Windowed decoder result: %s\n",
             print_as_bits(windowed_decoded_results[i]).c_str());
    }
  }
}

TEST(AsyncDecoderResultTest, MoveConstructorTransfersFuture) {
  std::promise<cudaq::qec::decoder_result> promise;
  std::future<cudaq::qec::decoder_result> future = promise.get_future();

  cudaq::qec::async_decoder_result original(std::move(future));
  EXPECT_TRUE(original.fut.valid());

  cudaq::qec::async_decoder_result moved(std::move(original));
  EXPECT_TRUE(moved.fut.valid());
  EXPECT_FALSE(original.fut.valid());
}

TEST(AsyncDecoderResultTest, MoveAssignmentTransfersFuture) {
  std::promise<cudaq::qec::decoder_result> promise;
  std::future<cudaq::qec::decoder_result> future = promise.get_future();

  cudaq::qec::async_decoder_result first(std::move(future));
  cudaq::qec::async_decoder_result second = std::move(first);

  EXPECT_TRUE(second.fut.valid());
  EXPECT_FALSE(first.fut.valid());
}
