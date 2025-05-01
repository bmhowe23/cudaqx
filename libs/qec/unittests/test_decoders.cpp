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

void SlidingWindowDecoderTest(bool run_batched) {
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

  const std::string inner_decoder_name = "single_error_lut";
  cudaqx::heterogeneous_map sliding_window_params;
  sliding_window_params.insert("window_size", window_size);
  sliding_window_params.insert("step_size", step_size);
  sliding_window_params.insert("num_syndromes_per_round",
                               n_syndromes_per_round);
  sliding_window_params.insert("error_vec", simplified_weights);
  sliding_window_params.insert("inner_decoder_name", inner_decoder_name);

  cudaqx::heterogeneous_map inner_decoder_params;
  inner_decoder_params.insert("circuit_level_like", true);
  sliding_window_params.insert("inner_decoder_params", inner_decoder_params);

  auto sliding_window_decoder = cudaq::qec::decoder::get(
      "sliding_window", simplified_pcm, sliding_window_params);

  // Create some random syndromes.
  const int num_syndromes = 1000;
  std::vector<std::vector<cudaq::qec::float_t>> syndromes(num_syndromes);

  // Set a fixed number of error mechanisms to be non-zero. Since we are using
  // "single_error_lut", let's only set 1 error mechanism for now.
  const int num_error_mechanisms_to_set = 1;
  std::uniform_int_distribution<uint32_t> dist(0, n_cols - 1);
  std::mt19937_64 rng(13);
  for (std::size_t i = 0; i < num_syndromes; ++i) {
    syndromes[i] = std::vector<cudaq::qec::float_t>(n_rows, 0.0);
    for (int e = 0; e < num_error_mechanisms_to_set; ++e) {
      auto col = dist(rng);
      // printf("For syndrome %zu, setting error mechanism %d at column %u\n",
      // i, e, col);
      for (std::size_t r = 0; r < n_rows; ++r)
        syndromes[i][r] = pcm.at({r, col});
      // syndromes[i].dump_bits();
    }
  }

  // First decode the syndromes using a global decoder.
  std::vector<std::vector<uint8_t>> global_decoded_results(num_syndromes);
  auto t0 = std::chrono::high_resolution_clock::now();
  {
    printf("Generating global_decoder with PCM dims %zu x %zu\n",
           pcm.shape()[0], pcm.shape()[1]);
    auto global_decoder = cudaq::qec::decoder::get(
        inner_decoder_name, simplified_pcm, inner_decoder_params);
    printf("Done\n");
    if (run_batched) {
      auto dec_results = global_decoder->decode_batch(syndromes);
      for (std::size_t i = 0; i < num_syndromes; ++i) {
        ASSERT_TRUE(dec_results[i].converged);
        cudaq::qec::convert_vec_soft_to_hard(dec_results[i].result, global_decoded_results[i]);
      }
    } else {
      for (std::size_t i = 0; i < num_syndromes; ++i) {
        // printf("Decoding syndrome %zu\n", i);
        // syndromes[i].dump_bits();
        auto d = global_decoder->decode(syndromes[i]);
        ASSERT_TRUE(d.converged);
        cudaq::qec::convert_vec_soft_to_hard(d.result, global_decoded_results[i]);
      }
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto duration_global = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
  printf("Global decoder time: %ld ms\n", duration_global.count());

  // Now decode each syndrome using a windowed approach.
  std::vector<std::vector<uint8_t>> windowed_decoded_results(num_syndromes);
  auto t2 = std::chrono::high_resolution_clock::now();
  if (run_batched) {
    printf("Running batched decoding\n");
    auto dec_results = sliding_window_decoder->decode_batch(syndromes);
    for (std::size_t i = 0; i < num_syndromes; ++i) {
      // ASSERT_TRUE(dec_results[i].converged);
      cudaq::qec::convert_vec_soft_to_hard(dec_results[i].result, windowed_decoded_results[i]);
    }
  } else {
    for (std::size_t i = 0; i < num_syndromes; ++i) {
      // printf(" ------ Decoding syndrome %zu ------ \n", i);
      auto decoded_result = sliding_window_decoder->decode(syndromes[i]);
      // ASSERT_TRUE(decoded_result.converged);
      cudaq::qec::convert_vec_soft_to_hard(decoded_result.result,
                                          windowed_decoded_results[i]);
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  auto duration_windowed = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
  printf("Windowed decoder time: %ld ms\n", duration_windowed.count());

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

TEST(SlidingWindowDecoder, SlidingWindowDecoderTestNonBatched) {
  SlidingWindowDecoderTest(false);
}

TEST(SlidingWindowDecoder, SlidingWindowDecoderTestBatched) {
  SlidingWindowDecoderTest(true);
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
