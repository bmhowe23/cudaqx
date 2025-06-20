/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/tensor.h"
#include <limits>
#include <random>

namespace cudaq::qec {

/// @brief Return a vector of column indices that would sort the PCM columns
/// in topological order.
/// @param row_indices For each column, a vector of row indices that have a
/// non-zero value in that column.
/// @param num_syndromes_per_round The number of syndromes per round. (Defaults
/// to 0, which means that no secondary per-round sorting will occur.)
/// @details This function tries to make a matrix that is close to a block
/// diagonal matrix from its input. Columns are first sorted by the index of the
/// first non-zero entry in the column, and if those match, then they are sorted
/// by the index of the last non-zero entry in the column. This ping pong
/// continues for the indices of the second non-zero element and the
/// second-to-last non-zero element, and so forth.
std::vector<std::uint32_t> get_sorted_pcm_column_indices(
    const std::vector<std::vector<std::uint32_t>> &row_indices,
    std::uint32_t num_syndromes_per_round = 0);

/// @brief Return a vector of column indices that would sort the PCM columns
/// in topological order.
/// @param num_syndromes_per_round The number of syndromes per round. (Defaults
/// to 0, which means that no secondary per-round sorting will occur.)
std::vector<std::uint32_t>
get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &pcm,
                              std::uint32_t num_syndromes_per_round = 0);

/// @brief Check if a PCM is sorted.
/// @param pcm The PCM to check.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @return True if the PCM is sorted, false otherwise.
bool pcm_is_sorted(const cudaqx::tensor<uint8_t> &pcm,
                   std::uint32_t num_syndromes_per_round = 0);

/// @brief Reorder the columns of a PCM according to the given column order.
/// Note: this may return a subset of the columns in the original PCM if the
/// \p column_order does not contain all of the columns in the original PCM.
/// @param pcm The PCM to reorder.
/// @param column_order The column order to use for reordering.
/// @param row_begin The first row to include in the reordering. Leave at the
/// default value to include all rows.
/// @param row_end The last row to include in the reordering. Leave at the
/// default value to include all rows.
/// @return A new PCM with the columns reordered according to the given column
/// order.
cudaqx::tensor<uint8_t>
reorder_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                    const std::vector<std::uint32_t> &column_order,
                    uint32_t row_begin = 0,
                    uint32_t row_end = std::numeric_limits<uint32_t>::max());

/// @brief Sort the columns of a PCM in topological order.
/// @param pcm The PCM to sort.
/// @return A new PCM with the columns sorted in topological order.
cudaqx::tensor<uint8_t>
sort_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                 std::uint32_t num_syndromes_per_round = 0);

/// @brief Simplify a PCM by removing duplicate columns and 0-weight columns,
/// and combine the probability weight vectors accordingly.
/// @param pcm The PCM to simplify.
/// @param weights The probability weight vectors to combine.
/// @return A new PCM with the columns sorted in topological order, and the
/// probability weight vectors combined accordingly.
std::pair<cudaqx::tensor<uint8_t>, std::vector<double>>
simplify_pcm(const cudaqx::tensor<uint8_t> &pcm,
             const std::vector<double> &weights,
             std::uint32_t num_syndromes_per_round = 0);

/// @brief Get a sub-PCM for a range of rounds. It is recommended (but not
/// required) that you call sort_pcm_columns() before calling this function.
/// @param pcm The PCM to get a sub-PCM for.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @param start_round The start round (0-based).
/// @param end_round The end round (0-based).
/// @param straddle_start_round Whether to include columns that straddle the
/// start_round (defaults to false)
/// @param straddle_end_round Whether to include columns that straddle the
/// end_round (defaults to false)
/// @return A tuple with the new PCM with the columns in the range [start_round,
/// end_round], the first column included, and the last column included.
std::tuple<cudaqx::tensor<uint8_t>, std::uint32_t, std::uint32_t>
get_pcm_for_rounds(const cudaqx::tensor<uint8_t> &pcm,
                   std::uint32_t num_syndromes_per_round,
                   std::uint32_t start_round, std::uint32_t end_round,
                   bool straddle_start_round = false,
                   bool straddle_end_round = false);

/// @brief Generate a random PCM with the given parameters.
/// @param n_rounds The number of rounds in the PCM.
/// @param n_errs_per_round The number of errors per round in the PCM.
/// @param n_syndromes_per_round The number of syndromes per round in the PCM.
/// @param weight The weight of the PCM.
/// @param rng The random number generator to use (e.g.
/// std::mt19937_64(your_seed))
/// @return A random PCM with the given parameters.
cudaqx::tensor<uint8_t> generate_random_pcm(std::size_t n_rounds,
                                            std::size_t n_errs_per_round,
                                            std::size_t n_syndromes_per_round,
                                            int weight, std::mt19937_64 &&rng);

/// @brief Randomly permute the columns of a PCM.
/// @param pcm The PCM to permute.
/// @param rng The random number generator to use (e.g.
/// std::mt19937_64(your_seed))
/// @return A new PCM with the columns permuted randomly.
cudaqx::tensor<uint8_t> shuffle_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                                            std::mt19937_64 &&rng);

/// @brief Extend a PCM to the given number of rounds.
/// @param pcm The PCM to extend.
/// @param num_syndromes_per_round The number of syndromes per round.
/// @param n_rounds The number of rounds to extend the PCM to.
/// @return A pair of the new PCM and the list of column indices from the
/// original PCM that were used to form the new PCM.
std::pair<cudaqx::tensor<uint8_t>, std::vector<std::uint32_t>>
pcm_extend_to_n_rounds(const cudaqx::tensor<uint8_t> &pcm,
                       std::size_t num_syndromes_per_round,
                       std::uint32_t n_rounds);

} // namespace cudaq::qec
