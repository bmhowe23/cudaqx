/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cuda-qx/core/tensor.h"

namespace cudaq::qec {

/// @brief Return a vector of column indices that would sort the PCM columns
/// in topological order.
/// @param row_indices For each column, a vector of row indices that have a
/// non-zero value in that column.
std::vector<std::uint32_t> get_sorted_pcm_column_indices(
    const std::vector<std::vector<std::uint32_t>> &row_indices);

/// @brief Return a vector of column indices that would sort the PCM columns
/// in topological order.
std::vector<std::uint32_t>
get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &pcm);

/// @brief Reorder the columns of a PCM according to the given column order.
/// @param pcm The PCM to reorder.
/// @param column_order The column order to use for reordering.
/// @return A new PCM with the columns reordered according to the given column
/// order.
cudaqx::tensor<uint8_t>
reorder_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                    const std::vector<std::uint32_t> &column_order);

/// @brief Sort the columns of a PCM in topological order.
/// @param pcm The PCM to sort.
/// @return A new PCM with the columns sorted in topological order.
cudaqx::tensor<uint8_t> sort_pcm_columns(const cudaqx::tensor<uint8_t> &pcm);

/// @brief Simplify a PCM by removing duplicate columns, and combine the
/// probability weight vectors accordingly.
/// @param pcm The PCM to simplify.
/// @param weights The probability weight vectors to combine.
/// @return A new PCM with the columns sorted in topological order, and the
/// probability weight vectors combined accordingly.
std::pair<cudaqx::tensor<uint8_t>, std::vector<double>>
simplify_pcm(const cudaqx::tensor<uint8_t> &pcm,
             const std::vector<double> &weights);

} // namespace cudaq::qec
