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

/// @brief Return a vector of column indices that would sort the pcm columns
/// in topological order.
/// @param row_indices For each column, a vector of row indices that have a
/// non-zero value in that column.
std::vector<std::uint32_t>
sort_pcm_columns(const std::vector<std::vector<std::uint32_t>> &row_indices);

/// @brief Return a vector of column indices that would sort the pcm columns
/// in topological order.
std::vector<std::uint32_t> sort_pcm_columns(const cudaqx::tensor<uint8_t> &pcm);

/// @brief Reorder the columns of a pcm according to the given column order.
/// @param pcm The pcm to reorder.
/// @param column_order The column order to use for reordering.
void reorder_pcm_columns(cudaqx::tensor<uint8_t> &pcm,
                         const std::vector<std::uint32_t> &column_order);

} // namespace cudaq::qec
