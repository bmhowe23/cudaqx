/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/pcm_utils.h"
#include <cstring>

namespace cudaq::qec {

/// @brief Return a vector of column indices that would sort the pcm columns
/// in topological order.
/// @param row_indices For each column, a vector of row indices that have a
/// non-zero value in that column.
std::vector<std::uint32_t> get_sorted_pcm_column_indices(
    const std::vector<std::vector<std::uint32_t>> &row_indices) {
  std::vector<std::uint32_t> column_order(row_indices.size());
  std::iota(column_order.begin(), column_order.end(), 0);
  std::sort(column_order.begin(), column_order.end(),
            [&row_indices](const std::uint32_t &a, const std::uint32_t &b) {
              const auto &a_vec = row_indices[a];
              const auto &b_vec = row_indices[b];
              // Traverse a and b in parallel until a difference is found.
              auto a_size = a_vec.size();
              auto b_size = b_vec.size();
              for (std::size_t i = 0; i < a_size && i < b_size; i++) {
                if (a_vec[i] != b_vec[i])
                  return a_vec[i] < b_vec[i];
              }
              // If all elements are the same (up to the minimum size) then
              // the shorter vector should come first.
              return a_size < b_size;
            });

  return column_order;
}

/// @brief Return a vector of column indices that would sort the pcm columns
/// in topological order.
std::vector<std::uint32_t>
get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &pcm) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("pcm must be a 2D tensor");
  }

  auto num_rows = pcm.shape()[0];
  auto num_cols = pcm.shape()[1];

  // Form a sparse representation of the pcm
  std::vector<std::vector<std::uint32_t>> row_indices(num_cols);
  for (std::size_t r = 0; r < num_rows; r++) {
    auto *row = &pcm.at({r, 0});
    for (std::size_t c = 0; c < num_cols; c++)
      if (row[c])
        row_indices[c].push_back(r);
  }

  return get_sorted_pcm_column_indices(row_indices);
}

/// @brief Reorder the columns of a PCM according to the given column order.
/// @param pcm The PCM to reorder.
/// @param column_order The column order to use for reordering.
cudaqx::tensor<uint8_t>
reorder_pcm_columns(const cudaqx::tensor<uint8_t> &pcm,
                    const std::vector<std::uint32_t> &column_order) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("reorder_pcm_columns: PCM must be a 2D tensor");
  }

  if (pcm.shape()[1] != column_order.size()) {
    throw std::invalid_argument(
        "reorder_pcm_columns: column_order must be the same size as the number "
        "of columns in PCM");
  }

  auto num_rows = pcm.shape()[0];
  auto num_cols = pcm.shape()[1];

  auto transposed_pcm = pcm.transpose();
  cudaqx::tensor<uint8_t> new_pcm_t(transposed_pcm.shape());
  for (std::size_t c = 0; c < num_cols; c++) {
    auto *orig_col = &transposed_pcm.at({column_order[c], 0});
    auto *new_col = &new_pcm_t.at({c, 0});
    std::memcpy(new_col, orig_col, num_rows * sizeof(uint8_t));
  }

  return new_pcm_t.transpose();
}

/// @brief Sort the columns of a PCM in topological order.
/// @param pcm The PCM to sort.
/// @return A new PCM with the columns sorted in topological order.
cudaqx::tensor<uint8_t> sort_pcm_columns(const cudaqx::tensor<uint8_t> &pcm) {
  auto column_order = get_sorted_pcm_column_indices(pcm);
  return reorder_pcm_columns(pcm, column_order);
}

} // namespace cudaq::qec
