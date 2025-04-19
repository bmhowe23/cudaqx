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

/// @brief Return a vector of column indices that would sort the PCM columns
/// in topological order.
/// @param row_indices For each column, a vector of row indices that have a
/// non-zero value in that column.
/// @details This function tries to make a matrix that is close to a block
/// diagonal matrix from its input. Columns are first sorted by the index of the
/// first non-zero entry in the column, and if those match, then they are sorted
/// by the index of the last non-zero entry in the column. This ping pong
/// continues for the indices of the second non-zero element and the
/// second-to-last non-zero element, and so forth.
std::vector<std::uint32_t> get_sorted_pcm_column_indices(
    const std::vector<std::vector<std::uint32_t>> &row_indices) {
  std::vector<std::uint32_t> column_order(row_indices.size());
  std::iota(column_order.begin(), column_order.end(), 0);
  std::sort(column_order.begin(), column_order.end(),
            [&row_indices](const std::uint32_t &a, const std::uint32_t &b) {
              const auto &a_vec = row_indices[a];
              const auto &b_vec = row_indices[b];

              if (a_vec.size() == 0)
                return true;
              if (b_vec.size() == 0)
                return false;
              if (a_vec.size() == 0 && b_vec.size() == 0)
                return a < b; // stable sort.

              // Now we know both vectors have at least one element.

              // Have a and b iterators, both head and tail versions of both.
              auto a_it_head = a_vec.begin();
              auto a_it_tail = a_vec.end() - 1;
              auto b_it_head = b_vec.begin();
              auto b_it_tail = b_vec.end() - 1;

              do {
                // Compare the head elements.
                if (*a_it_head != *b_it_head)
                  return *a_it_head < *b_it_head;

                // Before checking the tail iterators, make sure they are not
                // aliased to the head elements that we just compared. If so,
                // we've exhausted one of the vectors and will return
                // accordingly.

                // Check if we ran out of "a" elements.
                if (a_it_head == a_it_tail && b_it_head != b_it_tail)
                  return true;
                // Check if we ran out of "b" elements.
                if (a_it_head != a_it_tail && b_it_head == b_it_tail)
                  return false;
                if (a_it_head == a_it_tail && b_it_head == b_it_tail)
                  return a < b; // stable sort.

                // Compare the tail elements.
                if (*a_it_tail != *b_it_tail)
                  return *a_it_tail < *b_it_tail;

                // Advance the head iterators.
                a_it_head++;
                b_it_head++;

                // Check to see if the new head iterators match the tail
                // iterators that we just compared. If so, we've exhausted one
                // of the vectors and will return accordingly.
                if (a_it_head == a_it_tail && b_it_head != b_it_tail)
                  return true;
                if (a_it_head != a_it_tail && b_it_head == b_it_tail)
                  return false;
                if (a_it_head == a_it_tail && b_it_head == b_it_tail)
                  return a < b; // stable sort.

                // Decrement the tail iterators.
                a_it_tail--;
                b_it_tail--;
              } while (true);

              // Unreachable.
              return a < b;
            });

  return column_order;
}

/// @brief Return a sparse representation of the PCM.
std::vector<std::vector<std::uint32_t>>
get_sparse_pcm(const cudaqx::tensor<uint8_t> &pcm) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("get_sparse_pcm: PCM must be a 2D tensor");
  }

  auto num_rows = pcm.shape()[0];
  auto num_cols = pcm.shape()[1];

  // Form a sparse representation of the PCM.
  std::vector<std::vector<std::uint32_t>> row_indices(num_cols);
  for (std::size_t r = 0; r < num_rows; r++) {
    auto *row = &pcm.at({r, 0});
    for (std::size_t c = 0; c < num_cols; c++)
      if (row[c])
        row_indices[c].push_back(r);
  }

  return row_indices;
}

/// @brief Return a vector of column indices that would sort the pcm columns
/// in topological order.
std::vector<std::uint32_t>
get_sorted_pcm_column_indices(const cudaqx::tensor<uint8_t> &pcm) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument(
        "get_sorted_pcm_column_indices: PCM must be a 2D tensor");
  }

  auto row_indices = get_sparse_pcm(pcm);

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

/// @brief Simplify a PCM by removing duplicate columns, and combine the
/// probability weight vectors accordingly.
/// @param pcm The PCM to simplify.
/// @param weights The probability weight vectors to combine.
/// @return A new PCM with the columns sorted in topological order, and the
/// probability weight vectors combined accordingly.
std::pair<cudaqx::tensor<uint8_t>, std::vector<double>>
simplify_pcm(const cudaqx::tensor<uint8_t> &pcm,
             const std::vector<double> &weights) {
  auto row_indices = get_sparse_pcm(pcm);
  auto column_order = get_sorted_pcm_column_indices(pcm);
  // March through the columns in topological order, and combine the probability
  // weight vectors if the columns have the same row indices.
  std::vector<std::vector<std::uint32_t>> new_row_indices;
  std::vector<double> new_weights;
  const auto num_cols = column_order.size();
  for (std::size_t c = 0; c < num_cols; c++) {
    auto column_index = column_order[c];
    auto &curr_row_indices = row_indices[column_index];
    if (c == 0) {
      // The first column is always added to the new PCM.
      new_row_indices.push_back(curr_row_indices);
      new_weights.push_back(weights[column_index]);
    } else {
      auto &prev_row_indices = new_row_indices.back();
      if (prev_row_indices == curr_row_indices) {
        // The current column has the same row indices as the previous column,
        // so we update the weights and do NOT add the duplicate column.
        auto prev_weight = new_weights.back();
        auto curr_weight = weights[column_index];
        auto new_weight = 1.0 - (1.0 - prev_weight) * (1.0 - curr_weight);
        new_weights.back() = new_weight;
      } else {
        // The current column has different row indices than the previous
        // column. So we add the current column to the new PCM, and update the
        // weights.
        new_row_indices.push_back(curr_row_indices);
        new_weights.push_back(weights[column_index]);
      }
    }
  }

  cudaqx::tensor<uint8_t> new_pcm(pcm.shape());
  for (std::size_t c = 0; c < new_row_indices.size(); c++)
    for (auto r : new_row_indices[c])
      new_pcm.at({r, c}) = 1;

  return std::make_pair(new_pcm, new_weights);
}

} // namespace cudaq::qec
