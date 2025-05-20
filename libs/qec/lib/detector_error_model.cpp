/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/pcm_utils.h"

namespace cudaq::qec {

std::size_t detector_error_model::num_detectors() const {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[0];
  return 0;
}

std::size_t detector_error_model::num_error_mechanisms() const {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[1];
  return 0;
}

// FIXME - likely not correct
std::size_t detector_error_model::num_measurements() const {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[0];
  return 0;
}

/// @brief Return a sparse representation of the PCM.
/// @return A vector of vectors that sparsely represents the PCM. The size of
/// the outer vector is the number of columns in the PCM, and the i-th element
/// contains an inner vector of the row indices of the non-zero elements in the
/// i-th column of the PCM.
static std::vector<std::vector<std::uint32_t>>
dense_to_sparse(const cudaqx::tensor<uint8_t> &pcm) {
  if (pcm.rank() != 2) {
    throw std::invalid_argument("dense_to_sparse: PCM must be a 2D tensor");
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

void detector_error_model::canonicalize(uint32_t num_syndromes_per_round) {
  auto row_indices = dense_to_sparse(detector_error_matrix);
  auto column_order = get_sorted_pcm_column_indices(detector_error_matrix,
                                                    num_syndromes_per_round);
  decltype(column_order) obs_matrix_column_order;
  // March through the columns in topological order, and combine the probability
  // weight vectors if the columns have the same row indices.
  std::vector<std::vector<std::uint32_t>> new_row_indices;
  std::vector<double> new_weights;
  std::size_t num_obs = 0;
  if (this->observables_flips_matrix.rank() == 2)
    num_obs = this->observables_flips_matrix.shape()[0];
  const auto num_cols = column_order.size();
  for (std::size_t c = 0; c < num_cols; c++) {
    auto column_index = column_order[c];
    auto &curr_row_indices = row_indices[column_index];
    // If the column has no non-zero elements, or a weight of 0, then we skip
    // it.
    if (curr_row_indices.size() == 0 || error_rates[column_index] == 0)
      continue;
    if (new_row_indices.empty()) {
      new_row_indices.push_back(curr_row_indices);
      new_weights.push_back(error_rates[column_index]);
      obs_matrix_column_order.push_back(column_index);
    } else {
      auto &prev_row_indices = new_row_indices.back();
      if (prev_row_indices == curr_row_indices) {
        // The current column has the same row indices as the previous column,
        // so we update the error_rates and do NOT add the duplicate column.
        auto prev_weight = new_weights.back();
        auto curr_weight = error_rates[column_index];
        auto new_weight = 1.0 - (1.0 - prev_weight) * (1.0 - curr_weight);
        new_weights.back() = new_weight;
        // Verify that the observables are the same for the duplicate column.
        auto previous_column = column_order[c - 1];
        bool match = true;
        for (std::size_t r = 0; r < num_obs; r++) {
          if (this->observables_flips_matrix.at({r, previous_column}) !=
              this->observables_flips_matrix.at({r, column_index})) {
            match = false;
            break;
          }
        }
        if (!match) {
          throw std::runtime_error("detector_error_model::canonicalize: "
                                   "observables_flips_matrix is not "
                                   "consistent");
        }
      } else {
        // The current column has different row indices than the previous
        // column. So we add the current column to the new PCM, and update the
        // error_rates.
        new_row_indices.push_back(curr_row_indices);
        new_weights.push_back(error_rates[column_index]);
        obs_matrix_column_order.push_back(column_index);
      }
    }
  }

  auto old_shape = detector_error_matrix.shape();

  // The new PCM may have fewer columns than the original PCM.
  this->detector_error_matrix = cudaqx::tensor<uint8_t>(
      std::vector<std::size_t>{old_shape[0], new_row_indices.size()});
  for (std::size_t c = 0; c < new_row_indices.size(); c++)
    for (auto r : new_row_indices[c])
      this->detector_error_matrix.at({r, c}) = 1;

  // Reorder the observables_flips_matrix by making a copy of the original and
  // then creating the new, reordered version.
  cudaqx::tensor<uint8_t> obs_save(this->observables_flips_matrix.shape());
  obs_save.copy(this->observables_flips_matrix.data(),
                this->observables_flips_matrix.shape());
  this->observables_flips_matrix = cudaqx::tensor<uint8_t>(
      std::vector<std::size_t>{num_obs, obs_matrix_column_order.size()});
  for (std::size_t c = 0; c < obs_matrix_column_order.size(); c++)
    for (std::size_t r = 0; r < num_obs; r++)
      this->observables_flips_matrix.at({r, c}) =
          obs_save.at({r, obs_matrix_column_order[c]});
}

} // namespace cudaq::qec
