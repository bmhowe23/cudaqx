/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

namespace cudaq::qec {

/// @brief This is a simple LUT (LookUp Table) decoder that demonstrates how to
/// build a simple decoder that can decode errors with a small number of errors
/// in the block.
class pymatching : public decoder {
private:
  pm::UserGraph user_graph;

  // Input parameters
  std::vector<double> error_rate_vec;

public:
  pymatching(const cudaqx::tensor<uint8_t> &H,
             const cudaqx::heterogeneous_map &params)
      : decoder(H) {

    if (params.contains("error_rate_vec")) {
      error_rate_vec = params.get<std::vector<double>>("error_rate_vec");
      if (error_rate_vec.size() != block_size) {
        throw std::runtime_error("error_rate_vec must be of size block_size");
      }
      // Validate that the values in the error_rate_vec are between 0 and 1.
      for (auto error_rate : error_rate_vec) {
        if (error_rate < 0.0 || error_rate > 1.0) {
          throw std::runtime_error(
              "error_rate_vec value is out of range [0, 1]");
        }
      }
    }

    user_graph = pm::UserGraph(H.shape()[0]);

    // Transpose H to allow us to quickly traverse the columns of H.
    auto sparse = cudaq::qec::dense_to_sparse(H.transpose());
    std::vector<size_t> observables;
    std::size_t col_idx = 0;
    for (auto &col : sparse) {
      double weight = 1.0;
      if (col_idx < error_rate_vec.size()) {
        weight = error_rate_vec[col_idx];
      }
      if (col.size() == 2) {
        user_graph.add_or_merge_edge(col[0], col[1], observables, weight, 0.0,
                                     pm::MERGE_STRATEGY::DISALLOW);
      }
      col_idx++;
    }
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) {
    decoder_result result{false, std::vector<float_t>(block_size, 0.0)};
    auto &mwpm = user_graph.get_mwpm();
    std::vector<int64_t> edges;
    std::vector<uint64_t> detection_events;
    pm::decode_detection_events_to_edges(mwpm, detection_events, edges);
    result.result.resize(edges.size());
    // TODO - very that the edge numbers are the column indices of H.
    for (auto e : edges) {
      result.result[e] = 1.0;
    }
    return result;
  }

  virtual ~pymatching() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      pymatching, static std::unique_ptr<decoder> create(
                      const cudaqx::tensor<uint8_t> &H,
                      const cudaqx::heterogeneous_map &params) {
        return std::make_unique<pymatching>(H, params);
      })
};

CUDAQ_REGISTER_TYPE(pymatching)

} // namespace cudaq::qec
