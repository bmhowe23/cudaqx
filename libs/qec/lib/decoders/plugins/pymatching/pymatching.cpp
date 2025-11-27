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

  // Map of edge pairs to column indices. This does not seem particularly
  // efficient.
  std::map<std::pair<int64_t, int64_t>, size_t> edge2col_idx;

  // Helper function to make a canonical edge from two nodes.
  std::pair<int64_t, int64_t> make_canonical_edge(int64_t node1,
                                                  int64_t node2) {
    return std::make_pair(std::min(node1, node2), std::max(node1, node2));
  }

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

    auto sparse = cudaq::qec::dense_to_sparse(H);
    std::vector<size_t> observables;
    std::size_t col_idx = 0;
    for (auto &col : sparse) {
      double weight = 1.0;
      if (col_idx < error_rate_vec.size()) {
        weight = error_rate_vec[col_idx];
      }
      if (col.size() == 2) {
        edge2col_idx[make_canonical_edge(col[0], col[1])] = col_idx;
        user_graph.add_or_merge_edge(col[0], col[1], observables, weight, 0.0,
                                     pm::MERGE_STRATEGY::DISALLOW);
      } else if (col.size() == 1) {
        edge2col_idx[make_canonical_edge(col[0], -1)] = col_idx;
        user_graph.add_or_merge_boundary_edge(col[0], observables, weight, 0.0,
                                              pm::MERGE_STRATEGY::DISALLOW);
      } else {
        throw std::runtime_error(
            "Invalid column in H: " + std::to_string(col_idx) + " has " +
            std::to_string(col.size()) + " ones. Must have 1 or 2 ones.");
      }
      col_idx++;
    }
  }

  virtual decoder_result decode(const std::vector<float_t> &syndrome) {
    decoder_result result{false, std::vector<float_t>(block_size, 0.0)};
    auto &mwpm = user_graph.get_mwpm_with_search_graph();
    std::vector<int64_t> edges;
    std::vector<uint64_t> detection_events;
    detection_events.reserve(syndrome.size());
    for (size_t i = 0; i < syndrome.size(); i++)
      if (syndrome[i] > 0.5)
        detection_events.push_back(i);
    pm::decode_detection_events_to_edges(mwpm, detection_events, edges);
    // Loop over the edge pairs
    assert(edges.size() % 2 == 0);
    for (size_t i = 0; i < edges.size(); i += 2) {
      auto edge = make_canonical_edge(edges.at(i), edges.at(i + 1));
      auto col_idx = edge2col_idx.at(edge);
      result.result[col_idx] = 1.0;
    }
    result.converged = true; // TODO - validate?
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
