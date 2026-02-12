/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"
#include "stim.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/pcm_utils.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

namespace cudaq::qec {

/// @brief This is a wrapper around the PyMatching library that implements the
/// MWPM decoder.
class pymatching : public decoder {
private:
  pm::UserGraph user_graph;

  // Input parameters
  std::vector<double> error_rate_vec;
  pm::MERGE_STRATEGY merge_strategy_enum = pm::MERGE_STRATEGY::DISALLOW;
  bool enable_correlations = false;

  // Map of edge pairs to column indices. This does not seem particularly
  // efficient.
  std::map<std::pair<int64_t, int64_t>, size_t> edge2col_idx;

  // Helper function to make a canonical edge from two nodes.
  std::pair<int64_t, int64_t> make_canonical_edge(int64_t node1,
                                                  int64_t node2) {
    return std::make_pair(std::min(node1, node2), std::max(node1, node2));
  }

public:
  pymatching(const cudaq::qec::sparse_binary_matrix &H,
             const cudaqx::heterogeneous_map &params)
      : decoder(H) {

    if (params.contains("error_rate_vec")) {
      error_rate_vec = params.get<std::vector<double>>("error_rate_vec");
      if (error_rate_vec.size() != block_size) {
        throw std::runtime_error("error_rate_vec must be of size block_size");
      }
      // Validate that the values in the error_rate_vec are between 0 and 0.5.
      // Values > 0.5 would have negative LLR, which is not supported by
      // PyMatching.
      for (auto error_rate : error_rate_vec) {
        if (error_rate <= 0.0 || error_rate > 0.5) {
          throw std::runtime_error(
              "error_rate_vec value is out of range (0, 0.5]");
        }
      }
    }

    if (params.contains("merge_strategy")) {
      std::string merge_strategy = params.get<std::string>("merge_strategy");
      if (merge_strategy == "disallow") {
        merge_strategy_enum = pm::MERGE_STRATEGY::DISALLOW;
      } else if (merge_strategy == "independent") {
        merge_strategy_enum = pm::MERGE_STRATEGY::INDEPENDENT;
      } else if (merge_strategy == "smallest_weight") {
        merge_strategy_enum = pm::MERGE_STRATEGY::SMALLEST_WEIGHT;
      } else if (merge_strategy == "keep_original") {
        merge_strategy_enum = pm::MERGE_STRATEGY::KEEP_ORIGINAL;
      } else if (merge_strategy == "replace") {
        merge_strategy_enum = pm::MERGE_STRATEGY::REPLACE;
      } else {
        throw std::runtime_error(
            "merge_strategy must be one of: disallow, independent, "
            "smallest_weight, keep_original, replace");
      }
    }

    if (params.contains("enable_correlations")) {
      enable_correlations = params.get<bool>("enable_correlations");
    }

    std::vector<std::vector<std::uint32_t>> H_e2d = H.to_nested_csc();
    std::vector<size_t> observables;
    std::size_t col_idx = 0;

    if (enable_correlations) {
      std::cout << "Using correlated error model." << std::endl;
      // Verify that dem_string is provided.
      if (!params.contains("dem_string")) {
        throw std::runtime_error(
            "dem_string must be provided when enable_correlations is true");
      }
      std::string dem_string = params.get<std::string>("dem_string");
      auto dem = stim::DetectorErrorModel(dem_string);
      user_graph = pm::detector_error_model_to_user_graph(
          dem, true, pm::NUM_DISTINCT_WEIGHTS);
      // Verify that the user graph has the same number of nodes as the check
      // matrix.
      if (user_graph.get_num_edges() != H.num_cols()) {
        throw std::runtime_error(
            "Number of edges in the user graph does not match the number of "
            "columns in the check matrix");
      }
      for (std::size_t col = 0; col < block_size; col++) {
        double weight = 1.0;
        if (H_e2d[col].size() == 2) {
          edge2col_idx[make_canonical_edge(H_e2d[col][0], H_e2d[col][1])] =
              col_idx;
        } else if (H_e2d[col].size() == 1) {
          edge2col_idx[make_canonical_edge(H_e2d[col][0], -1)] = col_idx;
        }
        col_idx++;
      }
      return;
    }

    // Non-correlated error model.
    user_graph = pm::UserGraph(H.num_rows());

    for (std::size_t col = 0; col < block_size; col++) {
      double weight = 1.0;
      if (col_idx < error_rate_vec.size()) {
        weight = -std::log(error_rate_vec[col_idx] /
                           (1.0 - error_rate_vec[col_idx]));
      }
      if (H_e2d[col].size() == 2) {
        edge2col_idx[make_canonical_edge(H_e2d[col][0], H_e2d[col][1])] =
            col_idx;
        user_graph.add_or_merge_edge(H_e2d[col][0], H_e2d[col][1], observables,
                                     weight, 0.0, merge_strategy_enum);
      } else if (H_e2d[col].size() == 1) {
        edge2col_idx[make_canonical_edge(H_e2d[col][0], -1)] = col_idx;
        user_graph.add_or_merge_boundary_edge(H_e2d[col][0], observables,
                                              weight, 0.0, merge_strategy_enum);
      } else {
        throw std::runtime_error("Invalid column in H: " + std::to_string(col) +
                                 " has " + std::to_string(H_e2d[col].size()) +
                                 " ones. Must have 1 or 2 ones.");
      }
      col_idx++;
    }
  }

  /// @brief Decode the syndrome using the MWPM decoder.
  /// @param syndrome The syndrome to decode.
  /// @return The decoder result.
  /// @throws std::runtime_error if no matching solution is found, or
  /// std::out_of_range if an edge is not found in the edge2col_idx map.
  virtual decoder_result decode(const std::vector<float_t> &syndrome) {
    decoder_result result{false, std::vector<float_t>(block_size, 0.0)};
    auto &mwpm = user_graph.get_mwpm_with_search_graph();
    std::vector<int64_t> edges;
    std::vector<uint64_t> detection_events;
    detection_events.reserve(syndrome.size());
    for (size_t i = 0; i < syndrome.size(); i++)
      if (syndrome[i] > 0.5)
        detection_events.push_back(i);
    if (enable_correlations) {
      pm::decode_detection_events_to_edges_with_edge_correlations(
          mwpm, detection_events, edges);
    } else {
      pm::decode_detection_events_to_edges(mwpm, detection_events, edges);
    }
    // Loop over the edge pairs
    assert(edges.size() % 2 == 0);
    for (size_t i = 0; i < edges.size(); i += 2) {
      auto edge = make_canonical_edge(edges.at(i), edges.at(i + 1));
      auto col_idx = edge2col_idx.at(edge);
      result.result[col_idx] = 1.0;
    }
    // An exception is thrown if no matching solution is found, so we can just
    // set converged to true.
    result.converged = true;
    return result;
  }

  virtual ~pymatching() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      pymatching, static std::unique_ptr<decoder> create(
                      const cudaq::qec::sparse_binary_matrix &H,
                      const cudaqx::heterogeneous_map &params) {
        return std::make_unique<pymatching>(H, params);
      })
};

CUDAQ_REGISTER_TYPE(pymatching)

} // namespace cudaq::qec
