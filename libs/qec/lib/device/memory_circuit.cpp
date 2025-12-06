/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "memory_circuit.h"

namespace cudaq::qec {

__qpu__ void __memory_circuit_stabs(
    cudaq::qview<> data, cudaq::qview<> xstab_anc, cudaq::qview<> zstab_anc,
    const code::stabilizer_round &stabilizer_round,
    const code::one_qubit_encoding &statePrep, std::size_t numRounds,
    bool keep_x_stabilizers, bool keep_z_stabilizers,
    const std::vector<std::size_t> &x_stabilizers,
    const std::vector<std::size_t> &z_stabilizers) {
  // Create the logical patch
  patch logical(data, xstab_anc, zstab_anc);

  // Prepare the initial state fault tolerantly
  statePrep({data, xstab_anc, zstab_anc});

  std::int64_t zstab_size_i64 = static_cast<std::int64_t>(zstab_anc.size());
  std::int64_t xstab_size_i64 = static_cast<std::int64_t>(xstab_anc.size());

  // Generate syndrome data
  for (std::size_t round = 0; round < numRounds; round++) {
    // Run the stabilizer round, generate the syndrome measurements
    auto syndrome = stabilizer_round(logical, x_stabilizers, z_stabilizers);
    std::int64_t syndrome_size_i64 = static_cast<std::int64_t>(syndrome.size());
    if (round == 0) {
      if (keep_z_stabilizers) {
        for (std::int64_t i = 0; i < zstab_size_i64; i++) {
          cudaq::detector(-syndrome_size_i64 + i, 2);
        }
      }
      if (keep_x_stabilizers) {
        for (std::int64_t i = 0; i < xstab_size_i64; i++) {
          cudaq::detector(-xstab_size_i64 + i, 2);
        }
      }
    } else {
      if (keep_z_stabilizers) {
        for (std::int64_t i = 0; i < zstab_size_i64; i++) {
          cudaq::detector(-2 * syndrome_size_i64 + i, -syndrome_size_i64 + i);
        }
      }
      if (keep_x_stabilizers) {
        for (std::int64_t i = 0; i < xstab_size_i64; i++) {
          cudaq::detector(-2 * syndrome_size_i64 + zstab_size_i64 + i,
                          -syndrome_size_i64 + zstab_size_i64 + i);
        }
      }
    }
  }
}

__qpu__ void memory_circuit_mz(const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t numData, std::size_t numAncx,
                               std::size_t numAncz, std::size_t numRounds,
                               bool keep_x_stabilizers, bool keep_z_stabilizers,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers,
                               bool include_final_round_detectors) {

  // Allocate the data and ancilla qubits
  cudaq::qvector data(numData), xstab_anc(numAncx), zstab_anc(numAncz);

  // Persists ancilla measures
  __memory_circuit_stabs(data, xstab_anc, zstab_anc, stabilizer_round,
                         statePrep, numRounds, keep_x_stabilizers,
                         keep_z_stabilizers, x_stabilizers, z_stabilizers);

  auto dataResults = mz(data);

  // Add the detectors after the final round.
  if (include_final_round_detectors) {
    // For each ancz, find the data qubits that support it.
    for (size_t zi = 0; zi < numAncz; ++zi) {
      int num_dets_indices_required = 1; // 1 for the stabilizer
      for (size_t di = 0; di < numData; ++di)
        if (z_stabilizers[zi * numData + di] == 1)
          num_dets_indices_required++; // 1 for each data qubit
      std::vector<std::int64_t> rec(num_dets_indices_required);
      int count = 0;
      for (size_t di = 0; di < numData; ++di) {
        if (z_stabilizers[zi * numData + di] == 1) {
          // This stabilizer is supported by data qubit di. Convert di to a
          // relative measurement index.
          rec[count++] = di - numData;
        }
      }
      // Now get the z stabilizer measurement index. We must skip over the x
      // stabilizer measurements.
      rec[count++] = -numData - numAncx - numAncz + zi;
      cudaq::detector(rec);
    }
  }
}

__qpu__ void memory_circuit_mx(const code::stabilizer_round &stabilizer_round,
                               const code::one_qubit_encoding &statePrep,
                               std::size_t numData, std::size_t numAncx,
                               std::size_t numAncz, std::size_t numRounds,
                               bool keep_x_stabilizers, bool keep_z_stabilizers,
                               const std::vector<std::size_t> &x_stabilizers,
                               const std::vector<std::size_t> &z_stabilizers,
                               bool include_final_round_detectors) {

  // Allocate the data and ancilla qubits
  cudaq::qvector data(numData), xstab_anc(numAncx), zstab_anc(numAncz);

  // Persists ancilla measures
  __memory_circuit_stabs(data, xstab_anc, zstab_anc, stabilizer_round,
                         statePrep, numRounds, keep_x_stabilizers,
                         keep_z_stabilizers, x_stabilizers, z_stabilizers);

  h(data);
  auto dataResults = mz(data);

  // Add the detectors after the final round.
  // TODO - verify this logic is correct.
  if (include_final_round_detectors) {
    // For each ancx, find the data qubits that support it.
    for (size_t xi = 0; xi < numAncx; ++xi) {
      int num_dets_indices_required = 1; // 1 for the stabilizer
      for (size_t di = 0; di < numData; ++di)
        if (x_stabilizers[xi * numData + di] == 1)
          num_dets_indices_required++; // 1 for each data qubit
      std::vector<std::int64_t> rec(num_dets_indices_required);
      int count = 0;
      for (size_t di = 0; di < numData; ++di) {
        if (x_stabilizers[xi * numData + di] == 1) {
          // This stabilizer is supported by data qubit di. Convert di to a
          // relative measurement index.
          rec[count++] = di - numData;
        }
      }
      // Now get the x stabilizer measurement index. The x stabilizers are the
      // 2nd half of the per-round syndrome, so we do not need to be aware of
      // the z stabilizers.
      rec[count++] = -numData - numAncx + xi;
      cudaq::detector(rec);
    }
  }
}
} // namespace cudaq::qec
