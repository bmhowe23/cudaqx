/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/qec/patch.h"

namespace cudaq::qec::surface_code {

__qpu__ void x(patch logicalQubit) { x(logicalQubit.data); }
__qpu__ void z(patch logicalQubit) { z(logicalQubit.data); }

__qpu__ void cx(patch logicalQubitA, patch logicalQubitB) {
  for (std::size_t i = 0; i < logicalQubitA.data.size(); i++) {
    x<cudaq::ctrl>(logicalQubitA.data[i], logicalQubitB.data[i]);
  }
}

__qpu__ void cz(patch logicalQubitA, patch logicalQubitB) {
  for (std::size_t i = 0; i < logicalQubitA.data.size(); i++) {
    z<cudaq::ctrl>(logicalQubitA.data[i], logicalQubitB.data[i]);
  }
}

// Transversal state prep, turn on stabilizers after these ops
__qpu__ void prep0(patch logicalQubit) {
  for (std::size_t i = 0; i < logicalQubit.data.size(); i++)
    reset(logicalQubit.data[i]);
}

__qpu__ void prep1(patch logicalQubit) {
  prep0(logicalQubit);
  x(logicalQubit.data);
}

__qpu__ void prepp(patch logicalQubit) {
  prep0(logicalQubit);
  h(logicalQubit.data);
}

__qpu__ void prepm(patch logicalQubit) {
  prep0(logicalQubit);
  x(logicalQubit.data);
  h(logicalQubit.data);
}

__qpu__ std::vector<cudaq::measure_result>
stabilizer(patch logicalQubit, const std::vector<std::size_t> &x_stabilizers,
           const std::vector<std::size_t> &z_stabilizers) {
  h(logicalQubit.ancx);
  for (std::size_t xi = 0; xi < logicalQubit.ancx.size(); ++xi)
    for (std::size_t di = 0; di < logicalQubit.data.size(); ++di)
      if (x_stabilizers[xi * logicalQubit.data.size() + di] == 1)
        cudaq::x<cudaq::ctrl>(logicalQubit.ancx[xi], logicalQubit.data[di]);
  h(logicalQubit.ancx);

  // Now apply z_stabilizer circuit
  for (size_t zi = 0; zi < logicalQubit.ancz.size(); ++zi)
    for (size_t di = 0; di < logicalQubit.data.size(); ++di)
      if (z_stabilizers[zi * logicalQubit.data.size() + di] == 1)
        cudaq::x<cudaq::ctrl>(logicalQubit.data[di], logicalQubit.ancz[zi]);

  // S = (S_X, S_Z), (x flip syndromes, z flip syndromes).
  // x flips are triggered by z-stabilizers (ancz)
  // z flips are triggered by x-stabilizers (ancx)
  auto results = mz(logicalQubit.ancz, logicalQubit.ancx);

  for (std::size_t i = 0; i < logicalQubit.ancx.size(); i++)
    reset(logicalQubit.ancx[i]);
  for (std::size_t i = 0; i < logicalQubit.ancz.size(); i++)
    reset(logicalQubit.ancz[i]);

  return results;
}

} // namespace cudaq::qec::surface_code
