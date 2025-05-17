/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/detector_error_model.h"

namespace cudaq::qec {

std::size_t detector_error_model::num_detectors() {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[0];
  return 0;
}

std::size_t detector_error_model::num_error_mechanisms() {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[1];
  return 0;
}

// FIXME - likely not correct
std::size_t detector_error_model::num_measurements() {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[0];
  return 0;
}

} // namespace cudaq::qec