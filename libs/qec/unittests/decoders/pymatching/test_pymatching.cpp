/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <vector>

// Test parameter validation function
TEST(PyMatchingDecoder, checkAPI) {
  using cudaq::qec::float_t;

  // Use Hx from the [7,1,3] Steane code from
  // https://en.wikipedia.org/wiki/Steane_code.
  std::size_t block_size = 7;
  std::size_t syndrome_size = 3;
  cudaqx::heterogeneous_map custom_args;

  std::vector<uint8_t> H_vec = {0, 0, 0, 1, 1, 1, 1,  // IIIXXXX
                                0, 1, 1, 0, 0, 1, 1,  // IXXIIXX
                                1, 0, 1, 0, 1, 0, 1}; // XIXIXIX
  cudaqx::tensor<uint8_t> H;
  H.copy(H_vec.data(), {syndrome_size, block_size});
  auto d = cudaq::qec::decoder::get("pymatching", H, custom_args);
}
