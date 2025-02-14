/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/experiments.h"

#include "device/memory_circuit.h"

using namespace cudaqx;

namespace cudaq::qec {

namespace details {
auto __sample_code_capacity(const cudaqx::tensor<uint8_t> &H,
                            std::size_t nShots, double error_probability,
                            unsigned seed) {
  // init RNG
  std::mt19937 rng(seed);
  std::bernoulli_distribution dist(error_probability);

  // Each row is a shot
  // Each row elem is a 1 if error, 0 else.
  cudaqx::tensor<uint8_t> data({nShots, H.shape()[1]});
  cudaqx::tensor<uint8_t> syndromes({nShots, H.shape()[0]});

  std::vector<uint8_t> bits(nShots * H.shape()[1]);
  std::generate(bits.begin(), bits.end(), [&]() { return dist(rng); });

  data.copy(bits.data(), data.shape());

  // Syn = D * H^T
  // [n,s] = [n,d]*[d,s]
  syndromes = data.dot(H.transpose()) % 2;

  return std::make_tuple(syndromes, data);
}
} // namespace details

// Single shot version
cudaqx::tensor<uint8_t> generate_random_bit_flips(size_t numBits,
                                                  double error_probability) {
  // init RNG
  std::random_device rd;
  std::mt19937 rng(rd());
  std::bernoulli_distribution dist(error_probability);

  // Each row is a shot
  // Each row elem is a 1 if error, 0 else.
  cudaqx::tensor<uint8_t> data({numBits});
  std::vector<uint8_t> bits(numBits);
  std::generate(bits.begin(), bits.end(), [&]() { return dist(rng); });

  data.copy(bits.data(), data.shape());
  return data;
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const cudaqx::tensor<uint8_t> &H, std::size_t nShots,
                     double error_probability, unsigned seed) {
  return details::__sample_code_capacity(H, nShots, error_probability, seed);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const cudaqx::tensor<uint8_t> &H, std::size_t nShots,
                     double error_probability) {
  return details::__sample_code_capacity(H, nShots, error_probability,
                                         std::random_device()());
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const code &code, std::size_t nShots,
                     double error_probability) {
  return sample_code_capacity(code.get_parity(), nShots, error_probability);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_code_capacity(const code &code, std::size_t nShots,
                     double error_probability, unsigned seed) {
  return sample_code_capacity(code.get_parity(), nShots, error_probability,
                              seed);
}

// Get the PCM
template <
    typename QuantumKernel, typename... Args,
    typename = std::enable_if_t<std::is_invocable_v<QuantumKernel, Args...>>>
cudaqx::tensor<uint8_t> get_pcm(cudaq::noise_model &noise,
                                QuantumKernel &&kernel,
                                Args &&...args) {

  cudaq::ExecutionContext ctx_pcm_size("pcm_size");
  ctx_pcm_size.noiseModel = &noise;
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_pcm_size);
  std::invoke(std::forward<QuantumKernel>(kernel), std::forward<Args>(args)...);
  platform.reset_exec_ctx();

  // No noise is affecting this circuit, so no PCM can be generated.
  if (!ctx_pcm_size.pcm_dimensions ||
      ctx_pcm_size.pcm_dimensions.value().first == 0 ||
      ctx_pcm_size.pcm_dimensions.value().second == 0)
    return cudaqx::tensor<uint8_t>();

  cudaq::ExecutionContext ctx_pcm("pcm");
  ctx_pcm.noiseModel = &noise;
  ctx_pcm.pcm_dimensions = ctx_pcm_size.pcm_dimensions;
  platform.set_exec_ctx(&ctx_pcm);
  std::invoke(std::forward<QuantumKernel>(kernel), std::forward<Args>(args)...);
  platform.reset_exec_ctx();
  
  // FIXME - strip out the data qubits from the syndrome qubits.
  // Also, do we need to XOR syndromes together?

  // Also FIXME, do we need to strip out rows that are all 0's?

  // Look for duplicate columns
  const auto pcm_as_strings = ctx_pcm.result.sequential_data();
  std::unordered_map<std::string, std::vector<std::size_t>> unique_syndromes;
  for (std::size_t ix = 0; auto &col : pcm_as_strings) {
    unique_syndromes[col].push_back(ix);
    ix++;
  }

  printf("pcm_dimensions is %lu %lu\n", ctx_pcm.pcm_dimensions.value().first,
         ctx_pcm.pcm_dimensions.value().second);

  for (std::size_t r = 0; r < ctx_pcm.pcm_dimensions.value().first; r++) {
    for (std::size_t c = 0; c < ctx_pcm.pcm_dimensions.value().second; c++) {
      printf("%c", pcm_as_strings[c][r] == '1' ? '1' : '.');
    }
    printf("\n");
  }

  // Now make a second pass, collapsing all the duplicates, adding the
  // probabilities for error mechanisms that produce the same syndrome.
  std::vector<std::size_t> collapsed2orig(unique_syndromes.size());
  std::vector<std::size_t> orig2collapsed(pcm_as_strings.size());
  std::vector<double> collapsed_prob;
  auto &pcm_probabilities = ctx_pcm.pcm_probabilities.value();
  collapsed_prob.reserve(unique_syndromes.size());
  auto num_rows = pcm_as_strings[0].size();
  auto num_cols = unique_syndromes.size();
  cudaqx::tensor<uint8_t> pcm({num_rows, num_cols});
  for (std::size_t ix = 0; auto &col : pcm_as_strings) {
    if (unique_syndromes[col].front() == ix) {
      double p_not = 1.0;
      // Loop over the indices of the error mechanisms that all share this
      // unique syndrome.
      for (auto ix2 : unique_syndromes[col]) {
        p_not *= (1.0 - pcm_probabilities[ix2]);
      }
      collapsed2orig[collapsed_prob.size()] = ix;
      orig2collapsed[ix] = collapsed_prob.size();
      for (std::size_t row = 0; row < num_rows; row++)
        pcm.at({row, collapsed_prob.size()}) = col[row] - '0';
      collapsed_prob.push_back(1.0 - p_not);
    }
    ix++;
  }

  pcm.dump();
  for (std::size_t r = 0; r < pcm.shape()[0]; r++) {
    for (std::size_t c = 0; c < pcm.shape()[1]; c++) {
      printf("%c", pcm.at({r, c}) ? '1' : '.');
    }
    printf("\n");
  }

  return pcm;
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation statePrep,
                      std::size_t numShots, std::size_t numRounds,
                      cudaq::noise_model &noise) {
  if (!code.contains_operation(statePrep))
    throw std::runtime_error(
        "sample_memory_circuit_error - requested state prep kernel not found.");

  auto &prep = code.get_operation<code::one_qubit_encoding>(statePrep);

  if (!code.contains_operation(operation::stabilizer_round))
    throw std::runtime_error("sample_memory_circuit error - no stabilizer "
                             "round kernel for this code.");

  auto &stabRound =
      code.get_operation<code::stabilizer_round>(operation::stabilizer_round);

  auto parity_x = code.get_parity_x();
  auto parity_z = code.get_parity_z();
  auto numData = code.get_num_data_qubits();
  auto numAncx = code.get_num_ancilla_x_qubits();
  auto numAncz = code.get_num_ancilla_z_qubits();

  std::vector<std::size_t> xVec(parity_x.data(),
                                parity_x.data() + parity_x.size());
  std::vector<std::size_t> zVec(parity_z.data(),
                                parity_z.data() + parity_z.size());

  std::size_t numRows = numShots * numRounds;
  std::size_t numCols = numAncx + numAncz;

  // Allocate the tensor data for the syndromes and data.
  cudaqx::tensor<uint8_t> syndromeTensor({numShots * numRounds, numCols});
  cudaqx::tensor<uint8_t> dataResults({numShots, numData});

  cudaq::sample_options opts;
  opts.shots = numShots;
  opts.noise = noise;
  opts.explicit_measurements = true;

  cudaq::sample_result result;

  // Run the memory circuit experiment
  if (statePrep == operation::prep0 || statePrep == operation::prep1) {
    auto x = get_pcm(noise, memory_circuit_mz, stabRound, prep, numData,
                     numAncx, numAncz, numRounds, xVec, zVec);
    // run z basis
    result = cudaq::sample(opts, memory_circuit_mz, stabRound, prep, numData,
                           numAncx, numAncz, numRounds, xVec, zVec);
  } else if (statePrep == operation::prepp || statePrep == operation::prepm) {
    // run x basis
    result = cudaq::sample(opts, memory_circuit_mx, stabRound, prep, numData,
                           numAncx, numAncz, numRounds, xVec, zVec);
  } else {
    throw std::runtime_error(
        "sample_memory_circuit_error - invalid requested state prep kernel.");
  }

  const auto bitstrings = result.sequential_data();
  const auto numColsBeforeData = numCols * numRounds;
  for (std::size_t shot = 0; shot < numShots; shot++)
    for (std::size_t d = 0; d < numData; d++)
      dataResults.at({shot, d}) = bitstrings[shot][numColsBeforeData + d] - '0';

  // First round, store bare syndrome measurement
  for (std::size_t shot = 0; shot < numShots; ++shot) {
    for (std::size_t col = 0; col < numCols; ++col) {
      std::size_t round = 0;
      std::size_t measIdx = shot * numRounds + round;
      syndromeTensor.at({measIdx, col}) = bitstrings[shot][col] - '0';
    }
  }

  // After first round, store syndrome flips
  // #pragma omp parallel for collapse(2)
  for (std::size_t shot = 0; shot < numShots; ++shot) {
    const auto &shotBitstr = bitstrings[shot];
    for (std::size_t round = 1; round < numRounds; ++round) {
      for (std::size_t col = 0; col < numCols; ++col) {
        std::size_t measIdx = shot * numRounds + round;
        syndromeTensor.at({measIdx, col}) =
            shotBitstr[round * numCols + col] ^
            shotBitstr[(round - 1) * numCols + col];
      }
    }
  }

  // Return the data.
  return std::make_tuple(syndromeTensor, dataResults);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, operation op, std::size_t numShots,
                      std::size_t numRounds) {
  cudaq::noise_model noise;
  return sample_memory_circuit(code, op, numShots, numRounds, noise);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds) {
  return sample_memory_circuit(code, operation::prep0, numShots, numRounds);
}

std::tuple<cudaqx::tensor<uint8_t>, cudaqx::tensor<uint8_t>>
sample_memory_circuit(const code &code, std::size_t numShots,
                      std::size_t numRounds, cudaq::noise_model &noise) {
  return sample_memory_circuit(code, operation::prep0, numShots, numRounds,
                               noise);
}

} // namespace cudaq::qec
