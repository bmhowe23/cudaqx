/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * Standalone benchmark: builds a detector matrix via                          *
 * generate_timelike_sparse_detector_matrix, then times preprocess_all.       *
 ******************************************************************************/

#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/realtime/gpu_kernels.cuh"
#include "cudaq/qec/realtime/sparse_to_csr.h"

#include "bench_preprocess_all_data.h"

using realtime_float_t = cudaq::qec::realtime::float_t;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// Single template kernel: DoPreprocess == false => baseline (grid.sync overhead
// only); DoPreprocess == true => full (preprocess_all + sync). Baseline is
// subtracted from full to report preprocess_all-only time.
template <bool DoPreprocess>
__global__ void launch_preprocess_all_impl(
    const uint8_t *measurements, const uint32_t *D_row_ptr,
    const uint32_t *D_col_idx, realtime_float_t *soft_syndrome,
    std::size_t num_detectors, unsigned long long *d_cycle_elapsed) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();

  grid.sync();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_cycle_elapsed = clock64();
  }
  grid.sync();

  if constexpr (DoPreprocess) {
    cudaq::qec::realtime::preprocess_all(measurements, D_row_ptr, D_col_idx,
                                         soft_syndrome, num_detectors);
  }

  grid.sync();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long start_cycle = *d_cycle_elapsed;
    unsigned long long end_cycle = clock64();
    *d_cycle_elapsed = end_cycle - start_cycle;
  }
}

int main(int argc, char **argv) {
  std::uint32_t num_syndromes_per_round = 1000;
  std::uint32_t num_rounds = 20;
  bool include_first_round = false;
  int num_warmup = 3;
  int num_timed = 100;
  unsigned int grid_size_arg = 0;  // 0 = compute from num_detectors
  unsigned int block_size_arg = 0; // 0 = use default 256

  if (argc >= 4) {
    num_syndromes_per_round = static_cast<std::uint32_t>(std::atoi(argv[1]));
    num_rounds = static_cast<std::uint32_t>(std::atoi(argv[2]));
    include_first_round = (std::atoi(argv[3]) != 0);
  }
  if (argc >= 6) {
    num_warmup = std::atoi(argv[4]);
    num_timed = std::atoi(argv[5]);
  }
  if (argc >= 7) {
    grid_size_arg = static_cast<unsigned int>(std::atoi(argv[6]));
  }
  if (argc >= 8) {
    block_size_arg = static_cast<unsigned int>(std::atoi(argv[7]));
  }

  // Count how many -1's are in D_sparse.
  int num_rows = 0;
  int num_cols = 0;
  int num_cols_this_row = 0;
  int max_cols_in_row = 0;
  for (auto d : D_sparse_canned) {
    if (d == -1) {
      num_rows++;
      max_cols_in_row = std::max(max_cols_in_row, num_cols_this_row);
      num_cols_this_row = 0;
    } else {
      num_cols_this_row++;
    }
    num_cols = std::max(num_cols, static_cast<int>(d));
  }
  num_cols++;
  printf("Detector matrix: %d rows, %d cols\n", num_rows, num_cols);
  if (num_syndromes_per_round == 0 && num_rounds == 0 &&
      include_first_round == false) {
    printf(
        "Using canned detector matrix of size %d x %d, max cols per row: %d\n",
        num_rows, num_cols, max_cols_in_row);
  } else {
    std::printf("Building detector matrix: syndromes_per_round=%u, rounds=%u, "
                "include_first_round=%d\n",
                num_syndromes_per_round, num_rounds, include_first_round);
  }

  std::size_t num_measurements = num_syndromes_per_round * num_rounds;
  std::vector<std::int64_t> D_sparse;
  if (num_syndromes_per_round == 0 && num_rounds == 0 &&
      include_first_round == false) {
    D_sparse = D_sparse_canned;
    num_measurements = num_cols;
  } else {
    D_sparse = cudaq::qec::generate_timelike_sparse_detector_matrix(
        num_syndromes_per_round, num_rounds, include_first_round);
  }

  std::vector<std::uint32_t> D_row_ptr, D_col_idx;
  std::size_t num_detectors =
      cudaq::qec::realtime::sparse_vec_to_csr(D_sparse, D_row_ptr, D_col_idx);

  std::printf("Detector matrix: %zu detectors, %zu measurements, %zu nnz\n",
              num_detectors, num_measurements, D_col_idx.size());

  uint8_t *d_measurements = nullptr;
  uint32_t *d_D_row_ptr = nullptr;
  uint32_t *d_D_col_idx = nullptr;
  realtime_float_t *d_soft_syndrome = nullptr;
  unsigned long long *d_cycle_elapsed = nullptr;

  CUDA_CHECK(cudaMalloc(&d_measurements, num_measurements));
  CUDA_CHECK(cudaMalloc(&d_D_row_ptr, D_row_ptr.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_D_col_idx, D_col_idx.size() * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMalloc(&d_soft_syndrome, num_detectors * sizeof(realtime_float_t)));
  CUDA_CHECK(cudaMalloc(&d_cycle_elapsed, sizeof(unsigned long long)));

  std::vector<uint8_t> h_measurements(num_measurements, 0);
  CUDA_CHECK(cudaMemcpy(d_measurements, h_measurements.data(), num_measurements,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D_row_ptr, D_row_ptr.data(),
                        D_row_ptr.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D_col_idx, D_col_idx.data(),
                        D_col_idx.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  std::size_t block_size = (block_size_arg != 0) ? block_size_arg : 256;
  std::size_t grid_size = (grid_size_arg != 0)
                              ? grid_size_arg
                              : (num_detectors + block_size - 1) / block_size;
  if (grid_size > 65535)
    grid_size = 65535;
  std::printf("Launch config: grid_size=%zu, block_size=%zu (cooperative)\n",
              grid_size, block_size);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int cooperative_supported = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cooperative_supported,
                                    cudaDevAttrCooperativeLaunch, device));
  if (!cooperative_supported) {
    std::fprintf(stderr,
                 "This device does not support cooperative kernel launch.\n");
    std::exit(1);
  }

  void *args[] = {&d_measurements,  &d_D_row_ptr,   &d_D_col_idx,
                  &d_soft_syndrome, &num_detectors, &d_cycle_elapsed};
  dim3 grid(static_cast<unsigned>(grid_size));
  dim3 block(static_cast<unsigned>(block_size));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  double clock_khz = static_cast<double>(prop.clockRate);
  double clock_mhz = clock_khz / 1000.0;
  std::printf("GPU: %s, clock: %.1f MHz\n", prop.name, clock_mhz);
  // cycles / (cycles per second) = seconds; * 1000 = ms
  double cycles_to_ms = 1000.0 / (clock_khz * 1000.0);

  // Baseline: same kernel structure but no preprocess_all (grid.sync overhead
  // only).
  for (int i = 0; i < num_warmup; ++i) {
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void *)launch_preprocess_all_impl<false>, grid, block, args));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  unsigned long long total_baseline_cycles = 0;
  for (int i = 0; i < num_timed; ++i) {
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void *)launch_preprocess_all_impl<false>, grid, block, args));
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long h_cycles = 0;
    CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycle_elapsed,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    total_baseline_cycles += h_cycles;
  }
  double baseline_cycles =
      static_cast<double>(total_baseline_cycles) / num_timed;

  // Full: preprocess_all between the two grid.sync()s.
  for (int i = 0; i < num_warmup; ++i) {
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void *)launch_preprocess_all_impl<true>, grid, block, args));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  unsigned long long total_full_cycles = 0;
  for (int i = 0; i < num_timed; ++i) {
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void *)launch_preprocess_all_impl<true>, grid, block, args));
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long h_cycles = 0;
    CUDA_CHECK(cudaMemcpy(&h_cycles, d_cycle_elapsed,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    total_full_cycles += h_cycles;
  }
  double full_cycles = static_cast<double>(total_full_cycles) / num_timed;

  double baseline_ms = baseline_cycles * cycles_to_ms;
  double full_ms = full_cycles * cycles_to_ms;
  double ms = (full_cycles - baseline_cycles) * cycles_to_ms;
  if (ms < 0.0)
    ms = 0.0;

  std::printf(
      "preprocess_all (device timing, grid.sync overhead subtracted):\n");
  std::printf("  baseline (grid.sync only): %.4f ms, %.2f us\n", baseline_ms,
              baseline_ms * 1000.0);
  std::printf("  full (preprocess_all + sync): %.4f ms, %.2f us\n", full_ms,
              full_ms * 1000.0);
  std::printf("  preprocess_all only: %.4f ms per call (%d iterations)\n", ms,
              num_timed);
  std::printf("  preprocess_all only: %.2f us per call\n", ms * 1000.0);

  cudaFree(d_measurements);
  cudaFree(d_D_row_ptr);
  cudaFree(d_D_col_idx);
  cudaFree(d_soft_syndrome);
  cudaFree(d_cycle_elapsed);

  return 0;
}
