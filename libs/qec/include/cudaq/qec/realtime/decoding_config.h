/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace cudaq::qec::decoding::config {

struct srelay_bp_config {
  std::optional<std::size_t> pre_iter;
  std::optional<std::size_t> num_sets;
  std::optional<std::string> stopping_criterion;
  std::optional<std::size_t> stop_nconv;

  bool operator==(const srelay_bp_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static srelay_bp_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct nv_qldpc_decoder_config {
  std::optional<bool> use_sparsity;
  std::optional<double> error_rate;
  std::optional<std::vector<double>> error_rate_vec;
  std::optional<int> max_iterations;
  std::optional<int> n_threads;
  std::optional<bool> use_osd;
  std::optional<int> osd_method;
  std::optional<int> osd_order;
  std::optional<int> bp_batch_size;
  std::optional<int> osd_batch_size;
  std::optional<int> iter_per_check;
  std::optional<double> clip_value;
  std::optional<int> bp_method;
  std::optional<double> scale_factor;
  std::optional<std::string> proc_float;
  std::optional<double> gamma0;
  std::optional<std::vector<double>> gamma_dist;
  std::optional<std::vector<std::vector<double>>> explicit_gammas;
  std::optional<srelay_bp_config> srelay_config;
  std::optional<int> bp_seed;
  std::optional<int> composition;
  // When set with a positive clip_value, forces the plugin's fixed-point
  // arithmetic path so corrections are reproducible across runs and across
  // hardware revisions. Required for the dispatcher-contract test which
  // compares decoded corrections against a recorded fixture byte-by-byte.
  std::optional<bool> repeatable;

  bool operator==(const nv_qldpc_decoder_config &) const = default;
  // opt_results is currently not supported for real-time decoding.

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static nv_qldpc_decoder_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct multi_error_lut_config {
  std::optional<int> lut_error_depth;
  // error_rate_vec is currently not supported for real-time decoding.
  // opt_results is currently not supported for real-time decoding.

  bool operator==(const multi_error_lut_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static multi_error_lut_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct single_error_lut_config {
  bool operator==(const single_error_lut_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static single_error_lut_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct pymatching_config {
  std::optional<std::vector<double>> error_rate_vec;
  std::optional<std::string> merge_strategy;

  bool operator==(const pymatching_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static pymatching_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct chromobius_config {
  std::optional<bool> drop_mobius_errors_involving_remnant_errors;
  std::optional<bool> ignore_decomposition_failures;
  std::optional<bool> include_coords_in_mobius_dem;
  std::optional<bool> return_weight;
  std::optional<bool> write_mobius_match_to_stderr;

  bool operator==(const chromobius_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static chromobius_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct trt_decoder_config {
  std::optional<std::string> onnx_load_path;
  std::optional<std::string> engine_load_path;
  std::optional<std::string> engine_save_path;
  std::optional<std::string> precision;
  std::optional<std::size_t> memory_workspace;
  std::optional<std::size_t> batch_size;
  std::optional<bool> use_cuda_graph;
  std::optional<std::string> global_decoder;
  /// Constructor parameters for the global decoder named by
  /// `global_decoder`. Keys are validated against the parameter schema that
  /// decoder registered (see cudaq/qec/decoder_config_schema.h); no decoder
  /// types are hardcoded here.
  std::optional<cudaqx::heterogeneous_map> global_decoder_params;

  __attribute__((visibility("default"))) bool
  operator==(const trt_decoder_config &) const;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static trt_decoder_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

struct sliding_window_config {
  std::optional<std::size_t> window_size;
  std::optional<std::size_t> step_size;
  std::optional<std::size_t> num_syndromes_per_round;
  std::optional<bool> straddle_start_round;
  std::optional<bool> straddle_end_round;
  std::vector<double> error_rate_vec;
  std::string inner_decoder_name;

  /// Constructor parameters for the inner decoder named by
  /// `inner_decoder_name`. Keys are validated against the parameter schema
  /// that decoder registered; no decoder types are hardcoded here.
  cudaqx::heterogeneous_map inner_decoder_params;

  __attribute__((visibility("default"))) bool
  operator==(const sliding_window_config &) const;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  to_heterogeneous_map() const;

  __attribute__((visibility("default"))) static sliding_window_config
  from_heterogeneous_map(const cudaqx::heterogeneous_map &map);
};

/// Transport type for a decoder session.
/// cpu_roce: CpuRoceTransceiver / SoftRoCE (dev, CI, no GPU required)
/// gpu_roce: GpuRoceTransceiver / DOCA (production, real ConnectX)
enum class DecoderTransport { cpu_roce, gpu_roce };

/// @brief Decoder-specific constructor arguments, stored as a
/// `cudaqx::heterogeneous_map` -- the form every decoder's constructor
/// consumes. YAML conversion is driven by the decoder's registered
/// `decoder_schema` (see cudaq/qec/decoder_config_schema.h), so out-of-tree
/// decoders participate without any framework changes.
///
/// The typed convenience structs above (`nv_qldpc_decoder_config`, ...) can
/// still be assigned directly; they are converted through their
/// `to_heterogeneous_map()`. Use `as<T>()` to recover a typed view.
class decoder_custom_args_t {
public:
  decoder_custom_args_t() = default;
  decoder_custom_args_t(const cudaqx::heterogeneous_map &m) : map_(m) {}
  template <typename T,
            typename = std::void_t<
                decltype(std::declval<const T &>().to_heterogeneous_map())>>
  decoder_custom_args_t(const T &cfg) : map_(cfg.to_heterogeneous_map()) {}

  decoder_custom_args_t &operator=(const cudaqx::heterogeneous_map &m) {
    map_ = m;
    return *this;
  }
  template <typename T,
            typename = std::void_t<
                decltype(std::declval<const T &>().to_heterogeneous_map())>>
  decoder_custom_args_t &operator=(const T &cfg) {
    map_ = cfg.to_heterogeneous_map();
    return *this;
  }

  /// Recover a typed view (e.g. `as<nv_qldpc_decoder_config>()`).
  template <typename T>
  T as() const {
    return T::from_heterogeneous_map(map_);
  }

  cudaqx::heterogeneous_map &map() { return map_; }
  const cudaqx::heterogeneous_map &map() const { return map_; }
  bool empty() const { return map_.empty(); }

  /// Deep equality over the canonical custom-args value kinds.
  __attribute__((visibility("default"))) bool
  operator==(const decoder_custom_args_t &other) const;

private:
  cudaqx::heterogeneous_map map_;
};

/// @brief Configuration structure for decoder options.
struct decoder_config {
  int64_t id = 0;
  std::string type;
  /// Transport used to receive syndromes and send corrections for this decoder.
  /// Defaults to cpu_roce.  Set to gpu_roce for decoders where syndrome bits
  /// are DMA'd directly to GPU VRAM (e.g. nv_qldpc_decoder with RelayBP).
  DecoderTransport transport = DecoderTransport::cpu_roce;
  uint64_t block_size = 0;
  uint64_t syndrome_size = 0;
  std::vector<std::int64_t> H_sparse;
  std::vector<std::int64_t> O_sparse;
  std::vector<std::int64_t> D_sparse;
  decoder_custom_args_t decoder_custom_args;

  bool operator==(const decoder_config &) const = default;

  __attribute__((visibility("default"))) cudaqx::heterogeneous_map
  decoder_custom_args_to_heterogeneous_map() const {
    return decoder_custom_args.map();
  }

  __attribute__((visibility("default"))) void
  set_decoder_custom_args_from_heterogeneous_map(
      const cudaqx::heterogeneous_map &map) {
    decoder_custom_args = map;
  }

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static decoder_config
  from_yaml_str(const std::string &yaml_str);
};

class multi_decoder_config {
public:
  std::vector<decoder_config> decoders;

  bool operator==(const multi_decoder_config &) const = default;

  __attribute__((visibility("default"))) std::string
  to_yaml_str(int column_wrap = 80);
  __attribute__((visibility("default"))) static multi_decoder_config
  from_yaml_str(const std::string_view yaml_str);
};

/// @brief Configure the decoders (`multi_decoder_config` variant). This
/// function configures both local decoders, and if running on remote target
/// hardware, will submit the configuration to the remote target for further
/// processing.
/// @param config The configuration to use.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders(multi_decoder_config &config);

/// @brief Configure the decoders from a file. This function configures both
/// local decoders, and if running on remote target hardware, will submit the
/// configuration to the remote target for further processing.
/// @param config_file The file to read the configuration from.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders_from_file(const char *config_file);

/// @brief Configure the decoders from a string. This function configures both
/// local decoders, and if running on remote target hardware, will submit the
/// configuration to the remote target for further processing.
/// @param config_str The string to read the configuration from.
/// @return 0 on success, non-zero on failure.
__attribute__((visibility("default"))) int
configure_decoders_from_str(const char *config_str);

/// @brief Finalize the decoders. This function finalizes local decoders.
__attribute__((visibility("default"))) void finalize_decoders();

/// @brief Return the most recently passed multi_decoder_config, or an empty
/// pointer if configure_decoders() has not been called in this process.
/// Used by the decoding-server DeviceCallService plugin to build
/// DecodingSessions on the in-process host_dispatch path without requiring
/// CUDAQ_QEC_DECODER_CONFIG.  Returns shared ownership: a concurrent
/// configure_decoders() replaces the stored config but cannot free it out
/// from under the caller.
__attribute__((visibility("default")))
std::shared_ptr<const multi_decoder_config>
last_configured_multi_decoder_config();

} // namespace cudaq::qec::decoding::config
