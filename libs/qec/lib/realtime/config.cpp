/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "realtime_decoding.h"
#include "cudaq/qec/decoder_config_payload.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <any>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>

// Helper function(s) to remove the optional wrapper from a type.
// Primary template for non-optional types
template <typename T>
struct remove_optional {
  using type = T;
};
// Partial specialization for std::optional<T>
template <typename T>
struct remove_optional<std::optional<T>> {
  using type = T;
};
// Alias
template <typename T>
using remove_optional_t = typename remove_optional<T>::type;

namespace cudaq::qec::decoding::config {

#define INSERT_ARG(arg_name)                                                   \
  do {                                                                         \
    if (this->arg_name.has_value()) {                                          \
      config_map.insert(#arg_name, this->arg_name.value());                    \
    }                                                                          \
  } while (false)

#define INSERT_ARG_PLAIN(arg_name)                                             \
  do {                                                                         \
    config_map.insert(#arg_name, this->arg_name);                              \
  } while (false)

#define GET_ARG(arg_name)                                                      \
  do {                                                                         \
    if (map.contains(#arg_name)) {                                             \
      config.arg_name =                                                        \
          map.get<remove_optional_t<decltype(config.arg_name)>>(#arg_name);    \
    }                                                                          \
  } while (false)

#define GET_ARG_PLAIN(arg_name)                                                \
  do {                                                                         \
    if (map.contains(#arg_name)) {                                             \
      config.arg_name =                                                        \
          map.get<std::decay_t<decltype(config.arg_name)>>(#arg_name);         \
    }                                                                          \
  } while (false)

// ------ relay_bp_config ------
cudaqx::heterogeneous_map srelay_bp_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;

  INSERT_ARG(pre_iter);
  INSERT_ARG(num_sets);
  INSERT_ARG(stopping_criterion);
  INSERT_ARG(stop_nconv);

  return config_map;
}

srelay_bp_config
srelay_bp_config::from_heterogeneous_map(const cudaqx::heterogeneous_map &map) {
  srelay_bp_config config;
  GET_ARG(pre_iter);
  GET_ARG(num_sets);
  GET_ARG(stopping_criterion);
  GET_ARG(stop_nconv);

  return config;
}

// ------ nv_qldpc_decoder_config ------
cudaqx::heterogeneous_map
nv_qldpc_decoder_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;

  INSERT_ARG(use_sparsity);
  INSERT_ARG(error_rate);
  INSERT_ARG(error_rate_vec);
  INSERT_ARG(max_iterations);
  INSERT_ARG(n_threads);
  INSERT_ARG(use_osd);
  INSERT_ARG(osd_method);
  INSERT_ARG(osd_order);
  INSERT_ARG(bp_batch_size);
  INSERT_ARG(osd_batch_size);
  INSERT_ARG(iter_per_check);
  INSERT_ARG(clip_value);
  INSERT_ARG(bp_method);
  INSERT_ARG(scale_factor);
  INSERT_ARG(proc_float);
  INSERT_ARG(gamma0);
  INSERT_ARG(gamma_dist);
  INSERT_ARG(explicit_gammas);
  INSERT_ARG(bp_seed);
  INSERT_ARG(composition);
  INSERT_ARG(repeatable);
  // srelay_config needs to be converted to heterogeneous_map for decoder
  // compatibility
  if (srelay_config.has_value()) {
    config_map.insert("srelay_config", srelay_config->to_heterogeneous_map());
  }

  return config_map;
}

nv_qldpc_decoder_config nv_qldpc_decoder_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  nv_qldpc_decoder_config config;
  GET_ARG(use_sparsity);
  GET_ARG(error_rate);
  GET_ARG(error_rate_vec);
  GET_ARG(max_iterations);
  GET_ARG(n_threads);
  GET_ARG(use_osd);
  GET_ARG(osd_method);
  GET_ARG(osd_order);
  GET_ARG(bp_batch_size);
  GET_ARG(osd_batch_size);
  GET_ARG(iter_per_check);
  GET_ARG(clip_value);
  GET_ARG(bp_method);
  GET_ARG(scale_factor);
  GET_ARG(proc_float);
  GET_ARG(gamma0);
  GET_ARG(gamma_dist);
  GET_ARG(explicit_gammas);
  GET_ARG(bp_seed);
  GET_ARG(composition);
  GET_ARG(repeatable);
  // Special handling for srelay_config - it might be stored as a nested
  // heterogeneous_map
  if (map.contains("srelay_config")) {
    try {
      // Try to get it as a srelay_bp_config first (for C++ -> C++)
      config.srelay_config = map.get<srelay_bp_config>("srelay_config");
    } catch (...) {
      // If that fails, try to get it as a heterogeneous_map (for Python
      // round-trip)
      auto nested_map = map.get<cudaqx::heterogeneous_map>("srelay_config");
      config.srelay_config =
          srelay_bp_config::from_heterogeneous_map(nested_map);
    }
  }

  return config;
}

// ------ multi_error_lut_config ------
cudaqx::heterogeneous_map multi_error_lut_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;
  INSERT_ARG(lut_error_depth);
  return config_map;
}

multi_error_lut_config multi_error_lut_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  multi_error_lut_config config;
  GET_ARG(lut_error_depth);
  return config;
}

// ------ single_error_lut_config ------
cudaqx::heterogeneous_map
single_error_lut_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;
  return config_map;
}

single_error_lut_config single_error_lut_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  single_error_lut_config config;
  return config;
}

cudaqx::heterogeneous_map global_decoder_config_to_heterogeneous_map(
    const global_decoder_config &global_decoder_params) {
  return std::visit(
      [](const auto &params) -> cudaqx::heterogeneous_map {
        using config_t = std::decay_t<decltype(params)>;
        if constexpr (std::is_same_v<config_t, std::monostate>) {
          return cudaqx::heterogeneous_map();
        } else {
          return params.to_heterogeneous_map();
        }
      },
      global_decoder_params);
}

global_decoder_config global_decoder_config_from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map,
    const std::optional<std::string> &global_decoder) {
  if (!global_decoder.has_value()) {
    throw std::runtime_error(
        "global_decoder_params present but global_decoder is not set.");
  }

  if (global_decoder.value() == "pymatching") {
    return pymatching_config::from_heterogeneous_map(map);
  }

  if (global_decoder.value() == "chromobius") {
    return chromobius_config::from_heterogeneous_map(map);
  }

  throw std::runtime_error(
      "global_decoder_params does not support global_decoder '" +
      global_decoder.value() + "'.");
}

global_decoder_config default_global_decoder_params(
    const std::optional<std::string> &global_decoder) {
  if (!global_decoder.has_value())
    return std::monostate{};

  if (global_decoder.value() == "pymatching")
    return pymatching_config{};

  if (global_decoder.value() == "chromobius")
    return chromobius_config{};

  return std::monostate{};
}

void validate_global_decoder_params(
    const global_decoder_config &global_decoder_params,
    const std::optional<std::string> &global_decoder);

global_decoder_config global_decoder_config_from_value(
    const std::any &val, const std::optional<std::string> &global_decoder) {
  if (!global_decoder.has_value()) {
    throw std::runtime_error(
        "global_decoder_params present but global_decoder is not set.");
  }

  if (auto *global_cfg = std::any_cast<global_decoder_config>(&val)) {
    validate_global_decoder_params(*global_cfg, global_decoder);
    return *global_cfg;
  }

  if (auto *nested_map = std::any_cast<cudaqx::heterogeneous_map>(&val)) {
    return global_decoder_config_from_heterogeneous_map(*nested_map,
                                                        global_decoder);
  }

  global_decoder_config parsed_params;
  if (auto *pymatching_cfg = std::any_cast<pymatching_config>(&val)) {
    parsed_params = *pymatching_cfg;
  } else if (auto *chromobius_cfg = std::any_cast<chromobius_config>(&val)) {
    parsed_params = *chromobius_cfg;
  } else {
    throw std::runtime_error(
        "global_decoder_params has an unsupported value type for "
        "global_decoder '" +
        global_decoder.value() + "'.");
  }

  validate_global_decoder_params(parsed_params, global_decoder);
  return parsed_params;
}

void validate_global_decoder_params(
    const global_decoder_config &global_decoder_params,
    const std::optional<std::string> &global_decoder) {
  if (std::holds_alternative<std::monostate>(global_decoder_params))
    return;

  if (!global_decoder.has_value()) {
    throw std::runtime_error(
        "global_decoder_params present but global_decoder is not set.");
  }

  if (global_decoder.value() == "pymatching" &&
      std::holds_alternative<pymatching_config>(global_decoder_params)) {
    return;
  }

  if (global_decoder.value() == "chromobius" &&
      std::holds_alternative<chromobius_config>(global_decoder_params)) {
    return;
  }

  throw std::runtime_error(
      "global_decoder_params type does not match global_decoder '" +
      global_decoder.value() + "'.");
}

// ------ pymatching_config ------
cudaqx::heterogeneous_map pymatching_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;

  INSERT_ARG(error_rate_vec);
  INSERT_ARG(merge_strategy);

  return config_map;
}

pymatching_config pymatching_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  pymatching_config config;
  GET_ARG(error_rate_vec);
  GET_ARG(merge_strategy);
  return config;
}

// ------ chromobius_config ------
cudaqx::heterogeneous_map chromobius_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;

  INSERT_ARG(drop_mobius_errors_involving_remnant_errors);
  INSERT_ARG(ignore_decomposition_failures);
  INSERT_ARG(include_coords_in_mobius_dem);
  INSERT_ARG(return_weight);
  INSERT_ARG(write_mobius_match_to_stderr);

  return config_map;
}

chromobius_config chromobius_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  chromobius_config config;
  GET_ARG(drop_mobius_errors_involving_remnant_errors);
  GET_ARG(ignore_decomposition_failures);
  GET_ARG(include_coords_in_mobius_dem);
  GET_ARG(return_weight);
  GET_ARG(write_mobius_match_to_stderr);
  return config;
}

// ------ trt_decoder_config ------
cudaqx::heterogeneous_map trt_decoder_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;

  INSERT_ARG(onnx_load_path);
  INSERT_ARG(engine_load_path);
  INSERT_ARG(engine_save_path);
  INSERT_ARG(precision);
  INSERT_ARG(memory_workspace);
  INSERT_ARG(batch_size);
  INSERT_ARG(use_cuda_graph);
  INSERT_ARG(global_decoder);
  auto effective_global_decoder_params = global_decoder_params;
  if (std::holds_alternative<std::monostate>(effective_global_decoder_params))
    effective_global_decoder_params =
        default_global_decoder_params(global_decoder);
  if (!std::holds_alternative<std::monostate>(
          effective_global_decoder_params)) {
    validate_global_decoder_params(effective_global_decoder_params,
                                   global_decoder);
    config_map.insert("global_decoder_params",
                      global_decoder_config_to_heterogeneous_map(
                          effective_global_decoder_params));
  }

  return config_map;
}

trt_decoder_config trt_decoder_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  trt_decoder_config config;
  GET_ARG(onnx_load_path);
  GET_ARG(engine_load_path);
  GET_ARG(engine_save_path);
  GET_ARG(precision);
  GET_ARG(memory_workspace);
  GET_ARG(batch_size);
  GET_ARG(use_cuda_graph);
  GET_ARG(global_decoder);
  if (map.contains("global_decoder_params")) {
    for (const auto &[key, val] : map) {
      if (key == "global_decoder_params") {
        config.global_decoder_params =
            global_decoder_config_from_value(val, config.global_decoder);
        break;
      }
    }
  } else {
    config.global_decoder_params =
        default_global_decoder_params(config.global_decoder);
  }

  return config;
}

// ------ sliding_window_config ------
cudaqx::heterogeneous_map sliding_window_config::to_heterogeneous_map() const {
  cudaqx::heterogeneous_map config_map;
  INSERT_ARG(window_size);
  INSERT_ARG(step_size);
  INSERT_ARG(num_syndromes_per_round);
  INSERT_ARG(straddle_start_round);
  INSERT_ARG(straddle_end_round);
  INSERT_ARG_PLAIN(error_rate_vec);
  INSERT_ARG_PLAIN(inner_decoder_name);

  // Handle concrete inner decoder configs
  cudaqx::heterogeneous_map inner_decoder_params;
  if (single_error_lut_params.has_value()) {
    inner_decoder_params =
        single_error_lut_params.value().to_heterogeneous_map();
  } else if (multi_error_lut_params.has_value()) {
    inner_decoder_params =
        multi_error_lut_params.value().to_heterogeneous_map();
  } else if (nv_qldpc_decoder_params.has_value()) {
    inner_decoder_params =
        nv_qldpc_decoder_params.value().to_heterogeneous_map();
  }
  if (!inner_decoder_params.empty()) {
    config_map.insert("inner_decoder_params", inner_decoder_params);
  }

  return config_map;
}

sliding_window_config sliding_window_config::from_heterogeneous_map(
    const cudaqx::heterogeneous_map &map) {
  sliding_window_config config;
  GET_ARG(window_size);
  GET_ARG(step_size);
  GET_ARG(num_syndromes_per_round);
  GET_ARG(straddle_start_round);
  GET_ARG(straddle_end_round);
  GET_ARG_PLAIN(error_rate_vec);
  GET_ARG_PLAIN(inner_decoder_name);

  // Reconstruct inner decoder configs based on the decoder type
  if (!config.inner_decoder_name.empty() &&
      map.contains("inner_decoder_params")) {
    const auto &inner_decoder_params =
        map.get<cudaqx::heterogeneous_map>("inner_decoder_params");
    const std::string &decoder_name = config.inner_decoder_name;

    if (decoder_name == "single_error_lut") {
      config.single_error_lut_params =
          single_error_lut_config::from_heterogeneous_map(inner_decoder_params);
    } else if (decoder_name == "multi_error_lut") {
      config.multi_error_lut_params =
          multi_error_lut_config::from_heterogeneous_map(inner_decoder_params);
    } else if (decoder_name == "nv-qldpc-decoder") {
      config.nv_qldpc_decoder_params =
          nv_qldpc_decoder_config::from_heterogeneous_map(inner_decoder_params);
    }
  }

  return config;
}

#undef INSERT_ARG
#undef GET_ARG

bool decoder_custom_args_t::operator==(
    const decoder_custom_args_t &other) const {
  return custom_args_maps_equal(map_, other.map_);
}

// Post-parse pass over a schema-parsed custom-args map: materialize defaulted
// discriminated sections, enforce required keys, and run per-schema validation
// hooks. Only invoked when the section was present in the input document,
// mirroring the previous behavior where an absent decoder_custom_args section
// skipped its mapping (and therefore its required-key checks) entirely.
static void finalize_parsed_args(const decoder_schema &schema,
                                 cudaqx::heterogeneous_map &map,
                                 const std::string &context) {
  for (const auto &spec : schema.params) {
    if (spec.kind == param_kind::discriminated) {
      std::string discriminator_value;
      if (map.contains(spec.discriminator))
        discriminator_value = map.get<std::string>(spec.discriminator);
      const auto *nested_schema = discriminator_value.empty()
                                      ? nullptr
                                      : find_decoder_schema(discriminator_value);
      if (spec.materialize_empty && nested_schema && !map.contains(spec.key))
        map.insert(spec.key, cudaqx::heterogeneous_map());
      if (nested_schema && map.contains(spec.key)) {
        auto nested = map.get<cudaqx::heterogeneous_map>(spec.key);
        finalize_parsed_args(*nested_schema, nested, context + "." + spec.key);
        map.insert(spec.key, nested);
      }
    } else if (spec.kind == param_kind::subschema) {
      if (map.contains(spec.key)) {
        if (const auto *nested_schema = find_decoder_schema(spec.subschema)) {
          auto nested = map.get<cudaqx::heterogeneous_map>(spec.key);
          finalize_parsed_args(*nested_schema, nested,
                               context + "." + spec.key);
          map.insert(spec.key, nested);
        }
      }
    }
    if (spec.required && !map.contains(spec.key))
      throw std::runtime_error("Missing required key '" + spec.key + "' in " +
                               context + ".");
  }
  if (schema.validate)
    schema.validate(map);
}

} // namespace cudaq::qec::decoding::config

LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<double>)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::qec::decoding::config::decoder_config)

namespace llvm::yaml {

// Binds a heterogeneous_map to the decoder_schema that describes it so the
// generic mapping traits below can convert between the two. The schema drives
// everything: which keys are legal, the canonical storage type of each value,
// and how nested sections resolve their schemas.
struct schema_binding {
  cudaqx::heterogeneous_map *map = nullptr;
  const cudaq::qec::decoding::config::decoder_schema *schema = nullptr;
};

namespace {

template <typename T>
void input_schema_scalar(IO &io, const std::string &key,
                         cudaqx::heterogeneous_map &map) {
  T value{};
  io.mapRequired(key.c_str(), value);
  map.insert(key, value);
}

template <typename T>
void output_schema_scalar(IO &io, const std::string &key,
                          const cudaqx::heterogeneous_map &map) {
  T value = map.get<T>(key);
  io.mapRequired(key.c_str(), value);
}

} // namespace

template <>
struct CustomMappingTraits<schema_binding> {
  using param_kind = cudaq::qec::decoding::config::param_kind;
  using param_spec = cudaq::qec::decoding::config::param_spec;

  static void inputOne(IO &io, StringRef key, schema_binding &binding) {
    const std::string key_str = key.str();
    const param_spec *spec = nullptr;
    for (const auto &candidate : binding.schema->params) {
      if (candidate.key == key_str) {
        spec = &candidate;
        break;
      }
    }
    if (!spec)
      throw std::runtime_error("Unknown key '" + key_str + "' in '" +
                               binding.schema->name + "' parameters.");

    switch (spec->kind) {
    case param_kind::boolean:
      input_schema_scalar<bool>(io, key_str, *binding.map);
      break;
    case param_kind::int32:
      input_schema_scalar<int>(io, key_str, *binding.map);
      break;
    case param_kind::uint64:
      input_schema_scalar<std::size_t>(io, key_str, *binding.map);
      break;
    case param_kind::f64:
      input_schema_scalar<double>(io, key_str, *binding.map);
      break;
    case param_kind::string:
      input_schema_scalar<std::string>(io, key_str, *binding.map);
      break;
    case param_kind::f64_vec:
      input_schema_scalar<std::vector<double>>(io, key_str, *binding.map);
      break;
    case param_kind::f64_matrix:
      input_schema_scalar<std::vector<std::vector<double>>>(io, key_str,
                                                            *binding.map);
      break;
    case param_kind::subschema: {
      const auto *nested_schema =
          cudaq::qec::decoding::config::find_decoder_schema(spec->subschema);
      if (!nested_schema)
        throw std::runtime_error("No schema registered under '" +
                                 spec->subschema + "' (needed to parse '" +
                                 key_str + "').");
      cudaqx::heterogeneous_map nested;
      schema_binding nested_binding{&nested, nested_schema};
      io.mapRequired(key_str.c_str(), nested_binding);
      binding.map->insert(key_str, nested);
      break;
    }
    case param_kind::discriminated: {
      // The nested schema is named by a sibling key. Read it through the IO
      // (document order does not matter; mapping keys are random access).
      std::string discriminator_value;
      io.mapOptional(spec->discriminator.c_str(), discriminator_value);
      if (discriminator_value.empty())
        throw std::runtime_error("'" + key_str + "' is present but '" +
                                 spec->discriminator + "' is not set.");
      const auto *nested_schema =
          cudaq::qec::decoding::config::find_decoder_schema(
              discriminator_value);
      if (!nested_schema)
        throw std::runtime_error(
            "'" + key_str + "' does not support " + spec->discriminator +
            " '" + discriminator_value +
            "': no parameter schema is registered under that name.");
      cudaqx::heterogeneous_map nested;
      schema_binding nested_binding{&nested, nested_schema};
      io.mapRequired(key_str.c_str(), nested_binding);
      binding.map->insert(key_str, nested);
      break;
    }
    }
  }

  static void output(IO &io, schema_binding &binding) {
    // Emit in schema declaration order so output is deterministic.
    for (const auto &spec : binding.schema->params) {
      if (!binding.map->contains(spec.key))
        continue;
      switch (spec.kind) {
      case param_kind::boolean:
        output_schema_scalar<bool>(io, spec.key, *binding.map);
        break;
      case param_kind::int32:
        output_schema_scalar<int>(io, spec.key, *binding.map);
        break;
      case param_kind::uint64:
        output_schema_scalar<std::size_t>(io, spec.key, *binding.map);
        break;
      case param_kind::f64:
        output_schema_scalar<double>(io, spec.key, *binding.map);
        break;
      case param_kind::string:
        output_schema_scalar<std::string>(io, spec.key, *binding.map);
        break;
      case param_kind::f64_vec:
        output_schema_scalar<std::vector<double>>(io, spec.key, *binding.map);
        break;
      case param_kind::f64_matrix:
        output_schema_scalar<std::vector<std::vector<double>>>(io, spec.key,
                                                               *binding.map);
        break;
      case param_kind::subschema: {
        const auto *nested_schema =
            cudaq::qec::decoding::config::find_decoder_schema(spec.subschema);
        if (!nested_schema)
          throw std::runtime_error("No schema registered under '" +
                                   spec.subschema + "' (needed to emit '" +
                                   spec.key + "').");
        auto nested = binding.map->get<cudaqx::heterogeneous_map>(spec.key);
        schema_binding nested_binding{&nested, nested_schema};
        io.mapRequired(spec.key.c_str(), nested_binding);
        break;
      }
      case param_kind::discriminated: {
        std::string discriminator_value;
        if (binding.map->contains(spec.discriminator))
          discriminator_value =
              binding.map->get<std::string>(spec.discriminator);
        const auto *nested_schema =
            discriminator_value.empty()
                ? nullptr
                : cudaq::qec::decoding::config::find_decoder_schema(
                      discriminator_value);
        if (!nested_schema)
          throw std::runtime_error(
              "'" + spec.key + "' is present but no parameter schema is "
              "registered for " + spec.discriminator + " '" +
              discriminator_value + "'.");
        auto nested = binding.map->get<cudaqx::heterogeneous_map>(spec.key);
        schema_binding nested_binding{&nested, nested_schema};
        io.mapRequired(spec.key.c_str(), nested_binding);
        break;
      }
      }
    }
  }
};

template <>
struct ScalarEnumerationTraits<cudaq::qec::decoding::config::DecoderTransport> {
  static void
  enumeration(IO &io, cudaq::qec::decoding::config::DecoderTransport &value) {
    io.enumCase(value, "cpu_roce",
                cudaq::qec::decoding::config::DecoderTransport::cpu_roce);
    io.enumCase(value, "gpu_roce",
                cudaq::qec::decoding::config::DecoderTransport::gpu_roce);
  }
};

template <>
struct MappingTraits<cudaq::qec::decoding::config::decoder_config> {
  static void mapping(IO &io,
                      cudaq::qec::decoding::config::decoder_config &config) {
    io.mapRequired("id", config.id);
    io.mapRequired("type", config.type);
    io.mapOptional("transport", config.transport,
                   cudaq::qec::decoding::config::DecoderTransport::cpu_roce);
    io.mapRequired("block_size", config.block_size);
    io.mapRequired("syndrome_size", config.syndrome_size);
    io.mapRequired("H_sparse", config.H_sparse);
    io.mapRequired("O_sparse", config.O_sparse);
    io.mapRequired("D_sparse", config.D_sparse);

    // Validate that the number of rows in the H_sparse vector is equal to
    // syndrome_size.
    auto num_H_rows =
        std::count(config.H_sparse.begin(), config.H_sparse.end(), -1);
    if (num_H_rows != config.syndrome_size) {
      throw std::runtime_error(
          "Number of rows in H_sparse vector is not equal to syndrome_size: " +
          std::to_string(num_H_rows) +
          " != " + std::to_string(config.syndrome_size));
    }

    // Validate that no values in the H_sparse vector are out of range.
    for (auto value : config.H_sparse) {
      if (value < -1 || (value >= 0 && value >= config.block_size)) {
        throw std::runtime_error("Value in H_sparse vector is out of range: " +
                                 std::to_string(value));
      }
    }

    // Validate that no values in the O_sparse vector are out of range.
    for (auto value : config.O_sparse) {
      if (value < -1 || (value >= 0 && value >= config.block_size)) {
        throw std::runtime_error("Value in O_sparse vector is out of range: " +
                                 std::to_string(value));
      }
    }

    // Validate that if the D_sparse is provided, it is a valid D matrix. That
    // means that the number of rows in the D_sparse matrix should be equal to
    // the number of rows in the H_sparse matrix, and no row should be empty.
    if (!config.D_sparse.empty()) {
      auto num_D_rows =
          std::count(config.D_sparse.begin(), config.D_sparse.end(), -1);
      if (num_D_rows != config.syndrome_size) {
        throw std::runtime_error("Number of rows in D_sparse vector is not "
                                 "equal to syndrome_size: " +
                                 std::to_string(num_D_rows) +
                                 " != " + std::to_string(config.syndrome_size));
      }
      // No row should be empty, which means that there should be no
      // back-to-back -1 values.
      for (std::size_t i = 0; i < config.D_sparse.size() - 1; ++i) {
        if (config.D_sparse.at(i) == -1 && config.D_sparse.at(i + 1) == -1) {
          throw std::runtime_error("D_sparse row is empty for decoder " +
                                   std::to_string(config.id));
        }
      }
    }

    // Convert decoder_custom_args through the schema registered for this
    // decoder type. When no schema is registered, the key is intentionally
    // left unconsumed on input so the YAML parser's strict unknown-key check
    // rejects the section -- a decoder must register a schema (from its own
    // plugin library) to accept custom args.
    const auto *schema =
        cudaq::qec::decoding::config::find_decoder_schema(config.type);
    if (io.outputting()) {
      if (!config.decoder_custom_args.empty()) {
        if (!schema) {
          // Match the historical emission behavior (args for unknown types
          // were silently dropped) so configuration flows still fail with a
          // status code at decoder construction rather than throwing here.
          CUDA_QEC_WARN(
              "decoder_custom_args set for decoder type '{}' but no parameter "
              "schema is registered under that name; the args are omitted "
              "from the emitted YAML.",
              config.type);
        } else {
          auto args_map = config.decoder_custom_args.map();
          schema_binding binding{&args_map, schema};
          io.mapRequired("decoder_custom_args", binding);
        }
      }
    } else if (schema) {
      bool args_present = false;
      for (const auto key : io.keys()) {
        if (key == "decoder_custom_args") {
          args_present = true;
          break;
        }
      }
      cudaqx::heterogeneous_map args_map;
      schema_binding binding{&args_map, schema};
      io.mapOptional("decoder_custom_args", binding);
      if (args_present)
        cudaq::qec::decoding::config::finalize_parsed_args(
            *schema, args_map, "decoder_custom_args (" + config.type + ")");
      config.decoder_custom_args = args_map;
    }
  }
};

// multi_decoder_config mapping traits
template <>
struct MappingTraits<cudaq::qec::decoding::config::multi_decoder_config> {
  static void
  mapping(IO &io, cudaq::qec::decoding::config::multi_decoder_config &config) {
    io.mapRequired("decoders", config.decoders);
  }
};

} // namespace llvm::yaml

// Static method to convert a YAML string to a multi_decoder_config.
cudaq::qec::decoding::config::multi_decoder_config
cudaq::qec::decoding::config::multi_decoder_config::from_yaml_str(
    const std::string_view yaml_str) {
  multi_decoder_config config;
  llvm::yaml::Input yaml_in(yaml_str);
  yaml_in >> config;
  if (const auto error = yaml_in.error())
    throw std::runtime_error("Invalid decoder configuration YAML: " +
                             error.message());
  return config;
}

std::string cudaq::qec::decoding::config::multi_decoder_config::to_yaml_str(
    int column_wrap) {
  std::string yaml_str;
  llvm::raw_string_ostream yaml_stream(yaml_str);
  llvm::yaml::Output yaml_out(yaml_stream, nullptr, column_wrap);
  yaml_out << *this;
  return yaml_str;
}

cudaq::qec::decoding::config::decoder_config
cudaq::qec::decoding::config::decoder_config::from_yaml_str(
    const std::string &yaml_str) {
  decoder_config config;
  llvm::yaml::Input yaml_in(yaml_str);
  yaml_in >> config;
  return config;
}

std::string
cudaq::qec::decoding::config::decoder_config::to_yaml_str(int column_wrap) {
  std::string yaml_str;
  llvm::raw_string_ostream yaml_stream(yaml_str);
  llvm::yaml::Output yaml_out(yaml_stream, nullptr, column_wrap);
  yaml_out << *this;
  return yaml_str;
}

namespace cudaq::qec::decoding::config {

// Stash a copy for consumers that build their own decoder instances from the
// process-wide configuration -- the decoding-server DeviceCallService plugin
// reads it when CUDAQ_QEC_DECODER_CONFIG is not set (in-process path).
// shared_ptr + mutex: the plugin reads this from the realtime dispatcher
// thread while the application thread may call configure_decoders() again;
// shared ownership keeps the reader's config alive across a concurrent
// replacement.
static std::mutex g_last_multi_decoder_config_mutex;
static std::shared_ptr<const multi_decoder_config> g_last_multi_decoder_config;

int configure_decoders(multi_decoder_config &config) {
  CUDA_QEC_INFO("Initializing realtime decoding library with config object");
  {
    std::lock_guard<std::mutex> lock(g_last_multi_decoder_config_mutex);
    g_last_multi_decoder_config =
        std::make_shared<const multi_decoder_config>(config);
  }
  // Publish the decoder configuration so CUDA-Q can inject it into
  // remote-target job requests. The cudaq integration (ExtraPayloadProvider) is
  // installed by cudaq-qec at load time; this call is a no-op when cudaq-qec is
  // not loaded, keeping this library free of any direct cudaq-common
  // dependency.
  cudaq::qec::publish_decoder_config_payload(config.to_yaml_str());
  return cudaq::qec::decoding::host::configure_decoders(config);
}

std::shared_ptr<const multi_decoder_config>
last_configured_multi_decoder_config() {
  std::lock_guard<std::mutex> lock(g_last_multi_decoder_config_mutex);
  return g_last_multi_decoder_config;
}

void log_config(const char *config_str, bool from_file) {
  const bool dump_config = []() {
    if (auto *ch = std::getenv("CUDAQ_QEC_DEBUG_DECODER"))
      if (ch[0] == '1' || ch[0] == 'y' || ch[0] == 'Y')
        return true;
    return false;
  }();

  if (dump_config) {
    if (cudaq::qec::detail::should_log(cudaq::qec::detail::log_level::info)) {
      CUDA_QEC_INFO(
          "Initializing realtime decoding library with config string: {}",
          config_str);
    } else {
      printf("Initializing realtime decoding library with config string: %s\n",
             config_str);
    }
  }
}

int configure_decoders_from_file(const char *config_file) {
  std::string config_file_str(config_file);
  CUDA_QEC_INFO("Initializing realtime decoding library with config file: {}",
                config_file_str);

  // Verify that the file exists.
  if (!std::filesystem::exists(config_file_str)) {
    CUDA_QEC_WARN("Config file does not exist: {}", config_file_str);
    return 1;
  }

  // Read the config file into a string.
  std::string config_str;
  std::ifstream config_file_stream(config_file_str);
  config_str = std::string(std::istreambuf_iterator<char>(config_file_stream),
                           std::istreambuf_iterator<char>());
  log_config(config_str.c_str(), /*from_file=*/true);
  auto config = multi_decoder_config::from_yaml_str(config_str);
  return configure_decoders(config);
}

int configure_decoders_from_str(const char *config_str) {
  CUDA_QEC_INFO(
      "Initializing realtime decoding library with raw config string");
  log_config(config_str, /*from_file=*/false);
  auto config = multi_decoder_config::from_yaml_str(config_str);
  return configure_decoders(config);
}

void finalize_decoders() { cudaq::qec::decoding::host::finalize_decoders(); }

} // namespace cudaq::qec::decoding::config
