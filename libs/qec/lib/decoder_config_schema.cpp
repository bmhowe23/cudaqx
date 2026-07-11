/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder_config_schema.h"
#include <any>
#include <map>
#include <mutex>
#include <optional>

namespace cudaq::qec::decoding::config {

namespace {

// Node-based map: pointers returned by find_decoder_schema stay valid across
// later registrations. Schemas are never erased.
std::map<std::string, decoder_schema> &schema_registry() {
  static std::map<std::string, decoder_schema> registry;
  return registry;
}

std::mutex &schema_registry_mutex() {
  static std::mutex m;
  return m;
}

} // namespace

void register_decoder_schema(decoder_schema schema) {
  std::lock_guard<std::mutex> lock(schema_registry_mutex());
  auto name = schema.name;
  schema_registry().insert_or_assign(std::move(name), std::move(schema));
}

const decoder_schema *find_decoder_schema(const std::string &name) {
  std::lock_guard<std::mutex> lock(schema_registry_mutex());
  auto &registry = schema_registry();
  auto iter = registry.find(name);
  if (iter == registry.end())
    return nullptr;
  return &iter->second;
}

std::vector<std::string> registered_decoder_schema_names() {
  std::lock_guard<std::mutex> lock(schema_registry_mutex());
  std::vector<std::string> names;
  for (const auto &[name, schema] : schema_registry())
    names.push_back(name);
  return names;
}

// ---------------------------------------------------------------------------
// Deep equality over canonical custom-args value kinds
// ---------------------------------------------------------------------------

namespace {

std::optional<long long> as_integer(const std::any &v) {
  if (auto *p = std::any_cast<int>(&v))
    return static_cast<long long>(*p);
  if (auto *p = std::any_cast<long>(&v))
    return static_cast<long long>(*p);
  if (auto *p = std::any_cast<long long>(&v))
    return *p;
  if (auto *p = std::any_cast<short>(&v))
    return static_cast<long long>(*p);
  if (auto *p = std::any_cast<unsigned int>(&v))
    return static_cast<long long>(*p);
  if (auto *p = std::any_cast<unsigned long>(&v))
    return static_cast<long long>(*p);
  if (auto *p = std::any_cast<unsigned long long>(&v))
    return static_cast<long long>(*p);
  if (auto *p = std::any_cast<unsigned short>(&v))
    return static_cast<long long>(*p);
  return std::nullopt;
}

std::optional<double> as_floating(const std::any &v) {
  if (auto *p = std::any_cast<double>(&v))
    return *p;
  if (auto *p = std::any_cast<float>(&v))
    return static_cast<double>(*p);
  return std::nullopt;
}

bool any_values_equal(const std::any &a, const std::any &b) {
  // bool is kept distinct from the integer family so `true` never silently
  // matches `1` from a differently-built configuration.
  if (auto *pa = std::any_cast<bool>(&a)) {
    auto *pb = std::any_cast<bool>(&b);
    return pb && *pa == *pb;
  }
  if (std::any_cast<bool>(&b))
    return false;

  if (auto ia = as_integer(a)) {
    if (auto ib = as_integer(b))
      return *ia == *ib;
    if (auto fb = as_floating(b))
      return static_cast<double>(*ia) == *fb;
    return false;
  }
  if (auto fa = as_floating(a)) {
    if (auto fb = as_floating(b))
      return *fa == *fb;
    if (auto ib = as_integer(b))
      return *fa == static_cast<double>(*ib);
    return false;
  }
  if (auto *pa = std::any_cast<std::string>(&a)) {
    auto *pb = std::any_cast<std::string>(&b);
    return pb && *pa == *pb;
  }
  if (auto *pa = std::any_cast<std::vector<double>>(&a)) {
    auto *pb = std::any_cast<std::vector<double>>(&b);
    return pb && *pa == *pb;
  }
  if (auto *pa = std::any_cast<std::vector<std::vector<double>>>(&a)) {
    auto *pb = std::any_cast<std::vector<std::vector<double>>>(&b);
    return pb && *pa == *pb;
  }
  if (auto *pa = std::any_cast<cudaqx::heterogeneous_map>(&a)) {
    auto *pb = std::any_cast<cudaqx::heterogeneous_map>(&b);
    return pb && custom_args_maps_equal(*pa, *pb);
  }
  // Unknown value type: conservatively unequal.
  return false;
}

} // namespace

bool custom_args_maps_equal(const cudaqx::heterogeneous_map &a,
                            const cudaqx::heterogeneous_map &b) {
  std::size_t size_a = 0;
  for ([[maybe_unused]] const auto &kv : a)
    ++size_a;
  std::size_t size_b = 0;
  for ([[maybe_unused]] const auto &kv : b)
    ++size_b;
  if (size_a != size_b)
    return false;

  for (const auto &[key_a, val_a] : a) {
    bool found = false;
    for (const auto &[key_b, val_b] : b) {
      if (key_a != key_b)
        continue;
      found = true;
      if (!any_values_equal(val_a, val_b))
        return false;
      break;
    }
    if (!found)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Built-in decoder schemas
//
// These describe the custom-args of the decoders that ship with CUDA-Q QEC
// (plus the proprietary nv-qldpc-decoder plugin, whose schema is hosted here
// until the plugin registers it itself). Third-party decoders register their
// schema from their own plugin library instead of editing this file.
// ---------------------------------------------------------------------------

namespace {

struct builtin_schema_registrar {
  builtin_schema_registrar() {
    using k = param_kind;

    register_decoder_schema(
        {"srelay_bp",
         {
             {"pre_iter", k::uint64},
             {"num_sets", k::uint64},
             {"stopping_criterion", k::string},
             {"stop_nconv", k::uint64},
         }});

    register_decoder_schema(
        {"nv-qldpc-decoder",
         {
             {"use_sparsity", k::boolean},
             {"error_rate", k::f64},
             {"error_rate_vec", k::f64_vec},
             {"max_iterations", k::int32},
             {"n_threads", k::int32},
             {"use_osd", k::boolean},
             {"osd_method", k::int32},
             {"osd_order", k::int32},
             {"bp_batch_size", k::int32},
             {"osd_batch_size", k::int32},
             {"iter_per_check", k::int32},
             {"clip_value", k::f64},
             {"bp_method", k::int32},
             {"scale_factor", k::f64},
             {"proc_float", k::string},
             {"gamma0", k::f64},
             {"gamma_dist", k::f64_vec},
             {"explicit_gammas", k::f64_matrix},
             {"bp_seed", k::int32},
             {"srelay_config", k::subschema, false, "srelay_bp"},
             {"composition", k::int32},
             {"repeatable", k::boolean},
         }});

    register_decoder_schema({"single_error_lut", {}});

    register_decoder_schema({"multi_error_lut",
                             {
                                 {"lut_error_depth", k::int32},
                             }});

    register_decoder_schema({"pymatching",
                             {
                                 {"error_rate_vec", k::f64_vec},
                                 {"merge_strategy", k::string},
                             }});

    register_decoder_schema(
        {"chromobius",
         {
             {"drop_mobius_errors_involving_remnant_errors", k::boolean},
             {"ignore_decomposition_failures", k::boolean},
             {"include_coords_in_mobius_dem", k::boolean},
             {"return_weight", k::boolean},
             {"write_mobius_match_to_stderr", k::boolean},
         }});

    register_decoder_schema(
        {"trt_decoder",
         {
             {"onnx_load_path", k::string},
             {"engine_load_path", k::string},
             {"engine_save_path", k::string},
             {"precision", k::string},
             {"memory_workspace", k::uint64},
             {"batch_size", k::uint64},
             {"use_cuda_graph", k::boolean},
             {"global_decoder", k::string},
             {"global_decoder_params", k::discriminated, false, "",
              "global_decoder", /*materialize_empty=*/true},
         }});

    register_decoder_schema(
        {"sliding_window",
         {
             {"window_size", k::uint64},
             {"step_size", k::uint64},
             {"num_syndromes_per_round", k::uint64},
             {"straddle_start_round", k::boolean},
             {"straddle_end_round", k::boolean},
             {"error_rate_vec", k::f64_vec, /*required=*/true},
             {"inner_decoder_name", k::string, /*required=*/true},
             {"inner_decoder_params", k::discriminated, false, "",
              "inner_decoder_name", /*materialize_empty=*/false},
         }});
  }
};

builtin_schema_registrar builtin_schemas;

} // namespace

} // namespace cudaq::qec::decoding::config
