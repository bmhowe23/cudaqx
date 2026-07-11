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
#include <stdexcept>

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
// Schema validation of programmatically built custom-args maps
// ---------------------------------------------------------------------------

namespace {

void validate_args_against_schema(const decoder_schema &schema,
                                  const cudaqx::heterogeneous_map &args,
                                  const std::string &context) {
  for (const auto &kv : args) {
    const std::string &key = kv.first;
    const param_spec *spec = nullptr;
    for (const auto &candidate : schema.params) {
      if (candidate.key == key) {
        spec = &candidate;
        break;
      }
    }
    if (!spec)
      throw std::runtime_error("Unknown key '" + key + "' in " + context +
                               ".");
    if (spec->kind == param_kind::subschema) {
      const auto *nested_schema = find_decoder_schema(spec->subschema);
      if (!nested_schema)
        throw std::runtime_error("No schema registered under '" +
                                 spec->subschema + "' (needed to validate '" +
                                 key + "').");
      validate_args_against_schema(
          *nested_schema, args.get<cudaqx::heterogeneous_map>(key),
          context + "." + key);
    } else if (spec->kind == param_kind::discriminated) {
      std::string discriminator_value;
      if (args.contains(spec->discriminator))
        discriminator_value = args.get<std::string>(spec->discriminator);
      if (discriminator_value.empty())
        throw std::runtime_error("'" + key + "' is present but '" +
                                 spec->discriminator + "' is not set in " +
                                 context + ".");
      const auto *nested_schema = find_decoder_schema(discriminator_value);
      if (!nested_schema)
        throw std::runtime_error(
            "'" + key + "' does not support " + spec->discriminator + " '" +
            discriminator_value +
            "': no parameter schema is registered under that name.");
      validate_args_against_schema(
          *nested_schema, args.get<cudaqx::heterogeneous_map>(key),
          context + "." + key);
    }
  }
  for (const auto &spec : schema.params)
    if (spec.required && !args.contains(spec.key))
      throw std::runtime_error("Missing required key '" + spec.key + "' in " +
                               context + ".");
  if (schema.validate)
    schema.validate(args);
}

} // namespace

void validate_custom_args(const std::string &schema_name,
                          const cudaqx::heterogeneous_map &args) {
  const auto *schema = find_decoder_schema(schema_name);
  if (!schema) {
    if (args.empty())
      return;
    throw std::runtime_error(
        "Decoder type '" + schema_name +
        "' has no registered parameter schema; its decoder_custom_args "
        "cannot be validated (or serialized to YAML). Register a schema from "
        "the decoder's plugin library with register_decoder_schema().");
  }
  validate_args_against_schema(*schema, args,
                               "'" + schema_name + "' parameters");
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
// Hosted decoder schemas
//
// Decoder schemas are registered by the shared library that ships the
// decoder: lut.cpp and sliding_window.cpp register theirs in this library,
// and the pymatching / chromobius / trt_decoder plugins register theirs from
// their own .so. The nv-qldpc-decoder schema is hosted here temporarily: the
// decoder is a proprietary out-of-tree plugin, and its schema (plus the
// srelay_bp subschema it references) moves into that plugin once it links
// against this registry. Third-party decoders register their schema from
// their own plugin library instead of editing this file.
// ---------------------------------------------------------------------------

namespace {

struct hosted_schema_registrar {
  hosted_schema_registrar() {
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
  }
};

hosted_schema_registrar hosted_schemas;

} // namespace

} // namespace cudaq::qec::decoding::config
