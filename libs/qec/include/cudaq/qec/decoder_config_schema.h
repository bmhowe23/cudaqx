/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include <functional>
#include <string>
#include <vector>

namespace cudaq::qec::decoding::config {

/// Decoder parameter schemas.
///
/// A decoder that wants its constructor parameters to be configurable through
/// the realtime decoding YAML (the `decoder_custom_args` section of a
/// `decoder_config`) registers a `decoder_schema` describing those parameters,
/// keyed by the same name it registers its `cudaq::qec::decoder` extension
/// under. The YAML layer uses the schema to convert the section to and from a
/// `cudaqx::heterogeneous_map` generically -- no decoder-specific parsing code
/// is required in the framework, so out-of-tree decoder plugins can make
/// themselves "realtime ready" from their own shared library.
///
/// The value kinds are a deliberately small, closed set of canonical storage
/// types. Values parsed from YAML are stored in the map with exactly the type
/// listed for their kind; `heterogeneous_map::get` handles related-type
/// retrieval (e.g. an `int` stored for an `int32` parameter can be read back
/// as `std::size_t`).
enum class param_kind {
  boolean,    ///< stored as bool
  int32,      ///< stored as int
  uint64,     ///< stored as std::size_t
  f64,        ///< stored as double
  string,     ///< stored as std::string
  f64_vec,    ///< stored as std::vector<double>
  f64_matrix, ///< stored as std::vector<std::vector<double>>
  /// Nested mapping parsed with the fixed schema named by
  /// `param_spec::subschema`; stored as a nested heterogeneous_map.
  subschema,
  /// Nested mapping whose schema is selected at parse time by the value of
  /// the sibling string key named by `param_spec::discriminator` (e.g.
  /// `global_decoder_params` is parsed with the schema registered under the
  /// value of `global_decoder`); stored as a nested heterogeneous_map.
  discriminated,
};

/// One parameter of a decoder's custom-args schema.
struct param_spec {
  std::string key;
  param_kind kind = param_kind::string;
  /// Parsing fails when a required key is absent.
  bool required = false;
  /// For kind::subschema: the registered schema name to parse the nested
  /// mapping with.
  std::string subschema;
  /// For kind::discriminated: the sibling key whose (string) value names the
  /// schema to parse the nested mapping with.
  std::string discriminator;
  /// For kind::discriminated: when the discriminator is present and names a
  /// registered schema but this key is absent, insert an empty nested map so
  /// downstream consumers see the section (matches the trt_decoder
  /// global_decoder_params defaulting behavior).
  bool materialize_empty = false;
};

/// Declarative description of a decoder's `decoder_custom_args` section.
struct decoder_schema {
  /// Registry key. For decoders this must match the name the decoder is
  /// registered under (the YAML `type` value). Schemas that only serve as
  /// nested sections (e.g. "srelay_bp") may use any unique name.
  std::string name;
  std::vector<param_spec> params;
  /// Optional cross-field validation hook, invoked after a section has been
  /// parsed and its required keys checked. Throw std::runtime_error to
  /// reject the configuration.
  std::function<void(const cudaqx::heterogeneous_map &)> validate;
};

/// Register (or replace) a schema. Decoder plugins call this from a static
/// initializer in the same shared library that registers the decoder itself;
/// the plugin loader runs before any configuration is parsed.
__attribute__((visibility("default"))) void
register_decoder_schema(decoder_schema schema);

/// Look up a schema by name. Returns nullptr when no schema is registered
/// under `name`. The returned pointer remains valid for the process lifetime.
__attribute__((visibility("default"))) const decoder_schema *
find_decoder_schema(const std::string &name);

/// Names of all registered schemas (for diagnostics and introspection).
__attribute__((visibility("default"))) std::vector<std::string>
registered_decoder_schema_names();

/// Validate a custom-args map against the schema registered under
/// `schema_name`: every key must appear in the schema (nested sections are
/// checked recursively), every required key must be present, and the schema's
/// `validate` hook (if any) runs last. Throws std::runtime_error describing
/// the first violation. Maps that did not come from the YAML parser (e.g.
/// built programmatically or from a Python dict) get the same checks the
/// parser applies, so configurations can be validated before use. A map for
/// a name with no registered schema is rejected unless it is empty.
__attribute__((visibility("default"))) void
validate_custom_args(const std::string &schema_name,
                     const cudaqx::heterogeneous_map &args);

/// Deep equality over the canonical value kinds stored in custom-args maps
/// (scalars, double vectors/matrices, and nested maps). Values of other
/// types compare unequal.
__attribute__((visibility("default"))) bool
custom_args_maps_equal(const cudaqx::heterogeneous_map &a,
                       const cudaqx::heterogeneous_map &b);

} // namespace cudaq::qec::decoding::config
