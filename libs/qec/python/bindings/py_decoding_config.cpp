/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_decoding_config.h"

#include "type_casters.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaq::qec::decoding::config {
void bindDecodingConfig(nb::module_ &mod) {
  auto qecmod = nb::hasattr(mod, "qecrt")
                    ? nb::cast<nb::module_>(mod.attr("qecrt"))
                    : mod.def_submodule("qecrt");

  auto mod_cfg =
      qecmod.def_submodule("config", "Realtime decoding configuration");

  // decoder_config
  nb::class_<config::decoder_config>(mod_cfg, "decoder_config")
      .def(nb::init<>())
      .def_rw("id", &decoder_config::id)
      .def_rw("type", &decoder_config::type)
      .def_rw("block_size", &decoder_config::block_size)
      .def_rw("syndrome_size", &decoder_config::syndrome_size)
      .def_rw("H_sparse", &decoder_config::H_sparse)
      .def_rw("O_sparse", &decoder_config::O_sparse)
      .def_rw("D_sparse", &decoder_config::D_sparse)
      .def_prop_rw(
          "decoder_custom_args",
          [](const decoder_config &self) -> nb::object {
            // The decoder's parameter dict. Keys are governed by the
            // parameter schema the decoder registered (see
            // decoder_param_schema()); the same representation serves
            // built-in and third-party decoders alike.
            return nb::cast(self.decoder_custom_args.map());
          },
          [](decoder_config &self, nb::object value) {
            self.decoder_custom_args =
                nb::cast<cudaqx::heterogeneous_map>(value);
          })
      .def(
          "set_decoder_custom_args",
          [](config::decoder_config &self, nb::object custom_args) {
            self.decoder_custom_args =
                nb::cast<cudaqx::heterogeneous_map>(custom_args);
          },
          nb::arg("custom_args"),
          "Set the decoder parameter dict for this decoder.")
      .def("validate_custom_args", &decoder_config::validate_custom_args,
           "Validate decoder_custom_args against the parameter schema "
           "registered for this decoder type: unknown keys, missing required "
           "keys, and the schema's own validation hook. Raises RuntimeError "
           "on the first violation. YAML parsing applies the same checks "
           "automatically; call this to vet a configuration built "
           "programmatically before using it.")
      .def("to_yaml_str", &decoder_config::to_yaml_str,
           nb::arg("column_wrap") = 80)
      .def_static("from_yaml_str", &decoder_config::from_yaml_str,
                  nb::arg("yaml_str"))
      .def("__eq__", [](const decoder_config &a, const decoder_config &b) {
        return a == b;
      });

  // multi_decoder_config
  nb::class_<multi_decoder_config>(mod_cfg, "multi_decoder_config")
      .def(nb::init<>())
      .def_rw("decoders", &multi_decoder_config::decoders)
      .def("validate_custom_args", &multi_decoder_config::validate_custom_args,
           "Validate every decoder's custom args against its registered "
           "parameter schema (see decoder_config.validate_custom_args).")
      .def("to_yaml_str", &multi_decoder_config::to_yaml_str,
           nb::arg("column_wrap") = 80)
      .def_static("from_yaml_str", &multi_decoder_config::from_yaml_str,
                  nb::arg("yaml_str"))
      .def("__eq__", [](const multi_decoder_config &a,
                        const multi_decoder_config &b) { return a == b; });

  // Library helpers
  mod_cfg.def(
      "configure_decoders", &configure_decoders, nb::arg("config"),
      "Configure decoders in a multi_decoder_config list; returns int status.");
  mod_cfg.def("configure_decoders_from_file", &configure_decoders_from_file,
              nb::arg("config_file"),
              "Configure decoders from a YAML file; returns int status.");
  mod_cfg.def("configure_decoders_from_str", &configure_decoders_from_str,
              nb::arg("config_str"),
              "Configure decoders from a YAML string; returns int status.");
  mod_cfg.def("finalize_decoders", &finalize_decoders,
              "Finalize decoder resources.");
  mod_cfg.def(
      "decoder_param_schema",
      [](const std::string &name) -> nb::object {
        const auto *schema = find_decoder_schema(name);
        if (!schema)
          return nb::none();
        auto kind_name = [](param_kind kind) -> const char * {
          switch (kind) {
          case param_kind::boolean:
            return "bool";
          case param_kind::int32:
            return "int32";
          case param_kind::uint64:
            return "uint64";
          case param_kind::f64:
            return "float64";
          case param_kind::string:
            return "string";
          case param_kind::f64_vec:
            return "float64_vec";
          case param_kind::f64_matrix:
            return "float64_matrix";
          case param_kind::subschema:
            return "subschema";
          case param_kind::discriminated:
            return "discriminated";
          }
          return "unknown";
        };
        nb::list params;
        for (const auto &spec : schema->params) {
          nb::dict entry;
          entry["key"] = spec.key;
          entry["kind"] = kind_name(spec.kind);
          entry["required"] = spec.required;
          if (!spec.subschema.empty())
            entry["subschema"] = spec.subschema;
          if (!spec.discriminator.empty())
            entry["discriminator"] = spec.discriminator;
          params.append(entry);
        }
        return params;
      },
      nb::arg("decoder_name"),
      "Return the registered custom-args parameter schema for a decoder "
      "(list of parameter descriptors), or None if the decoder has not "
      "registered one.");
  mod_cfg.def(
      "registered_decoder_schemas", &registered_decoder_schema_names,
      "Names of all decoders (and nested sections) with registered "
      "custom-args parameter schemas.");
}
} // namespace cudaq::qec::decoding::config
