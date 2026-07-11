/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file decoding_server.cpp
/// @brief Standalone decoding-server process: the service end of a
/// CUDA-Q device_call transport, decoding on the CPU with whatever decoder a
/// YAML config file selects.
///
/// Both the decoder and the transport are configuration, not code:
///   - decoders come from `--config=<yaml>`
///     (multi_decoder_config::from_yaml_str);
///   - the transport comes from `--transport=<name|/path/to/lib.so>`: a CUDA-Q
///     realtime bridge PROVIDER, loaded at runtime through the transport-
///     provider interface (bridge_interface.h).  A bare name resolves to
///     `libcudaq-realtime-bridge-<name>.so` next to the CUDA-Q realtime
///     libraries (udp and cpu_roce ship there); a value containing '/' is
///     loaded verbatim, which is how a partner drops in an out-of-tree
///     transport library with NO changes to this server.
///
/// This server contains no transport-specific code: it forwards all
/// unrecognized command-line arguments to the provider's create() (e.g.
/// --port/--num-slots/--slot-size for udp; --device/--local-ip/--qp_config/
/// --peer-ip/--remote-qp/--frame-size for cpu_roce), derives the dispatcher
/// geometry from the provider's ring-geometry query, publishes readiness from
/// the provider's endpoint-info query, and drives the libcudaq-realtime
/// dispatcher object (HOST path, HOST_CALL table) over the provider's rings.
///
/// The function table comes from the decoding-server-cqr service plugin
/// (enqueue_syndromes / get_corrections / reset_decoder) regardless of
/// transport or decoder.
///
/// Prints `QEC_DECODING_SERVER_READY port=<P> ...` on stdout once the caller
/// can start connecting (the rest of the line is the provider's endpoint
/// description, e.g. `transport=udp` or `transport=cpu_roce roce_ip=<IP>`),
/// and `QEC_DECODING_SERVER_DISPATCHED count=<N>` at shutdown (the
/// two-process stand-in for the in-process
/// cudaqx_qec_device_call_dispatch_count() assertion).
///
/// Usage:
///   decoding_server --config=<decoders.yaml>
///                   [--transport=<name|/path/to/lib.so>] [--timeout=60]
///                   [provider args, forwarded verbatim...]
///
/// NOTE: --slot-size must match the caller channel's slot size (each frame
/// occupies one full slot stride on both wires).
///
/// The dispatch SHAPE is a per-decoder property of the YAML (`dispatch:
/// host|device_graph`): host decoders run through the CQR HOST_CALL
/// dispatcher below; a device_graph decoder routes the whole server through
/// the CQR DecodingServer, whose DeviceGraphTransceiver runs the
/// self-relaunching GPU scheduler over the same kind of runtime-loaded
/// provider.  `--transport=gpu_roce` remains a legacy alias for
/// device_graph-over-hololink.

#include "cudaq/qec/realtime/decoding_config.h"

#include "cudaq/realtime/device_call_service.h"

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#ifdef QEC_HAVE_DEVICE_GRAPH_DISPATCH
// DecodingServer.h (and DeviceGraphTransceiver.h via DecodingServer.cpp) live in
// the decoding-server-cqr directory, added to include paths by CMakeLists when
// CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE is true.
#include "DecodingServer.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern "C" void cudaqx_qec_realtime_device_call_service_force_link();
extern "C" std::uint64_t cudaqx_qec_device_call_dispatch_count();
extern "C" std::uint64_t cudaqx_qec_decoding_server_max_concurrent();
extern "C" void cudaqx_qec_decoding_server_shutdown();

namespace {

namespace config = cudaq::qec::decoding::config;

struct ServerConfig {
  std::string config_path;
  std::string transport = "udp";
  int timeout_sec = 60;
  // Everything the server itself does not consume, forwarded verbatim to the
  // provider's create() (providers ignore arguments they don't recognize, so
  // one command line can carry any provider's options).
  std::vector<std::string> provider_args;
};

bool starts_with(const std::string &s, const char *prefix) {
  const std::size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

bool parse_args(int argc, char **argv, ServerConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::cout << "Usage: " << argv[0]
                << " --config=<decoders.yaml> "
                   "[--transport=<name|/path/to/lib.so>] [--timeout=N] "
                   "[provider args, forwarded verbatim: e.g. --port=N "
                   "--num-slots=N --slot-size=N --device=NAME "
                   "--local-ip=ADDR --qp_config=rendezvous|hsb_fpga "
                   "--peer-ip=ADDR --remote-qp=N --frame-size=N]"
                << std::endl;
      return false;
    } else if (starts_with(a, "--config="))
      cfg.config_path = a.substr(9);
    else if (starts_with(a, "--transport="))
      cfg.transport = a.substr(12);
    else if (starts_with(a, "--timeout="))
      cfg.timeout_sec = std::stoi(a.substr(10));
    else
      cfg.provider_args.push_back(a);
  }
  if (cfg.config_path.empty()) {
    std::cerr << "ERROR: --config=<decoders.yaml> is required" << std::endl;
    return false;
  }
  return true;
}

std::atomic<int> g_shutdown{0};
void on_signal(int) { g_shutdown.store(1, std::memory_order_release); }

// Resolve a --transport value to the provider shared library to load:
// anything with a '/' is a caller-supplied library path (the partner
// drop-in); a bare name maps to libcudaq-realtime-bridge-<name>.so next to
// the CUDA-Q realtime libraries (QEC_BRIDGE_PROVIDER_DIR, baked in by
// CMake), falling back to the bare soname for the dynamic loader's regular
// search path.
std::string resolve_provider_lib(const std::string &transport) {
  if (transport.find('/') != std::string::npos)
    return transport;
  const std::string soname = "libcudaq-realtime-bridge-" + transport + ".so";
#ifdef QEC_BRIDGE_PROVIDER_DIR
  const std::string candidate =
      std::string(QEC_BRIDGE_PROVIDER_DIR) + "/" + soname;
  if (std::ifstream(candidate).good())
    return candidate;
#endif
  return soname;
}

// Publish the rendezvous endpoint for callers/orchestration.  The line is
// `QEC_DECODING_SERVER_READY port=<P> <rest>`: the port token is hoisted to
// the front (test fixtures sscanf "port=%hu" right after the prefix) and the
// rest of the provider's endpoint description follows verbatim.
void print_ready(const std::string &endpoint_info) {
  std::uint16_t port = 0;
  std::string rest;
  std::istringstream in(endpoint_info);
  std::string token;
  while (in >> token) {
    if (starts_with(token, "port="))
      port = static_cast<std::uint16_t>(std::stoul(token.substr(5)));
    else
      rest += (rest.empty() ? "" : " ") + token;
  }
  std::cout << "QEC_DECODING_SERVER_READY port=" << port
            << (rest.empty() ? "" : " ") << rest << std::endl;
  std::cout.flush();
}

} // namespace

int main(int argc, char **argv) {
  ServerConfig cfg;
  if (!parse_args(argc, argv, cfg))
    return 1;

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // [1] Validate the YAML and hand its path to the decoding-server service:
  // the DecodingServer (one DecodingSession worker thread per decoder) builds
  // the decoder instances itself when the dispatch session is created below.
  std::ifstream config_file(cfg.config_path);
  if (!config_file) {
    std::cerr << "ERROR: cannot open config file " << cfg.config_path
              << std::endl;
    return 1;
  }
  std::stringstream config_text;
  config_text << config_file.rdbuf();
  auto decoder_config =
      config::multi_decoder_config::from_yaml_str(config_text.str());
  if (decoder_config.decoders.empty()) {
    std::cerr << "ERROR: no decoders parsed from " << cfg.config_path
              << std::endl;
    return 1;
  }
  ::setenv("CUDAQ_QEC_DECODER_CONFIG", cfg.config_path.c_str(),
           /*overwrite=*/1);
  std::cout << "Configured " << decoder_config.decoders.size()
            << " decoder(s); decoder 0 type: "
            << decoder_config.decoders[0].type
            << "; transport: " << cfg.transport << std::endl;

  // The dispatch SHAPE comes from the decoder config (dispatch:
  // host|device_graph, or the legacy per-decoder transport: key); the wire
  // comes from --transport.  --transport=gpu_roce is kept as a legacy alias
  // for "device_graph dispatch over the built-in hololink provider".
  const bool wants_device_graph = std::any_of(
      decoder_config.decoders.begin(), decoder_config.decoders.end(),
      [](const auto &d) {
        return d.dispatch == config::DecoderDispatch::device_graph;
      });
  const bool legacy_gpu_roce = (cfg.transport == "gpu_roce");
  if (legacy_gpu_roce && !wants_device_graph) {
    // Fail loudly rather than silently running the host path: this exact
    // mismatch (CLI said gpu_roce, YAML defaulted to host) used to select a
    // stub transceiver and die with an unrelated error.
    std::cerr << "ERROR: --transport=gpu_roce (legacy alias for device_graph "
                 "dispatch) but no decoder in "
              << cfg.config_path
              << " declares `dispatch: device_graph` (or the legacy "
                 "`transport: gpu_roce`)"
              << std::endl;
    return 1;
  }

  // [2a] device_graph dispatch takes a different shape (device-side
  // scheduler): bypass the CQR DeviceCallService / HOST_CALL dispatcher and
  // use DecodingServer directly.  Must be checked before force-linking the
  // CQR plugin (which creates a DecodingServer internally for the HOST_CALL
  // path) to avoid double-init.
#ifdef QEC_HAVE_DEVICE_GRAPH_DISPATCH
  if (wants_device_graph) {
    // DecodingServer(config_yaml) reads the YAML, creates the
    // DeviceGraphTransceiver (which loads a bridge provider: the built-in
    // hololink one, or CUDAQ_REALTIME_BRIDGE_LIB), loads decoder sessions,
    // and calls launch_scheduler() to wire the CUDAQ device-graph scheduler
    // to the provider's GPU rings.  The GPU scheduler then handles
    // RX→dispatch→decode→TX autonomously; this thread just waits for signal.
    //
    // Construction throws when the device-graph component is not linked into
    // this binary (no proprietary cudevice archive) or when provider
    // bring-up fails.
    //
    // A non-legacy --transport value selects the provider for the
    // DeviceGraphTransceiver the same way it does for the host path.
    if (!legacy_gpu_roce && cfg.transport != "udp")
      ::setenv("CUDAQ_REALTIME_BRIDGE_LIB",
               resolve_provider_lib(cfg.transport).c_str(), /*overwrite=*/1);
    try {
      cudaq::qec::decoding_server::DecodingServer server(cfg.config_path);
      // QP/rkey/buf already printed to stdout by launch_scheduler() so the
      // orchestration script can grep them before the READY line.
      std::cout << "QEC_DECODING_SERVER_READY gpu_roce" << std::endl;
      std::cout.flush();
      std::thread server_thread([&server] { server.run(); });
      const auto start_time_gr = std::chrono::steady_clock::now();
      while (g_shutdown.load(std::memory_order_acquire) == 0) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time_gr)
                .count();
        if (elapsed > cfg.timeout_sec)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      server.stop();
      server_thread.join();
    } catch (const std::exception &e) {
      std::cerr << "ERROR: device_graph startup failed: " << e.what()
                << std::endl;
      return 1;
    }
    return 0;
  }
#else
  if (wants_device_graph) {
    std::cerr << "ERROR: this server was built without device_graph dispatch "
                 "support (CUDA-Q realtime bridge API or "
                 "cudaq-realtime-dispatch not found at build time)"
              << std::endl;
    return 1;
  }
#endif

  // [2] Pull the QEC HOST_CALL function table from the decoding-server-cqr
  // service plugin -- the same table the in-process host_dispatch test uses.
  cudaqx_qec_realtime_device_call_service_force_link();
  auto pluginInfo = cudaqGetDeviceCallServicePluginInfo();
  if (!pluginInfo.getService) {
    std::cerr << "ERROR: QEC device_call service plugin missing" << std::endl;
    return 1;
  }
  auto *service = pluginInfo.getService();
  if (!service) {
    std::cerr << "ERROR: QEC device_call service create failed" << std::endl;
    return 1;
  }
  // The session owns the function table; keep it alive for the server's
  // lifetime (the dispatcher reads table.entries in place).  Creating it
  // also starts the DecodingServer (decoder construction + one worker thread
  // per decoder) -- before the READY line below, so slow decoder
  // initialization never races the first client request.
  std::unique_ptr<cudaq::realtime::DeviceCallServiceSession> session;
  try {
    session = service->createDispatchSession(
        cudaq::realtime::DeviceCallDispatchMode::Host);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: decoding-server startup failed: " << e.what()
              << std::endl;
    return 1;
  }
  if (!session) {
    std::cerr << "ERROR: QEC device_call service does not support host "
                 "dispatch"
              << std::endl;
    return 1;
  }
  const auto &table = session->dispatchTable();
  if (!table.entries || table.count == 0) {
    std::cerr << "ERROR: QEC host dispatch table unavailable" << std::endl;
    return 1;
  }

  // [3] Load the transport provider and bring the endpoint up.  create()
  // takes the transport to the point where its endpoint identity is known
  // (bound port / QP), so readiness is published before connect() -- which,
  // for rendezvous-style transports, BLOCKS until the caller dials in.
  const std::string provider_lib = resolve_provider_lib(cfg.transport);
  ::setenv("CUDAQ_REALTIME_BRIDGE_LIB", provider_lib.c_str(), /*overwrite=*/1);

  std::vector<char *> provider_argv;
  provider_argv.reserve(cfg.provider_args.size());
  for (auto &a : cfg.provider_args)
    provider_argv.push_back(a.data());

  cudaq_realtime_bridge_handle_t bridge = nullptr;
  if (cudaq_bridge_create(&bridge, CUDAQ_PROVIDER_EXTERNAL,
                          static_cast<int>(provider_argv.size()),
                          provider_argv.data()) != CUDAQ_OK) {
    std::cerr << "ERROR: failed to load/create transport provider '"
              << provider_lib << "' (--transport=" << cfg.transport << ")"
              << std::endl;
    return 1;
  }

  // Dispatcher geometry comes from the provider, not from re-parsed CLI.
  std::uint32_t num_slots = 0, slot_size = 0;
  if (cudaq_bridge_get_ring_geometry(bridge, &num_slots, &slot_size) !=
      CUDAQ_OK) {
    std::cerr << "ERROR: transport provider does not report ring geometry "
                 "(bridge interface v2 required)"
              << std::endl;
    cudaq_bridge_destroy(bridge);
    return 1;
  }

  char endpoint_info[512] = {0};
  if (cudaq_bridge_get_endpoint_info(bridge, endpoint_info,
                                     sizeof(endpoint_info)) != CUDAQ_OK)
    std::snprintf(endpoint_info, sizeof(endpoint_info), "transport=%s",
                  cfg.transport.c_str());
  print_ready(endpoint_info);

  if (cudaq_bridge_connect(bridge) != CUDAQ_OK) {
    std::cerr << "ERROR: transport provider connect() failed" << std::endl;
    cudaq_bridge_destroy(bridge);
    return 1;
  }

  cudaq_ringbuffer_t ringbuffer{};
  if (cudaq_bridge_get_transport_context(bridge, RING_BUFFER, &ringbuffer) !=
      CUDAQ_OK) {
    std::cerr << "ERROR: transport provider has no ring-buffer context"
              << std::endl;
    cudaq_bridge_destroy(bridge);
    return 1;
  }

  // [4] Drive the libcudaq-realtime dispatcher object over the provider's
  // rings.  The dispatch table is HOST_CALL-only, so the HOST-path ring loop
  // runs the inline HOST_CALL path (the dispatcher creates no GRAPH_LAUNCH
  // engine for a table with no graph entries).
  int dispatcher_shutdown = 0;
  std::uint64_t packets_dispatched = 0;

  cudaq_dispatcher_config_t dispatch_config{};
  dispatch_config.num_slots = num_slots;
  dispatch_config.slot_size = slot_size;
  dispatch_config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  dispatch_config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  dispatch_config.kernel_type = CUDAQ_KERNEL_REGULAR;
  dispatch_config.skip_tx_markers = 1;

  cudaq_function_table_t function_table{};
  function_table.entries = table.entries;
  function_table.count = table.count;

  cudaq_dispatch_manager_t *manager = nullptr;
  cudaq_dispatcher_t *dispatcher = nullptr;
  if (cudaq_dispatch_manager_create(&manager) != CUDAQ_OK ||
      cudaq_dispatcher_create(manager, &dispatch_config, &dispatcher) !=
          CUDAQ_OK ||
      cudaq_dispatcher_set_ringbuffer(dispatcher, &ringbuffer) != CUDAQ_OK ||
      cudaq_dispatcher_set_function_table(dispatcher, &function_table) !=
          CUDAQ_OK ||
      cudaq_dispatcher_set_control(dispatcher, &dispatcher_shutdown,
                                   &packets_dispatched) != CUDAQ_OK ||
      cudaq_dispatcher_start(dispatcher) != CUDAQ_OK) {
    std::cerr << "ERROR: dispatcher bring-up failed" << std::endl;
    if (dispatcher)
      cudaq_dispatcher_destroy(dispatcher);
    if (manager)
      cudaq_dispatch_manager_destroy(manager);
    cudaq_bridge_destroy(bridge);
    return 1;
  }

  // Start the provider's I/O loop last, once the dispatcher is polling.
  if (cudaq_bridge_launch(bridge) != CUDAQ_OK) {
    std::cerr << "ERROR: transport provider launch() failed" << std::endl;
    cudaq_dispatcher_stop(dispatcher);
    cudaq_dispatcher_destroy(dispatcher);
    cudaq_dispatch_manager_destroy(manager);
    cudaq_bridge_destroy(bridge);
    return 1;
  }

  // [5] Run until signalled or timed out.
  const auto start_time = std::chrono::steady_clock::now();
  while (g_shutdown.load(std::memory_order_acquire) == 0) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - start_time)
                             .count();
    if (elapsed > cfg.timeout_sec)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // [6] Orderly shutdown: stop the dispatcher (sets the shutdown flag and
  // joins the loop thread), then the transport, then the DecodingServer
  // receive loop (a still-joinable static thread would std::terminate).
  cudaq_dispatcher_stop(dispatcher);
  cudaq_dispatcher_destroy(dispatcher);
  cudaq_dispatch_manager_destroy(manager);
  cudaq_bridge_disconnect(bridge);
  cudaq_bridge_destroy(bridge);
  cudaqx_qec_decoding_server_shutdown();

  std::cout << "QEC_DECODING_SERVER_DISPATCHED count="
            << cudaqx_qec_device_call_dispatch_count() << std::endl;
  // Concurrency evidence for multi-logical-qubit tests: high-water mark of
  // simultaneously-busy DecodingSession workers.
  std::cout << "QEC_DECODING_SERVER_MAX_CONCURRENT_DECODERS count="
            << cudaqx_qec_decoding_server_max_concurrent() << std::endl;
  return 0;
}
