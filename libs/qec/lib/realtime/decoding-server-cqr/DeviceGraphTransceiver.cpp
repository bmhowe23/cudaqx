/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifdef CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE

#include "DeviceGraphTransceiver.h"
#include "RpcWireFormat.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/graph_resources.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// CUDAQ device-graph scheduler types (cudaq-realtime-dispatch) and the
// RPCHeader wire struct the provider's --payload-size argument is defined
// against.
#include "cudaq/realtime/hololink_bridge_common.h"

namespace cudaq::qec::decoding_server {

// ---------------------------------------------------------------------------
// Internal helpers (same pattern as hololink_qldpc_graph_decoder_bridge.cpp)
// ---------------------------------------------------------------------------

namespace {

// Allocate \p bytes of CUDA pinned+mapped host memory and return both the host
// pointer and its device-mapped counterpart.  The memory is zero-initialised.
bool alloc_pinned_mapped(size_t bytes, void **host_out, void **dev_out) {
  void *h = nullptr;
  if (cudaHostAlloc(&h, bytes, cudaHostAllocMapped) != cudaSuccess)
    return false;
  void *d = nullptr;
  if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
    cudaFreeHost(h);
    return false;
  }
  std::memset(h, 0, bytes);
  *host_out = h;
  *dev_out = d;
  return true;
}

// Resolve a proprietary DEVICE_CALL populate shim via dlsym and stamp the
// function table entry.  The server process must absorb
// libcudaq-qec-realtime-cudevice-proprietary.a (WHOLE_ARCHIVE) and link with
// --export-dynamic so the symbols are visible.
using populate_fn = void (*)(void *);
bool populate_device_call(cudaq_function_entry_t &entry, const char *symbol,
                          uint32_t function_id) {
  auto fn = reinterpret_cast<populate_fn>(::dlsym(RTLD_DEFAULT, symbol));
  if (!fn) {
    CUDA_QEC_ERROR(
        "DeviceGraphTransceiver: dlsym({}) failed -- the server process must "
        "absorb libcudaq-qec-realtime-cudevice-proprietary.a as WHOLE_ARCHIVE "
        "and link with --export-dynamic",
        symbol);
    return false;
  }
  fn(&entry);
  entry.function_id = function_id;
  entry.routing_key = 0;
  if (entry.dispatch_mode != CUDAQ_DISPATCH_DEVICE_CALL ||
      !entry.handler.device_fn_ptr) {
    CUDA_QEC_ERROR("DeviceGraphTransceiver: {} did not produce a valid "
                   "DEVICE_CALL entry",
                   symbol);
    return false;
  }
  return true;
}

#define GPU_CUDA_CHECK(expr)                                                   \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess)                                                   \
      throw std::runtime_error(                                                \
          std::string("DeviceGraphTransceiver CUDA error: ") +                     \
          cudaGetErrorString(_err) + " (" #expr ")");                          \
  } while (0)

// Parse one `key=` token out of the provider's endpoint-info line (base 0:
// accepts the provider's 0x-hex qp/buffer_addr and decimal rkey alike).
// Leaves \p out untouched when the key is absent.
template <typename T>
void parse_endpoint_token(const std::string &info, const char *key, T &out) {
  std::istringstream in(info);
  std::string token;
  const std::string prefix = std::string(key) + "=";
  while (in >> token)
    if (token.rfind(prefix, 0) == 0) {
      out = static_cast<T>(
          std::stoull(token.substr(prefix.size()), nullptr, 0));
      return;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// DeviceGraphConfig::from_env
// ---------------------------------------------------------------------------

// Each knob reads QEC_DEVICE_GRAPH_<NAME> first and falls back to the legacy
// HOLOLINK_<NAME> spelling (the values are forwarded to whatever transport
// provider is loaded, which need not be Hololink; existing orchestration
// scripts still set the legacy names).
static const char *env_raw(const char *name) {
  const std::string primary = std::string("QEC_DEVICE_GRAPH_") + name;
  if (const char *v = std::getenv(primary.c_str()))
    return v;
  const std::string legacy = std::string("HOLOLINK_") + name;
  return std::getenv(legacy.c_str());
}
static std::string env_str(const char *name, const char *def = "") {
  const char *v = env_raw(name);
  return v ? v : def;
}
static uint32_t env_u32(const char *name, uint32_t def) {
  const char *v = env_raw(name);
  return v ? static_cast<uint32_t>(std::stoul(v)) : def;
}
static int env_int(const char *name, int def) {
  const char *v = env_raw(name);
  return v ? std::stoi(v) : def;
}
static size_t env_size(const char *name, size_t def) {
  const char *v = env_raw(name);
  return v ? static_cast<size_t>(std::stoull(v)) : def;
}

DeviceGraphConfig DeviceGraphConfig::from_env() {
  DeviceGraphConfig c;
  c.device_name = env_str("DEVICE");
  c.peer_ip = env_str("PEER_IP");
  c.remote_qp = env_u32("REMOTE_QP", 0);
  c.gpu_id = env_int("GPU_ID", 0);
  c.frame_size = env_size("FRAME_SIZE", 384);
  c.page_size = env_size("PAGE_SIZE", 0); // 0 → derived below
  c.num_pages = env_size("NUM_PAGES", 64);
  c.reserved_sms = env_int("RESERVED_SMS", 2);
  return c;
}

// ---------------------------------------------------------------------------
// DeviceGraphTransceiver constructor
// ---------------------------------------------------------------------------

DeviceGraphTransceiver::DeviceGraphTransceiver(const DeviceGraphConfig &config)
    : gpu_id_(config.gpu_id) {
  if (config.device_name.empty())
    throw std::runtime_error("DeviceGraphTransceiver: QEC_DEVICE_GRAPH_DEVICE "
                             "(or legacy HOLOLINK_DEVICE) not set");
  if (config.peer_ip.empty())
    throw std::runtime_error("DeviceGraphTransceiver: QEC_DEVICE_GRAPH_PEER_IP "
                             "(or legacy HOLOLINK_PEER_IP) not set");
  if (config.remote_qp == 0)
    throw std::runtime_error("DeviceGraphTransceiver: QEC_DEVICE_GRAPH_REMOTE_QP "
                             "(or legacy HOLOLINK_REMOTE_QP) not set");

  // Derive page_size from frame_size if not overridden, then round up to the
  // 128-byte Hololink granularity.  Mirrors the derivation in
  // hololink_qldpc_graph_decoder_bridge.cpp (lines 279-282).
  size_t page_size = config.page_size ? config.page_size : config.frame_size;
  page_size = (page_size + 127) & ~static_cast<size_t>(127);

  // The provider computes frame_size = sizeof(RPCHeader) + payload_size, so
  // hand it the payload remainder of our frame budget.
  if (config.frame_size < sizeof(cudaq::realtime::RPCHeader))
    throw std::runtime_error(
        "DeviceGraphTransceiver: HOLOLINK_FRAME_SIZE smaller than the RPC header");
  const size_t payload_size =
      config.frame_size - sizeof(cudaq::realtime::RPCHeader);

  // Bring the Hololink transceiver up through the bridge-provider interface:
  // create() = hololink_create_transceiver + hololink_start (3-kernel shape:
  // no --forward / --unified => rx_only + tx_only kernels, with dispatch
  // supplied by our device-graph scheduler in launch_scheduler()).
  const std::vector<std::string> args = {
      "--device=" + config.device_name,
      "--peer-ip=" + config.peer_ip,
      "--remote-qp=" + std::to_string(config.remote_qp),
      "--gpu=" + std::to_string(config.gpu_id),
      "--page-size=" + std::to_string(page_size),
      "--num-pages=" + std::to_string(config.num_pages),
      "--payload-size=" + std::to_string(payload_size),
  };
  std::vector<char *> argv;
  argv.reserve(args.size());
  for (auto &a : args)
    argv.push_back(const_cast<char *>(a.c_str()));

  // Built-in Hololink provider by default; CUDAQ_REALTIME_BRIDGE_LIB selects
  // a replacement provider library (same mechanism as the decoding server's
  // --transport=<path>.so partner drop-in).
  const auto provider = std::getenv("CUDAQ_REALTIME_BRIDGE_LIB")
                            ? CUDAQ_PROVIDER_EXTERNAL
                            : CUDAQ_PROVIDER_HOLOLINK;
  if (cudaq_bridge_create(&bridge_, provider, static_cast<int>(argv.size()),
                          argv.data()) != CUDAQ_OK ||
      !bridge_)
    throw std::runtime_error(
        "DeviceGraphTransceiver: bridge provider create failed for device=" +
        config.device_name + " peer=" + config.peer_ip +
        " (is libcudaq-realtime-bridge-hololink.so on the loader path, and "
        "does the IB netdev have an IPv4 address assigned for RoCE v2 GID?)");

  // Adopt the DOCA ring buffer GPU VRAM pointers from the provider.
  cudaq_ringbuffer_t ring{};
  if (cudaq_bridge_get_transport_context(bridge_, RING_BUFFER, &ring) !=
      CUDAQ_OK) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: provider has no ring-buffer context");
  }
  rx_ring_data_ = ring.rx_data;
  rx_ring_flag_ = ring.rx_flags;
  tx_ring_data_ = ring.tx_data;
  tx_ring_flag_ = ring.tx_flags;
  if (!rx_ring_data_ || !rx_ring_flag_ || !tx_ring_data_ || !tx_ring_flag_) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver: null DOCA ring pointer(s) from provider");
  }

  // Ring geometry from the provider when it supports the v2 query; fall back
  // to the locally-derived values (identical arithmetic) for a v1 provider.
  uint32_t num_slots = 0, slot_size = 0;
  if (cudaq_bridge_get_ring_geometry(bridge_, &num_slots, &slot_size) ==
      CUDAQ_OK) {
    num_pages_ = num_slots;
    page_size_ = slot_size;
  } else {
    num_pages_ = config.num_pages;
    page_size_ = page_size;
  }

  // RDMA target identity for the orchestration handshake (v2 query; a v1
  // provider prints its own '=== Bridge Ready ===' banner in connect() and
  // these stay 0).
  char info[512] = {0};
  if (cudaq_bridge_get_endpoint_info(bridge_, info, sizeof(info)) ==
      CUDAQ_OK) {
    const std::string s(info);
    parse_endpoint_token(s, "qp", qp_number_);
    parse_endpoint_token(s, "rkey", rkey_);
    parse_endpoint_token(s, "buffer_addr", buffer_addr_);
  }

  // connect(): the provider publishes its QP/RKey/Buffer handshake for the
  // orchestration script (no wire traffic; the playback tool alone programs
  // the FPGA control plane).
  if (cudaq_bridge_connect(bridge_) != CUDAQ_OK) {
    cudaq_bridge_destroy(bridge_);
    bridge_ = nullptr;
    throw std::runtime_error("DeviceGraphTransceiver: provider connect() failed");
  }

  CUDA_QEC_INFO("DeviceGraphTransceiver: Hololink provider started  device={} "
                "peer={} gpu={} pages={} page_size={}  "
                "QP=0x{:X} rkey={} buf=0x{:X}  "
                "(call launch_scheduler() before run())",
                config.device_name, config.peer_ip, config.gpu_id, num_pages_,
                page_size_, qp_number_, rkey_, buffer_addr_);
}

// ---------------------------------------------------------------------------
// launch_scheduler
// ---------------------------------------------------------------------------

void DeviceGraphTransceiver::launch_scheduler(void *raw_graph_resources) {
  auto *graph_res =
      static_cast<cudaq::qec::realtime::graph_resources *>(raw_graph_resources);
  if (!graph_res || !graph_res->graph_exec)
    throw std::runtime_error(
        "DeviceGraphTransceiver::launch_scheduler: null graph_exec "
        "(decoder must support_graph_dispatch() and capture_decode_graph())");

  GPU_CUDA_CHECK(cudaSetDevice(gpu_id_));

  void *ft_dev = nullptr;
  if (!alloc_pinned_mapped(3 * sizeof(cudaq_function_entry_t), &ft_host_,
                           &ft_dev))
    throw std::runtime_error("DeviceGraphTransceiver::launch_scheduler: "
                             "function-table pinned alloc failed");

  auto *entries = static_cast<cudaq_function_entry_t *>(ft_host_);
  bool ok =
      populate_device_call(entries[0],
                           "cudaqx_qec_realtime_dispatch_populate_enqueue_"
                           "syndromes_device_entry",
                           kEnqueueSyndromesFunctionId) &&
      populate_device_call(
          entries[1],
          "cudaqx_qec_realtime_dispatch_populate_get_corrections_device_entry",
          kGetCorrectionsFunctionId) &&
      populate_device_call(
          entries[2],
          "cudaqx_qec_realtime_dispatch_populate_reset_decoder_device_entry",
          kResetDecoderFunctionId);
  if (!ok) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver::launch_scheduler: populate_device_call failed "
        "(see error log above)");
  }

  // Resolve dispatch graph API via dlsym; cudaq-realtime-dispatch is linked
  // into the server (not this static lib) to keep the CUDA module in one copy.
  // Signatures must match create/launch/destroy_dispatch_graph_fn_t in
  // qec_realtime_session.cpp/.h exactly — calling-convention mismatch is UB.
  using create_fn_t = cudaError_t (*)(
      volatile std::uint64_t *, volatile std::uint64_t *, std::uint8_t *,
      std::uint8_t *, std::size_t, std::size_t, cudaq_function_entry_t *,
      std::size_t, void *, volatile int *, std::uint64_t *, std::size_t,
      std::uint32_t, std::uint32_t, cudaGraphExec_t, cudaStream_t,
      cudaq_dispatch_graph_context **);
  using launch_fn_t =
      cudaError_t (*)(cudaq_dispatch_graph_context *, cudaStream_t);
  using destroy_fn_t = cudaError_t (*)(cudaq_dispatch_graph_context *);

  auto create_dispatch = reinterpret_cast<create_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_create_dispatch_graph_regular"));
  auto launch_dispatch = reinterpret_cast<launch_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_launch_dispatch_graph"));
  auto destroy_dispatch = reinterpret_cast<destroy_fn_t>(
      ::dlsym(RTLD_DEFAULT, "cudaq_destroy_dispatch_graph"));

  if (!create_dispatch || !launch_dispatch || !destroy_dispatch) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    CUDA_QEC_ERROR(
        "DeviceGraphTransceiver: cudaq dispatch API not found via dlsym -- "
        "the server must link cudaq-realtime-dispatch with --export-dynamic");
    throw std::runtime_error(
        "DeviceGraphTransceiver::launch_scheduler: cudaq dispatch API not found "
        "(cudaq_create/launch/destroy_dispatch_graph_regular); "
        "link cudaq-realtime-dispatch into the server with --export-dynamic");
  }
  fn_destroy_dispatch_graph_ = destroy_dispatch;

  void *sd_host = nullptr, *sd_dev = nullptr;
  if (!alloc_pinned_mapped(sizeof(int), &sd_host, &sd_dev)) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    throw std::runtime_error("DeviceGraphTransceiver::launch_scheduler: "
                             "shutdown-flag pinned alloc failed");
  }
  shutdown_host_ = static_cast<volatile int *>(sd_host);
  shutdown_dev_ = static_cast<volatile int *>(sd_dev);

  if (cudaMalloc(&d_stats_, sizeof(uint64_t)) != cudaSuccess ||
      cudaMemset(d_stats_, 0, sizeof(uint64_t)) != cudaSuccess) {
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    cudaFreeHost(sd_host);
    shutdown_host_ = nullptr;
    shutdown_dev_ = nullptr;
    throw std::runtime_error(
        "DeviceGraphTransceiver::launch_scheduler: d_stats_ alloc failed");
  }

  GPU_CUDA_CHECK(cudaStreamCreate(&sched_stream_));

  cudaError_t cerr = create_dispatch(
      rx_ring_flag_, tx_ring_flag_, rx_ring_data_, tx_ring_data_, page_size_,
      page_size_, static_cast<cudaq_function_entry_t *>(ft_dev),
      /*func_count=*/3,
      /*graph_io_ctx=*/nullptr, shutdown_dev_, d_stats_, num_pages_,
      /*num_blocks=*/1, /*threads_per_block=*/64, graph_res->graph_exec,
      sched_stream_, &sched_ctx_);
  if (cerr != cudaSuccess) {
    cudaStreamDestroy(sched_stream_);
    sched_stream_ = nullptr;
    cudaFree(d_stats_);
    d_stats_ = nullptr;
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    cudaFreeHost(sd_host);
    shutdown_host_ = nullptr;
    shutdown_dev_ = nullptr;
    throw std::runtime_error(
        std::string("DeviceGraphTransceiver::launch_scheduler: "
                    "cudaq_create_dispatch_graph_regular: ") +
        cudaGetErrorString(cerr));
  }

  cerr = launch_dispatch(sched_ctx_, sched_stream_);
  if (cerr != cudaSuccess) {
    fn_destroy_dispatch_graph_(sched_ctx_);
    sched_ctx_ = nullptr;
    cudaStreamDestroy(sched_stream_);
    sched_stream_ = nullptr;
    cudaFree(d_stats_);
    d_stats_ = nullptr;
    cudaFreeHost(ft_host_);
    ft_host_ = nullptr;
    cudaFreeHost(sd_host);
    shutdown_host_ = nullptr;
    shutdown_dev_ = nullptr;
    throw std::runtime_error(
        std::string("DeviceGraphTransceiver::launch_scheduler: "
                    "cudaq_launch_dispatch_graph: ") +
        cudaGetErrorString(cerr));
  }

  // Start the provider's I/O loop (Hololink RX/TX kernels + monitor thread,
  // owned by the provider) now that the scheduler is polling the rings.
  if (cudaq_bridge_launch(bridge_) != CUDAQ_OK) {
    __atomic_store_n(shutdown_host_, 1, __ATOMIC_RELEASE);
    throw std::runtime_error(
        "DeviceGraphTransceiver::launch_scheduler: provider launch() failed");
  }

  CUDA_QEC_INFO("DeviceGraphTransceiver: GPU scheduler launched  "
                "QP=0x{:X} rkey={} buf=0x{:X}  "
                "(3 DEVICE_CALL entries, graph_exec={:p})",
                qp_number_, rkey_, buffer_addr_,
                static_cast<void *>(graph_res->graph_exec));

  // Print RDMA target info to stdout so the orchestration script can grep it.
  // Matches the format in hololink_qldpc_graph_decoder_bridge.cpp lines
  // 441-444.  Values come from the provider's v2 endpoint-info query; with a
  // v1 provider they are 0 here and the provider's own '=== Bridge Ready ==='
  // banner (printed by connect()) is the authoritative copy.
  std::cout << "QP Number: 0x" << std::hex << qp_number_ << std::dec << "\n"
            << "RKey: " << rkey_ << "\n"
            << "Buffer Addr: 0x" << std::hex << buffer_addr_ << std::dec
            << "\n";
  std::cout.flush();
}

// ---------------------------------------------------------------------------
// ITransceiver interface stubs (GPU scheduler handles the data path)
// ---------------------------------------------------------------------------

RxFrame DeviceGraphTransceiver::recv() {
  // The GPU device-graph scheduler handles RX→dispatch→decode→TX autonomously.
  // This method only exists so DecodingServer::run()'s recv loop blocks until
  // shutdown() is called.
  while (!stopped_.load(std::memory_order_acquire))
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  return {}; // shutdown sentinel: empty buf causes the recv loop to exit
}

void DeviceGraphTransceiver::send(const PeerId & /*peer*/, const uint8_t * /*data*/,
                              size_t /*len*/) {
  throw std::logic_error(
      "DeviceGraphTransceiver::send() must not be called: the CUDAQ device-graph "
      "scheduler writes TX responses directly to the Hololink ring buffer");
}

// ---------------------------------------------------------------------------
// shutdown / destructor
// ---------------------------------------------------------------------------

void DeviceGraphTransceiver::shutdown() {
  if (stopped_.exchange(true, std::memory_order_acq_rel))
    return; // already stopped

  // Signal the GPU scheduler kernel to stop its self-relaunch loop.
  if (shutdown_host_)
    __atomic_store_n(shutdown_host_, 1, __ATOMIC_RELEASE);

  // Stop the Hololink RX/TX kernels and join the provider's monitor thread.
  if (bridge_)
    cudaq_bridge_disconnect(bridge_);
}

DeviceGraphTransceiver::~DeviceGraphTransceiver() {
  // Ensure clean shutdown even if the caller omitted shutdown().
  if (!stopped_.exchange(true, std::memory_order_acq_rel)) {
    if (shutdown_host_)
      __atomic_store_n(shutdown_host_, 1, __ATOMIC_RELEASE);
    if (bridge_)
      cudaq_bridge_disconnect(bridge_);
  }

  if (sched_stream_) {
    cudaStreamSynchronize(sched_stream_); // drain the self-relaunch chain
    if (sched_ctx_ && fn_destroy_dispatch_graph_)
      fn_destroy_dispatch_graph_(sched_ctx_);
    cudaStreamDestroy(sched_stream_);
  }

  if (ft_host_)
    cudaFreeHost(ft_host_);
  if (shutdown_host_)
    cudaFreeHost(const_cast<int *>(shutdown_host_));
  if (d_stats_)
    cudaFree(d_stats_);
  if (bridge_)
    cudaq_bridge_destroy(bridge_);
}

} // namespace cudaq::qec::decoding_server

#endif // CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE
