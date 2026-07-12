# Decoding-Server Transport-Provider Design

Status: prototype complete and validated on branch pair
`cuda-quantum:bmh/realtime-bridge-providers` +
`cudaqx:bmh/decoding-server-bridge-670` (2026-07).
This document is written so the work can be reproduced from scratch by
reading it: it records the before/after architecture, the API contracts, the
commit-by-commit change plan, and the validation gates.

## 1. Problem

Before this work, `libs/qec/tools/decoding-server/decoding_server.cpp` was a
catalog of transports with a decoder attached:

- Every wire (`udp`, `cpu_roce`, `gpu_roce`) was a compiled-in branch behind
  its own `#ifdef`, CLI flags, and CMake detection block.
- The CPU RoCE TCP rendezvous protocol (a byte-exact `RendezvousInfo` struct
  "must match CpuRoceChannel byte-for-byte") and the HSB FPGA QP handshake
  were duplicated inline in the server -- a silent cross-repo drift hazard.
- `GpuRoceTransceiver` fused the Hololink transport bring-up with the QEC
  device-graph dispatch engine, so building the gpu_roce path required
  Holoscan-Sensor-Bridge, DOCA, hololink, and libcuda at *link time*.
- The server hand-rolled a `std::thread` around
  `cudaq_host_ring_dispatch_loop`, an API surface cuda-quantum PR 4869
  retires in favor of the dispatcher object.
- Transport selection had two knobs that had to agree: `--transport` on the
  CLI and a per-decoder `transport: cpu_roce|gpu_roce` YAML key.  When they
  disagreed, a stub transceiver was silently selected and died with an
  unrelated error.
- A partner transport library could not be integrated without adding
  code to this repository.

## 2. Goal

The decoding server should treat the wire the way it already treats
decoders: **as configuration**.  A partner ships one shared library; the
server changes zero lines.

## 3. Architecture after

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    decoding_server (transport-blind)                     │
│  YAML `dispatch:` picks the ENGINE     --transport picks the WIRE        │
│   host ─► dispatcher object over       <name>  ► libcudaq-realtime-      │
│           provider rings (4869 API)              bridge-<name>.so        │
│   device_graph ─► DecodingServer /     /path.so ► partner library,       │
│           DeviceGraphTransceiver                 verbatim, zero changes  │
│  geometry + READY line derived FROM the provider (v2 queries)            │
└──────────────────────────────────────────────────────────────────────────┘
     │ links                            │ links (weak factory, WHOLE_ARCHIVE)
┌────────────────────┐   ┌────────────────────────────────────────────────┐
│ CQR plugin         │   │ DecodingServer core · device-graph component:  │
│ (HOST_CALL table)  │   │ DeviceGraphTransceiver = dispatch ENGINE only  │
└────────────────────┘   └────────────────────────────────────────────────┘
     │ links (the ONLY realtime link dependency)
┌──────────────────────────────────────────────────────────────────────────┐
│  libcudaq-realtime.so — bridge loader (iface v2) + dispatcher object     │
└──────────────────────────────────────────────────────────────────────────┘
     ◌ dlopen ─────────────── runtime plug-in seam ────────────── ◌ dlopen
┌────────────┬──────────────────┬─────────────────┬────────────────────────┐
│ bridge-udp │ bridge-cpu-roce  │ bridge-hololink │ partner transport .so  │
│ .so        │ .so (rendezvous  │ .so (built on   │ (out of tree, ~9 C     │
│            │ + hsb_fpga)      │ the HSB rig)    │ functions)             │
└────────────┴──────────────────┴─────────────────┴────────────────────────┘
```

Design rules that fall out of the CUDA constraints:

- **Anything that must be device-linked or inlined stays in static archives
  inside the server's single device-link unit** (the dispatch kernel in
  `cudaq-realtime-dispatch.a`, the proprietary cudevice DEVICE_CALL entries,
  `CUDA_SEPARABLE_COMPILATION` + `CUDA_RESOLVE_DEVICE_SYMBOLS` +
  `--export-dynamic` + dlsym populate shims).  CUDA device function pointers
  are module-scoped; they cannot cross a dlopen boundary.
- **Anything that only moves bytes goes behind the provider `.so`.**  The
  bridge seam passes pointers (ring addresses, strides, QP identity) once at
  setup; the per-message data plane is copy-identical to the pre-refactor
  code (handlers still operate in place on RX/TX slots).

## 4. The contracts (API details)

### 4.1 Bridge provider interface (cuda-quantum, `bridge_interface.h`, v2)

A provider is one shared library exporting
`cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface()`
with:

```
version                (int; v2)
create(handle*, argc, argv)     bring-up to the point where the endpoint
                                identity is known (bind/QP), BEFORE any
                                blocking peer wait; parses its own args and
                                IGNORES unrecognized ones (callers forward
                                their whole arg list)
destroy(handle)
get_transport_context(handle, RING_BUFFER|UNIFIED, out)
connect(handle)                 may BLOCK on the peer (e.g. TCP rendezvous
                                accept); hsb_fpga-style no-peer methods make
                                this a no-op
launch(handle)                  start the provider's I/O loop/threads
disconnect(handle)
-- v2 fields (read only from providers reporting version >= 2) --
get_cpu_dataplane(handle, out)  unified single-thread loop shape; NULL if
                                unsupported
get_endpoint_info(handle, buf, len)   one-line `key=value ...` description
                                (port=, roce_ip=, qp=0x.., rkey=,
                                buffer_addr=0x..); valid after create()
get_ring_geometry(handle, num_slots*, slot_size*)
```

Loader rules (`bridge_interface_api.cpp`): accepts provider versions in
`[1, CURRENT]`; fields past `disconnect` are read only from v2+; missing
capability => `CUDAQ_ERR_UNSUPPORTED` (new status value).  Providers load
via `CUDAQ_PROVIDER_HOLOLINK` (builtin soname) or `CUDAQ_PROVIDER_EXTERNAL`
+ `CUDAQ_REALTIME_BRIDGE_LIB=<path>`.  Version compatibility is
asymmetric BY DESIGN: new core runs old providers; an old core rejects a v2
provider loudly at load ("expected 1, got 2") -- core and providers upgrade
together across the v1->v2 boundary.

Providers implemented (cuda-quantum `realtime/lib/daemon/bridge/`):

- `udp/` -> `libcudaq-realtime-bridge-udp.so`: wraps the udp ring
  transceiver.  Args: `--port=` (0=ephemeral) `--num-slots=` `--slot-size=`.
  create=bind, connect=no-op, launch=start threads.
- `cpu_roce/` -> `libcudaq-realtime-bridge-cpu-roce.so`: wraps the CPU RoCE
  transceiver AND both QP-exchange methods.  `--qp_config=rendezvous`
  (default): create=setup()+TCP listen, connect=accept+byte-exact
  RendezvousInfo swap+connect (the wire struct now lives next to
  CpuRoceChannel -- one copy).  `--qp_config=hsb_fpga`: create=one-shot
  start + the canonical `=== Bridge Ready ===` banner (strict-regex format
  from hololink_bridge_common.h, printed BEFORE any consumer readiness
  line), connect=no-op.  Extra args: `--device= --local-ip= --peer-ip=
  --remote-qp= --frame-size=`.  Gated on libibverbs at build.
- `hololink/` (pre-existing provider, extended): v2 endpoint-info
  (qp/rkey/buffer_addr/peer_ip) + ring geometry.  Only compiles with
  HSB+DOCA; must be build-verified on the rig.

### 4.2 Decoding server (cudaqx)

```
decoding_server --config=<decoders.yaml>
                [--transport=<name|/path/to/lib.so>]   default udp
                [--timeout=N]
                [everything else forwarded verbatim to provider create()]
```

- `--transport` name -> `libcudaq-realtime-bridge-<name>.so` resolved in
  QEC_BRIDGE_PROVIDER_DIR (baked at CMake time: the dir of
  libcudaq-realtime.so); a value containing '/' is loaded verbatim (partner
  drop-in).  The server setenv's CUDAQ_REALTIME_BRIDGE_LIB and uses the
  EXTERNAL provider path.
- Host path flow: CQR plugin session (decoders built BEFORE readiness) ->
  bridge create -> geometry query -> `QEC_DECODING_SERVER_READY port=<P>
  <endpoint info>` (port token hoisted first; test fixtures sscanf it) ->
  connect (may block on peer) -> ring context -> dispatcher object
  (create/set_ringbuffer/set_function_table/set_control/start;
  HOST_CALL-only table => no CUDA touched) -> bridge launch -> run ->
  ordered teardown -> `QEC_DECODING_SERVER_DISPATCHED count=` +
  `..._MAX_CONCURRENT_DECODERS count=`.
- Device path: YAML `dispatch: device_graph` on any decoder routes the whole
  server through the CQR `DecodingServer`; READY sentinel
  `QEC_DECODING_SERVER_READY device_graph`.

### 4.3 Dispatch shape vs wire (the two-axis split)

Per-decoder YAML key `dispatch: host | device_graph` (`DecoderDispatch`)
names HOW RPCs execute; `--transport` names the wire.  They are orthogonal:

- `host`: HOST_CALL through the CQR plugin dispatcher; any provider.
- `device_graph`: `DeviceGraphTransceiver` (decoding-server-cqr) -- the GPU
  device-graph dispatch ENGINE, transport-blind.  It consumes a bridge
  provider itself (builtin hololink by default; `--transport`/
  CUDAQ_REALTIME_BRIDGE_LIB overrides), REQUIRES v2 geometry+endpoint
  queries, configures via `QEC_DEVICE_GRAPH_{DEVICE,PEER_IP,REMOTE_QP,
  FRAME_SIZE,PAGE_SIZE,NUM_PAGES,GPU_ID,RESERVED_SMS}` env, and wires
  `cudaq_create/launch/destroy_dispatch_graph_regular` (dlsym) + the
  proprietary populate shims onto the provider's GPU rings.  Lives in the
  `cudaq-qec-decoding-server-device-graph` STATIC component behind PR 670's
  weak factory `cudaqx_qec_make_device_graph_transceiver` (WHOLE_ARCHIVE
  consumers; link canary target).  Gate: `CUDAQ_QEC_DEVICE_GRAPH_AVAILABLE`
  = realtime headers + libcudaq-realtime + cudaq-realtime-dispatch + CUDA
  toolkit.  NO HSB/DOCA/ibverbs at build time (verified: no such DT_NEEDED
  in the server binary).

There are deliberately NO legacy aliases (all-new component this cycle): no
`transport:` YAML key, no `--transport=gpu_roce`, no HOLOLINK_* env
fallback, no CpuRoceTransceiver stub.  YAML alias parsing note for anyone
re-adding compatibility keys: a second `mapOptional` onto the same field
must use the two-arg form, or an absent key resets the value to the default.

## 5. Reproduction plan (what an agent should do)

1. cuda-quantum worktree at PR 4869 head merged with origin/main (merge was
   clean).  Build realtime + full CUDA-Q to a FRESH prefix (e.g.
   /usr/local/cudaq-4869) via cudaqx scripts/build_cudaq_with_realtime.sh
   (CUDAQ_SRC must be a dir literally named `cudaq`).  Verify cudaqx main
   already contains the CUDA-Q pin bump compatible with 4869 (PR 654/675).
2. cuda-quantum commits, in order: (a) bridge interface v2 + loader rules +
   CUDAQ_ERR_UNSUPPORTED + fix caching-before-validation; (b) udp provider
   (+ reorder lib/ subdirs: cpu_transport before daemon); (c) cpu_roce
   provider (absorb rendezvous + hsb_fpga from the cudaqx server);
   (d) hololink provider v2 queries (syntax-check only without HSB/DOCA).
3. cudaqx branch off main: rewrite the server per 4.2 (drop all transport
   code + host-dispatch/ibverbs/transport link deps; dispatcher object;
   keep READY/DISPATCHED contracts byte-compatible).
4. Merge PR 670 (component split + weak factory + CMP0126 fix), then
   refactor GpuRoceTransceiver into DeviceGraphTransceiver as a bridge
   consumer (4.3), then the dispatch/transport split, then remove aliases.
5. Point the cudaqx build at the new prefix:
   `-DCUDAQ_DIR=<prefix>/lib/cmake/cudaq -DCUDAQ_REALTIME_ROOT=<prefix>`;
   pass `-DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE=<...>.a` to
   enable the device-graph component end to end.

## 6. Validation gates

- Two-process udp suite (`test_decoding_server`): 5/5, including dual
  decoders over one ring.
- `test_decoding_server_core`: 5/5; pymatching realtime: 6/6.
- Negative: `--transport=/nonexistent.so` and `--transport=bogusname` fail
  with the resolved path in the message; device_graph without the hololink
  provider .so fails at provider load with an actionable error; missing
  QEC_DEVICE_GRAPH_* env fails at config validation.
- Dependency hygiene: `objdump -p decoding_server | grep NEEDED` shows no
  doca/hololink/ibverbs.
- Known pre-existing failure (NOT caused by this work):
  PyMatchingDeviceCallRealtime.HostDispatch "null dispatch session" fails
  identically on untouched main + the old install.

## 7. Follow-ups (recorded, not done)

- Rig validation: build cudaq-realtime-bridge-hololink from the branch
  (verifies the v2 edits compile against real HSB headers) and run the HSB
  script end to end.
- Upstream the interface-v2 commit to cuda-quantum with the version-pairing
  rule called out; propose HOST_CALL handler-context (`host_fn(ctx, ...)`)
  and multi-writer-ring slot discipline for the fan-in topology.
- Top-level `transport:` YAML section + SIGHUP/RPC session-epoch reload.
- One ring dataplane per decoder via `cudaq::device_call(device_id ==
  decoder_id, ...)` (per-device sessions already exist in the cuda-quantum
  runtime); provider-internal QP fan-in for one-to-many decoder:QP
  topologies; `vp_id` in the enqueue_syndromes payload for the syndrome
  mapping table (cudaq-spec decoder_server_runtime.md).
- Public-surface cleanup in cudaq-realtime: deprecate the raw ring loop,
  graph_launch_engine surface, transport wrapper headers, and hololink
  headers; replace the `cudaq*` version-script wildcard with an explicit
  export list.
