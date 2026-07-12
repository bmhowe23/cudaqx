# One Ring Buffer (and One Dispatcher) per Decoder

Status: prototype working on branch pair
`cuda-quantum:bmh/realtime-bridge-providers` +
`cudaqx:bmh/decoding-server-bridge-670` (2026-07).  Runnable proof:
`app_examples/surface_code-5-per-decoder-rings` (+ its `-test.sh`).

Companion to `transport_provider_design.md` (the transport/bridge side of
the same architecture).

## 1. Topology

```
before:  decoder(N) ──► ring(1) ──► endpoint(1)     payload-demuxed
after:   decoder(1) ──► ring(1) ──► endpoint(1)     device_id == decoder_id
future:  decoder(1) ──► ring(1..k) ──► QP(m)        provider-internal fan-in
```

The mechanism is `cudaq::device_call`'s device-id overload.  The CUDA-Q
device_call runtime keys sessions, channels, rings, and dispatchers by
device id (`DeviceCallDispatch.cpp` `sessions` map; each session owns one
`RingBufferWrapper` and one dispatcher).  Routing every decoding RPC with
`device_id == decoder_id` therefore gives each decoder its own ring and its
own dispatcher for free -- no shared ring, no head-of-line blocking between
decoders, per-decoder backpressure.  The payload keeps its `decoder_id`
argument as a cross-check (and for the fan-in future where several sources
feed one decoder).

## 2. What changed, where

### cuda-quantum (`runtime/internal/device_call/DeviceCallDispatch.cpp`)

1. **Lazy per-device session init.**  `acquireFrameForDevice` now calls the
   idempotent `initializeServiceForDevice(deviceId)` for an unseen device id
   instead of throwing.  (Eager init only ever covered device 0; a nonzero
   device id previously trapped in lowered code as "illegal execution of
   unreachable code".)  External (non-builtin) channels share
   device 0's channel unless the device has its own scoped endpoint args
   (see section 5).
2. **Per-device channel spec.**  `CUDAQ_DEVICE_CALL_CHANNEL` (and
   `--cudaq-device-call[-channel]`) accept `<default>[,<id>=<channel>...]`:

   ```
   CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch,1=device_dispatch
   ```

   `initializeServiceForDevice` resolves the channel per device
   (`channelNameForDevice`), so one process can run one decoder on a
   host_dispatch ring and another on a device_dispatch ring simultaneously.

No compiler changes: the `-frealtime-lowering` pass already threads the
device-id operand (`RealtimeDeviceCall.cpp`, `devcall.getDevice()`) into
`__cudaq_device_call_acquire_realtime_frame(deviceId, ...)`.

### cudaqx (`libs/qec/lib/realtime/simulation-cqr/simulation_cqr_device.cpp`)

The QEC device wrappers route each decoding RPC by decoder:

```cpp
cudaq::device_call(decoder_id, ::enqueue_syndromes, decoder_id, tag,
                   kSyndromeMappingId, bits);
```

(same for `get_corrections` / `reset_decoder`).  The CQR service plugin
needs no change for the host path: every per-device session shares the same
three HOST_CALL handlers, and the decoder registry still routes by the
payload's `decoder_id` -- a decoder's ring simply only ever carries its own
id.

## 3. CPU-memory ring + GPU-memory ring, simultaneously

The channel chosen for a device id decides where its ring lives and what
polls it:

| channel          | ring memory                    | dispatcher                          | plugin session mode |
|------------------|--------------------------------|-------------------------------------|---------------------|
| `host_dispatch`  | `cudaHostAlloc` pinned+mapped  | dispatcher object, host thread      | `Host` (HOST_CALL fn table) |
| `device_dispatch`| pinned+mapped, polled from GPU | persistent GPU dispatch kernel      | `Gpu` (DEVICE_CALL device fn ptrs + `launchFn`) |

So `CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch,1=device_dispatch` is the whole
deployment story: decoder 0 on a CPU ring, decoder 1 on a GPU-polled ring.
The override plumbing is prototype-verified (the example's test script
shows device 1 being steered to `device_dispatch` while device 0 comes up
on its host ring).

**Remaining piece (designed, not implemented): the QEC plugin's Gpu-mode
session.**  `decoding_server_cqr.cpp`'s `createDispatchSession` currently
declines `DeviceCallDispatchMode::Gpu`.  The implementation recipe (all
ingredients exist in-tree):

- table entries: the three proprietary DEVICE_CALL populate shims
  (`cudaqx_qec_realtime_dispatch_populate_*_device_entry`, dlsym'd exactly
  as `DeviceGraphTransceiver::launch_scheduler` does; requires the cudevice
  archives whole-archived + `--export-dynamic` in the app, which
  `qec_realtime_app_link_options` already provides);
- `launchFn = cudaq_launch_dispatch_kernel_regular` (persistent kernel
  launcher from `cudaq-realtime-dispatch.a`), or the graph-dispatch
  launcher for decode-graph decoders -- see `qec_realtime_session.cpp`
  `populate_function_table()` for the DEVICE-mode table shape (N
  GRAPH_LAUNCH + 2 DEVICE_CALL) and `cudaq/unittests/device_call/
  DeviceCallDispatchTester.cpp` for a complete Gpu-mode session example;
- mode/`deviceId`/`synchronizeFn` per `device_call_service.h`;
- session selection: `createDispatchSession(mode)` has no device id, so a
  mixed deployment binds the Gpu session to the YAML decoder with
  `dispatch: device_graph` (unique today).  Passing the device id through
  `createDispatchSession` is the API extension to propose upstream when
  multiple device-graph decoders are needed.

## 4. The runnable example

`app_examples/surface_code-5-per-decoder-rings.cpp`: two pymatching
decoders, one kernel enqueuing a different syndrome to each and reading
each decoder's corrections back over its own ring.  Verified by the test
script:

1. correct per-decoder corrections + device_call dispatch count;
2. `service initialized for device 0` AND `device 1` in the runtime log
   (two sessions == two rings == two dispatchers);
3. the per-device channel override routes device 1 to `device_dispatch`
   (declined by the Host-only plugin today -- fails loudly, no silent
   fallback) while device 0 still comes up on its host ring.

Run: `CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch ./surface_code-5-per-decoder-rings`

## 5. Two-process form (IMPLEMENTED -- this is the product path)

The decoding server opens ONE provider instance per decoder in the YAML:
one endpoint, one ring, one dispatcher each (the dispatch table is shared;
handlers still route by payload decoder_id, and a decoder's ring only ever
carries its own id).  Wire contract:

- READY publishes every ring: `QEC_DECODING_SERVER_READY port=<P0>
  transport=udp ring0=<P0> ring1=<P1> ...` (leading port token = ring 0,
  for existing single-ring consumers).
- Shutdown publishes per-ring proof of traffic:
  `QEC_DECODING_SERVER_RING decoder=<id> dispatched=<n>` (one line per
  ring, before the DISPATCHED total).  NOTE: dispatcher stats flush when
  the loop exits, so counts are sampled after dispatcher_stop.
- Caller: device-scoped external channel args --
  `udp-port=<P0> udp-port.1=<P1>` -- give each device_call session its own
  channel/ring; devices without scoped args share device 0's channel
  (back-compat).

Validated by DecodingServerTwoProcess.TwoProcessPerDecoderRings: two
decoders in one server, two udp endpoints, one caller process; asserts the
kernel decodes correctly AND each ring's dispatched count covers its
decoder's reset/enqueue/get -- proving traffic was per-ring, not payload-
demuxed over one wire.  (Fixed en route: cudaq_bridge_create's
cached-provider path did not register new handles, breaking every second
bridge instance.)

## 6. Follow-ups

- QEC plugin Gpu-mode session (recipe above) => CPU+GPU rings running
  simultaneously end to end.
- Per-decoder rings over cpu_roce two-process (udp validated; cpu_roce
  needs per-ring rendezvous ports wired the same way).
- `createDispatchSession(mode, deviceId)` upstream API extension.
- Fan-in (decoder 1 <- ring 1..k <- QP m): provider-internal QP
  aggregation behind one ring context; `vp_id` in the enqueue_syndromes
  payload for the syndrome mapping table (cudaq-spec
  decoder_server_runtime.md).
