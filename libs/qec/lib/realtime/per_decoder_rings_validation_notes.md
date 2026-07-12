# Per-Decoder Rings / Mixed-Dispatch Server: Validation Notes

Campaign log for `per_decoder_rings_design.md` -- what was run, what it
proved, the dead ends, and the gotchas.  Read this before debugging the
decode-graph pipeline or reproducing the builds; read the design docs for
the architecture and contracts.

## 1. What is validated (2026-07-12, WSL2 laptop, sm_120 GPU)

- In-process per-decoder rings: `surface_code-5-per-decoder-rings` example
  + test script (two sessions/rings/dispatchers, per-device channel
  override routing).
- Two-process per-decoder rings over udp:
  `DecodingServerTwoProcess.TwoProcessPerDecoderRings` asserts >=3
  dispatches on EACH ring.
- Two-process CONTROL with rebuilt nv-qldpc (new ABI): full sc4 app, two
  host nv-qldpc rings, correct decode (`logical_errors=0/2` per decoder).
- Mixed-dispatch server bring-up: ring0 host + ring1 GPU scheduler over
  pinned-udp rings, one READY line, clean teardown when idle.
- The GPU dispatch graph SERVES DEVICE_CALLS OVER THE WIRE two-process:
  raw RPC datagram probes (hand-built RPCHeader, magic 0x43555152, fnv1a
  ids) round-trip reset_decoder and a full-window enqueue_syndromes with
  status=0 against the device ring.
- Device-side fire-and-forget graph launch works on this box (see probe
  notes below), native sm_120 SASS and forced-JIT compute_90 alike.

## 2. Open: the decode-graph wedge (locally reproducible)

After a window-completing enqueue triggers the decode graph, the
scheduler never dispatches again (subsequent RPCs time out; teardown
drain hangs in cudaStreamSynchronize).  Suspects, in order:

1. The TRIGGERED decode graph's instantiation/upload: device-side
   fire-and-forget requires the child instantiated with
   `cudaGraphInstantiateFlagDeviceLaunch` AND uploaded.  Verify what
   nv-qldpc's `capture_decode_graph` and
   `cudaq_create_dispatch_graph_regular` do with the triggered exec.
2. The dispatch graph's tail self-relaunch pattern.
3. The RelayBP decode graph's own execution on this GPU.

First diagnostic step: the dispatch kernel DROPS the device-side
`cudaGraphLaunch` return code -- surface it (pinned error slot or device
printf) in cudaq-realtime-dispatch.

## 3. Probe methodology (and the invalid-probe lesson)

- VALID probe for device graph launch: two-graph form.  Child graph
  (payload) instantiated `DeviceLaunch` + `cudaGraphUpload`ed; trigger
  kernel calling `cudaGraphLaunch(child, cudaStreamGraphFireAndForget)`
  placed INSIDE a device-launched parent graph; parent host-launched.
  Passes here (`payload ran: YES`), including with PTX-only compute_90
  compilation (JIT'd kernels are fine).
- INVALID probe (do not repeat): triggering fire-and-forget from a PLAIN
  kernel.  That is illegal per the CUDA contract (device graph launch is
  only legal from within a device-launched graph) and fails with
  `cudaErrorNotSupported` on EVERY platform -- including target rigs --
  visible only via `cudaGetLastError()` immediately after the trigger
  kernel launch (stream sync reports success).  This briefly produced a
  false "WSL2 platform limitation" diagnosis until a rig run showed the
  same failure.
- Also probed good on this box: GPU spin-polling of CPU-written
  pinned+mapped memory (`ack=1`), plain `DeviceLaunch` instantiation,
  `CUDA_MODULE_LOADING=EAGER` (no effect on any of the above).
- Artifact arch coverage on this box: GPU is sm_120; nv-qldpc plugin +
  cudevice archives + cudaq-realtime-dispatch carry <= sm_100 SASS (+
  PTX).  Ruled out as the wedge cause by the forced-JIT probe.

## 4. Rebuilding the proprietary artifacts against a cudaqx branch

Scratch worktree of the private cuda-qx repo (`bmh/private-device-ring`):
merge the cudaqx branch into the LOCAL `private/main` (the local branch
was AHEAD of the gitlab remote -- check before basing anything on the
remote tip).  The private realtime/pcm/decoder deltas were already
upstreamed to cudaqx main; the pieces that must survive the merge are the
`NV_PRIVATE` CMake hooks (`add_subdirectory(decoders/plugins/
nv-qldpc-decoder)` in libs/qec/lib/CMakeLists.txt, and the
`libs/qec/lib/realtime/private/` proprietary sources present on the local
branch).  Build with the same CUDAQ_DIR/CUDAQ_REALTIME_ROOT as the cudaqx
tree; pluck ONLY binaries (`libcudaq-qec-nv-qldpc-decoder.so`,
`libcudaq-qec-realtime-cudevice-proprietary.a`) back, and repoint
`CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE`.

## 5. Build/runtime gotchas (each cost real time)

- nvq++-compiled TUs (`cudaqx_add_device_code` custom commands) do NOT
  track header dependencies: after changing `decoding_config.h`, touch or
  clean those sources, or stale objects pass old-layout structs across
  the .so boundary (symptom: segfault inside `configure_decoders`).
- `libcudaq-device-call-runtime.so` must NOT re-export transport-archive
  symbols (`cpu_udp_*`): a provider .so carries its own copy and the
  dynamic linker binds across -- silent ODR violation corrupting teardown
  (fixed upstream with `--exclude-libs`).
- `cudaq_bridge_create`'s cached-provider path did not register new
  handles, so every SECOND bridge instance of one provider was unusable
  (fixed; surfaced by the N-ring server).
- Dispatcher stats flush when the loop EXITS: sample per-ring counters
  after `cudaq_dispatcher_stop`, never before.
- Shell hygiene: never `pkill -f <pattern>` where the pattern matches
  your own shell's command string.
