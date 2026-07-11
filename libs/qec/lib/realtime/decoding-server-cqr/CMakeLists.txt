# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Handle both standalone (libs/qec) and full-repo builds.
if(CUDAQX_QEC_STANDALONE_BUILD)
  set(_core_include_dir "${CMAKE_SOURCE_DIR}/../core/include")
else()
  set(_core_include_dir "${CMAKE_SOURCE_DIR}/libs/core/include")
endif()

# ---------------------------------------------------------------------------
# Optional: GPU RoCE transport via the CUDA-Q realtime bridge-provider
# interface.
#
# The Hololink transceiver sits behind a runtime-loaded provider
# (libcudaq-realtime-bridge-hololink.so, built in cuda-quantum where
# HSB + DOCA are provisioned), so compiling this path needs only the CUDA-Q
# realtime headers, libcudaq-realtime.so (the bridge loader + C API), and the
# CUDA toolkit -- NO Hololink / DOCA / HSB build-time dependencies here.  The
# provider .so is resolved at runtime (dlopen), so a build without it still
# produces a server whose gpu_roce mode fails loudly at startup.
#
# cudaq-realtime-dispatch (the device-graph scheduler shims resolved via
# dlsym) is found here for the server target's link; see
# tools/decoding-server/CMakeLists.txt.
# ---------------------------------------------------------------------------
set(CUDAQ_GPU_ROCE_AVAILABLE FALSE)

if(CUDAQ_REALTIME_INCLUDE_DIR)
  if(CUDAQ_REALTIME_LIBRARY)
    get_filename_component(_cudaq_rt_lib_dir "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
  else()
    get_filename_component(_cudaq_rt_lib_dir "${CUDAQ_REALTIME_INCLUDE_DIR}/../lib" ABSOLUTE)
  endif()
  find_library(CUDAQ_QEC_REALTIME_BRIDGE_API_LIBRARY
    NAMES cudaq-realtime
    HINTS "${_cudaq_rt_lib_dir}"
    NO_DEFAULT_PATH)

  # cudaq-realtime-dispatch: declares cudaq_create_dispatch_graph_regular,
  # cudaq_launch_dispatch_graph, cudaq_destroy_dispatch_graph, cudaq_function_entry_t.
  find_library(CUDAQ_REALTIME_DISPATCH_LIBRARY
    NAMES cudaq-realtime-dispatch
    HINTS "${_cudaq_rt_lib_dir}"
    NO_DEFAULT_PATH)

  if(CUDAQ_QEC_REALTIME_BRIDGE_API_LIBRARY AND CUDAQ_REALTIME_DISPATCH_LIBRARY
      AND CUDAToolkit_FOUND)
    set(CUDAQ_GPU_ROCE_AVAILABLE TRUE CACHE INTERNAL
      "GPU RoCE transport compiled into cudaq-qec-decoding-server")
    # CMP0126 (NEW): the plain set(... FALSE) above shadows the cache entry in
    # this directory scope; set the normal variable too so gates in THIS file
    # and its includers see TRUE (same fix as PR 670).
    set(CUDAQ_GPU_ROCE_AVAILABLE TRUE)
    message(STATUS
      "cudaq-qec-decoding-server: GPU RoCE ENABLED "
      "(bridge provider resolved at runtime; api=${CUDAQ_QEC_REALTIME_BRIDGE_API_LIBRARY})")
  else()
    message(STATUS
      "cudaq-qec-decoding-server: GPU RoCE DISABLED "
      "(missing one or more of: libcudaq-realtime "
      "cudaq-realtime-dispatch CUDA toolkit)")
  endif()
else()
  message(STATUS
    "cudaq-qec-decoding-server: GPU RoCE DISABLED "
    "(CUDAQ_REALTIME_INCLUDE_DIR not set)")
endif()

message(STATUS
  "cudaq-qec-decoding-server: CPU RoCE DISABLED (skeleton only; ibverbs implementation pending)")

# ---------------------------------------------------------------------------
# Core decoding server (always built): transport abstraction, session registry,
# per-decoder worker threads, RPC dispatcher.  No dependency on CUDAQ_REALTIME.
# ---------------------------------------------------------------------------
add_library(cudaq-qec-decoding-server STATIC
  RoundAccumulator.cpp
  DecodingSession.cpp
  SessionRegistry.cpp
  RpcDispatcher.cpp
  DecodingServer.cpp
  CpuRoceTransceiver.cpp
  GpuRoceTransceiver.cpp
)

target_compile_features(cudaq-qec-decoding-server PUBLIC cxx_std_20)

target_include_directories(cudaq-qec-decoding-server
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${_core_include_dir}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaq-qec-decoding-server
  PUBLIC
    # The decode-server core only needs the decoder interface + logger (in
    # cudaq-qec-decoders) and the realtime decoding config -- NOT the heavy
    # cudaq-qec (which pulls in libcustabilizer etc.). Keeping cudaq-qec here
    # violates the realtime-server dependency-closure contract.
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
)

if(CUDAQ_GPU_ROCE_AVAILABLE)
  target_compile_definitions(cudaq-qec-decoding-server PRIVATE CUDAQ_GPU_ROCE_AVAILABLE)
  target_include_directories(cudaq-qec-decoding-server PRIVATE
    "${CUDAQ_REALTIME_INCLUDE_DIR}"
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  # Only the bridge C API (loader + cudaq_bridge_*) and the CUDA runtime; the
  # Hololink/DOCA chain lives inside the provider .so loaded at runtime.
  target_link_libraries(cudaq-qec-decoding-server PRIVATE
    "${CUDAQ_QEC_REALTIME_BRIDGE_API_LIBRARY}"
    CUDA::cudart
  )
endif()

set_target_properties(cudaq-qec-decoding-server PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  POSITION_INDEPENDENT_CODE ON
)

install(TARGETS cudaq-qec-decoding-server
  COMPONENT qec-lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ---------------------------------------------------------------------------
# CQR host-dispatch plugin (only when CUDAQ device_call headers are available):
# CqrTransceiver bridges DeviceCallService handler callbacks to ITransceiver,
# then hands off to DecodingServer for session management and decoding.
# ---------------------------------------------------------------------------
if(CUDAQ_QEC_DEVICE_CALL_SERVICE_AVAILABLE)
  add_library(cudaq-qec-realtime-decoding-server-cqr SHARED
    decoding_server_cqr.cpp
  )

  target_compile_options(cudaq-qec-realtime-decoding-server-cqr
    PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

  target_include_directories(cudaq-qec-realtime-decoding-server-cqr
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CUDAQ_REALTIME_INCLUDE_DIR}
      ${CUDAQ_DEVICE_CALL_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_link_options(cudaq-qec-realtime-decoding-server-cqr PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,libcudaq-qec-decoding-server.a>
  )

  target_link_libraries(cudaq-qec-realtime-decoding-server-cqr
    PUBLIC
      cudaq-qec-realtime-decoding
    PRIVATE
      cudaq-qec-decoding-server
      cudaq::cudaq-common
      ${CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY}
  )

  set(_server_cqr_build_rpath "${CMAKE_BINARY_DIR}/lib")
  set(_server_cqr_install_rpath
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  if(_CUDAQ_REALTIME_LIB_DIR)
    list(APPEND _server_cqr_build_rpath "${_CUDAQ_REALTIME_LIB_DIR}")
    list(APPEND _server_cqr_install_rpath "${_CUDAQ_REALTIME_LIB_DIR}")
  endif()
  if(_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR)
    list(APPEND _server_cqr_build_rpath "${_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR}")
    list(APPEND _server_cqr_install_rpath
      "${_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR}")
  endif()

  set_target_properties(cudaq-qec-realtime-decoding-server-cqr PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    BUILD_RPATH "${_server_cqr_build_rpath}"
    INSTALL_RPATH "${_server_cqr_install_rpath}"
  )

  install(TARGETS cudaq-qec-realtime-decoding-server-cqr
    COMPONENT qec-lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
