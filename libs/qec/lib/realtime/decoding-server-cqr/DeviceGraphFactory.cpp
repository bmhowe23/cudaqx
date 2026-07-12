/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Strong definition of the GPU RoCE factory that DecodingServer.cpp declares
// weakly.  This translation unit lives in
// cudaq-qec-decoding-server-device-graph (NOT the core library) so that only
// binaries linking that component carry the DOCA / Hololink / CUDA-driver
// dependencies.  Consumers must link the component WHOLE_ARCHIVE: the sole
// reference to this symbol is weak, which does not pull archive members on its
// own.

#include "DeviceGraphTransceiver.h"

extern "C" cudaq::qec::decoding_server::ITransceiver *
cudaqx_qec_make_device_graph_transceiver() {
  using namespace cudaq::qec::decoding_server;
  return new DeviceGraphTransceiver(DeviceGraphConfig::from_env());
}
