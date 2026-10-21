/**
 * @file AdaptiveACKReadyCancellation.h
 * @brief Test-only native graph splice at the actual validated-ACK/fill→Ready edge.
 *
 * The fixture keeps every captured production node/edge, adds one ordered native
 * cancellation node before Ready, and instantiates once. No runtime hook,
 * alternate source, host value probe or production capability is introduced.
 */
#pragma once
#include "backends/IGPUGraphCapture.h"
#include "kernels/qwen4exp/ImmutableAdaptiveLookupKernels.h"

namespace llaminar2::test::adaptive_native
{
/**
 * @brief Authenticate and splice an uninstantiated original CUDA component graph.
 * @param graph Actual flat graph captured through the production kernel bundle.
 * @param expected Complete original launch operands, retained by the fixture.
 * @param stream Exact non-default native stream owning that capture.
 * @throws std::exception Before mutation for foreign/ambiguous nodes, operands or edges.
 * @note The graph and every operand owner must outlive all executable launches.
 */
void spliceCUDAAdaptiveACKReadyCancellation(IGPUGraphCapture &graph,
                                           const qwen4exp::AdaptiveLookupDeviceArgs &expected, void *stream);
/** @copydoc spliceCUDAAdaptiveACKReadyCancellation */
void spliceROCmAdaptiveACKReadyCancellation(IGPUGraphCapture &graph,
                                           const qwen4exp::AdaptiveLookupDeviceArgs &expected, void *stream);
} // namespace llaminar2::test::adaptive_native
