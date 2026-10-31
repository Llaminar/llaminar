/**
 * @file PooledACKReadyCancellation.h
 * @brief Setup-only authenticated splice before the pooled source Ready authorization.
 *
 * One diagnostic node changes only the original canonical controller. Native
 * provider, ACK, owner copy, grant and retirement operations remain production.
 */
#pragma once
#include "backends/IGPUGraphCapture.h"
#include "kernels/qwen4exp/PooledImmutableSourceOps.h"
namespace llaminar2::test::pooled_ack {
/** @brief Fixed diagnostic graph variant replaying the actual original missing-authorization behavior. */
enum class AuthorizationReplay {Production,OriginalUnchecked};
/** @brief Authenticate one original captured authorization node and add a controller-only predecessor. */
void spliceCUDA(IGPUGraphCapture&,const qwen4exp::pooled_source::DeviceArgs&,void* exact_stream,AuthorizationReplay);
/** @copydoc spliceCUDA */
void spliceROCm(IGPUGraphCapture&,const qwen4exp::pooled_source::DeviceArgs&,void* exact_stream,AuthorizationReplay);
}
