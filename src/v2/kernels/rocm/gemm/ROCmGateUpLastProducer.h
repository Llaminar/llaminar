/**
 * @file ROCmGateUpLastProducer.h
 * @brief Cold original-policy geometry and exact native last-producer identity.
 *
 * This internal boundary creates no allocation or device cache. The existing
 * ROCm MoE owner retains named arena counters and seals each borrowed pointer;
 * the GEMV translation unit remains the source of original split-K policy.
 */
#pragma once
#include "kernels/common/NativeTileCompletion.h"
namespace llaminar2::rocm
{
/** @brief Cold original versus paired-K32 operand consumption; arithmetic stays unchanged. */
enum class GateUpProducerLoads { Original, PairedK32 };
/** @return Calling worker's next cold recording policy, never live device state. */
GateUpProducerLoads currentGateUpProducerLoads() noexcept;
/** @brief Select one diagnostic producer before owner preparation and capture. */
class ScopedGateUpProducerLoads final
{
public:
    /**
     * @param loads Exact implemented load policy for this worker's new prepared owner.
     * @throws std::logic_error If selection occurs during native capture.
     * @throws std::invalid_argument For an unknown policy.
     */
    explicit ScopedGateUpProducerLoads(GateUpProducerLoads loads);
    /** @brief Restore the prior cold selection; retained graph identities stay sealed. */
    ~ScopedGateUpProducerLoads();
    ScopedGateUpProducerLoads(const ScopedGateUpProducerLoads&)=delete;
    ScopedGateUpProducerLoads& operator=(const ScopedGateUpProducerLoads&)=delete;
private:
    GateUpProducerLoads previous_;
};
/**
 * @param columns Original participant-local output width.
 * @param reduction Original H reduction width.
 * @param routes Original admitted route capacity.
 * @param policy_mask Complete immutable original source arithmetic policies.
 * @return Counter shape using the unchanged generated serial-M1 policy maximum.
 * @throws std::invalid_argument For an incomplete original source policy.
 */
NativeTileCompletionPlan gateUpLastProducerPlan(int columns,int reduction,int routes,std::uint32_t policy_mask);
/** @return Exact compiled64-thread original-grid native function, with no device query. */
const void* gateUpLastProducerKernelFunction() noexcept;
/** @param loads Exact sealed load policy. @return Its compiled native entry without a device query. */
const void* gateUpLastProducerKernelFunction(GateUpProducerLoads loads);
}
