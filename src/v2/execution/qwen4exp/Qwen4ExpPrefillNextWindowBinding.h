/**
 * @file Qwen4ExpPrefillNextWindowBinding.h
 * @brief Cold immutable borrow of the original admitted prefill request/chunk binding.
 */
#pragma once
#include "backends/DeviceId.h"
#include "execution/local_execution/graph/RequestInputPositionAdmission.h"
#include "kernels/qwen4exp/Qwen4ExpPrefillNextTokenWindow.h"
namespace llaminar2 {struct DevicePrefillChunkGraphBinding;}
namespace llaminar2::qwen4exp {
/** @brief Declaration only; it creates no source, buffer, count or execution permission. */
class Qwen4ExpPrefillNextWindowBinding final {
public:
    /**
     * @brief Borrow exact original prefill addresses before capture.
     * @param original Same admitted request/chunk binding used by the materializer.
     * @param consumer Actual GPU participant that owns the request banks.
     * @param nextBucket Next shape selected by the original retained chunk schedule.
     * @param layout Immutable retained-family position requirement, enforced anew at every original ingress.
     * @throws std::invalid_argument For incomplete, foreign or oversized declarations.
     */
    static Qwen4ExpPrefillNextWindowBinding fromOriginalChunk(
        const DevicePrefillChunkGraphBinding& original,DeviceId consumer,int nextBucket,
        RequestInputPositionLayoutRequirement layout);
    /** @return Unchanged borrowed native inputs, never their live values. */
    const Qwen4ExpPrefillNextTokenWindow& native()const noexcept{return window_;}
    /** @return Exact original participant. */
    DeviceId device()const noexcept{return device_;}
    /** @return Complete original declaration plus next-window identity. */
    std::uint64_t captureIdentity()const noexcept{return identity_;}
private:
    /** @brief Only the original-binding factory constructs this cold borrow. */
    Qwen4ExpPrefillNextWindowBinding(Qwen4ExpPrefillNextTokenWindow w,DeviceId d,std::uint64_t id)
        :window_(w),device_(d),identity_(id){}
    Qwen4ExpPrefillNextTokenWindow window_;
    DeviceId device_;
    std::uint64_t identity_;
};
} // namespace llaminar2::qwen4exp
